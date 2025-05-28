import torch
from torch.utils.data import DataLoader
from dataset import gimo_with_seg
from utils.logger import MetricTracker
from model.GAL import GazeguidedAffordanceLearner

def custom_collate_fn(batch):
    # we define the custom collate_fn functions because each object instance's number of point is different.
    gazes, poses_input, poses_label, joints_input, joints_label, scene_points, motion_label, indices, scene_instances = zip(*batch)
    gazes = torch.stack([torch.as_tensor(g) for g in gazes])
    poses_input = torch.stack([torch.as_tensor(p) for p in poses_input])
    poses_label = torch.stack([torch.as_tensor(p) for p in poses_label])
    joints_input = torch.stack([torch.as_tensor(j) for j in joints_input])
    joints_label = torch.stack([torch.as_tensor(j) for j in joints_label])
    scene_points = torch.stack([torch.as_tensor(s) for s in scene_points])
    motion_label = torch.stack([torch.as_tensor(m) for m in motion_label])
    indices = torch.tensor(indices)
    scene_instances = list(scene_instances)
    return (gazes, poses_input, poses_label, joints_input, joints_label, scene_points, motion_label, indices, scene_instances)

class Evalutor():
    def __init__(self, config, model):
        self.config = config
        self.DatasetWithSeg = self.config.DatasetWithSeg
        self.DatasetWithAug = self.config.DatasetWithAug
        self.Dataset = self.config.Dataset
        self.DiffusedPoses = True
        assert self.DatasetWithSeg + self.DatasetWithAug + self.Dataset == 1, "Only one of DatasetWithSeg, DatasetWithAug, and Dataset can be True"

        if self.DatasetWithSeg:
            from dataset import gimo_with_seg
            self.test_dataset = gimo_with_seg.EgoEvalDataset(config, train=False)
            self.pose_prompt_dataset = gimo_with_seg.PosePromptDataset()
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=9,
                shuffle=False,
                num_workers=4,
                drop_last=True,
                collate_fn=custom_collate_fn
            )
        elif self.DatasetWithAug:
            from dataset import gimo_with_augment
            self.test_dataset = gimo_with_augment.EgoEvalDataset(config, train=False)
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=9,
                shuffle=False,
                num_workers=4,
                drop_last=True,
            )
        else: 
            from dataset import gimo_dataset_with_motion_label
            self.test_dataset = gimo_dataset_with_motion_label.EgoEvalDataset(config, train=False)
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=9,
                shuffle=False,
                num_workers=4,
                drop_last=True,
            ) 
            
        self.gal_model = GazeguidedAffordanceLearner(config,).to(self.device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model(config, ).to(self.device)
        self.model = self.model.to(self.device)

    def eval(self):
        
        assert self.config.load_model_dir is not None
        print('loading pretrained model from ', self.config.load_model_dir)
        state_dict = torch.load(self.config.load_model_dir)
        state_dict = {k: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        print('load done!')

        with torch.no_grad():
            self.model.eval()
            test_metrics = MetricTracker('Traj-path', 'Traj-interaction', 'MPJPE', 'MPJPE-interaction')

            for i, data in enumerate(self.test_loader):
                if self.DatasetWithSeg:
                    gazes, poses_input, poses_label, joints_input, joints_label, scene_points, motion_label, index, scene_instances = data
                elif self.DatasetWithAug:
                    gazes, poses_input, poses_label, joints_input, joints_label, scene_points, motion_label = data
                else: 
                    gazes, poses_input, poses_label, joints_input, joints_label, scene_points, motion_label, index = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)
                motion_label = motion_label.to(self.device)
                
                if self.DatasetWithSeg:
                    interact_poses = self.pose_prompt_dataset.get_interaction_poses(index)
                    kwargs = {'interact_poses': interact_poses}
                else: 
                    kwargs = dict()
                    
                output = self.model(joints_input[:, :, :23], scene_points, gazes, kwargs)
                
                joints_predict = output['Motion']
                
                loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                    self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])
                    
                test_metrics.update("Traj-path", loss_trans_gcn[:, self.config.input_seq_len:].mean(), gazes.shape[0])
                test_metrics.update("Traj-interaction", loss_des_trans_gcn, gazes.shape[0])
                test_metrics.update("MPJPE", mpjpe_gcn[:, self.config.input_seq_len:].mean(), gazes.shape[0])
                test_metrics.update("MPJPE-interaction", des_mpjpe_gcn, gazes.shape[0])

            print(test_metrics.result())
            test_metrics.reset()

    def calc_loss_gcn(self, poses_predict, poses_label, poses_input):
        poses_label = torch.cat([poses_input, poses_label], dim=1)
        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)
        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1) 
        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()


if __name__ == '__main__':
    model_based_list = ['CrossAttention']
    model_based_select = model_based_list[0]

    if model_based_select == "CrossAttention":
        from config.motionbasedcaconfig import MotionBasedCAConfig
        config = MotionBasedCAConfig().parse_args()
        from model.DPM import DualPromptedMotionDecoder
        model = DualPromptedMotionDecoder
        
    evaluator = Evalutor(config, model)
    evaluator.eval()