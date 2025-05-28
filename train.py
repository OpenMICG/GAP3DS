import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from utils.logger import create_logger, MetricTracker
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

class Motion_evalutor():
    def __init__(self, config, model):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gal_model = GazeguidedAffordanceLearner(config,).to(self.device)
        self.model = model(config, ).to(self.device)
        self.DatasetWithSeg = self.config.DatasetWithSeg
        self.DatasetWithAug = self.config.DatasetWithAug
        self.Dataset = self.config.Dataset
        self.DiffusedPoses = True
    
        assert self.DatasetWithSeg + self.DatasetWithAug + self.Dataset == 1, "Only one of DatasetWithSeg, DatasetWithAug, and Dataset can be True"
        
        if self.config.model_base == 'CrossAttention' and  self.config.load_cross_attention_model_dir is not None:
            state_dict = torch.load(self.config.load_cross_attention_model_dir)
            state_dict = {k: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print('load done!')
            
        elif self.config.model_base == 'Tranformer' and  self.config.load_transformer_model_dir is not None:
            state_dict = torch.load(self.config.load_transformer_model_dir)
            state_dict = {k: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print('load done!')

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)
        
        if self.DatasetWithSeg:
            from dataset import gimo_with_seg
            self.train_dataset = gimo_with_seg.EgoEvalDataset(config, train=True)
            self.test_dataset = gimo_with_seg.EgoEvalDataset(config, train=False)
            self.pose_prompt_dataset = gimo_with_seg.PosePromptDataset()
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
                collate_fn=custom_collate_fn
            )
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
            self.train_dataset = gimo_with_augment.EgoEvalDataset(config, train=True)
            self.test_dataset = gimo_with_augment.EgoEvalDataset(config, train=False)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                drop_last=True,
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=9,
                shuffle=False,
                num_workers=4,
                drop_last=True,
            )
        else: 
            from dataset import gimo_dataset_with_motion_label
            self.train_dataset = gimo_dataset_with_motion_label.EgoEvalDataset(config, train=True)
            self.test_dataset = gimo_dataset_with_motion_label.EgoEvalDataset(config, train=False)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                drop_last=True,
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=9,
                shuffle=False,
                num_workers=4,
                drop_last=True,
            )      
        self.label_loss_fn = nn.CrossEntropyLoss()
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path, exist_ok=True)
        self.logger = create_logger(config.save_path)
        
        self.best_metrics = {k: float('inf') for k in ['Traj-path', 'Traj-interaction', 'MPJPE', 'MPJPE-interaction']}
        self.best_epoch = -1
        self.metric_keys = ['Traj-path', 'Traj-interaction', 'MPJPE', 'MPJPE-interaction']

    def train(self):
        train_metrics = MetricTracker('Traj-path', 'Traj-interaction', 'MPJPE', 'MPJPE-interaction')
        print(f"you are running the GAP3DS based {self.config.model_base}!")
        
        for epoch in range(self.config.epoch):
            for data in tqdm(self.train_loader):
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
                    
                if self.DatasetWithSeg:
                    visual_affordance, textual_affordance = self.gal_model(gazes, scene_instances)
                
                output = self.model(joints_input[:, :, :23], scene_points, gazes, **kwargs)
                joints_predict = output['Motion']

                loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

                loss = loss_trans_gcn.mean() + loss_des_trans_gcn + mpjpe_gcn.mean() + des_mpjpe_gcn
                
                if "Motion Label" in output:
                    pred_motion_label = output["Motion Label"]
                    B = joints_label.shape[0]
                    predict_label = torch.zeros(B, 5).cuda()
                    for i in range(B):
                        predict_motion = torch.zeros(5).cuda()
                        predict_motion[pred_motion_label[i]] = 1
                        predict_label[i] = predict_motion
                    loss_label = self.label_loss_fn(predict_label.float(), motion_label)
                    loss += loss_label

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_metrics.update("Traj-path", loss_trans_gcn[:, self.config.input_seq_len:].mean(), gazes.shape[0])
                train_metrics.update("Traj-interaction", loss_des_trans_gcn.mean(), gazes.shape[0])
                train_metrics.update("MPJPE", mpjpe_gcn[:, self.config.input_seq_len:].mean(), gazes.shape[0])
                train_metrics.update("MPJPE-interaction", des_mpjpe_gcn, gazes.shape[0])

            self.lr_adjuster.step()
            train_metrics.log(self.logger, epoch, train=True)
            train_metrics.reset()

            if epoch % self.config.val_fre == 0:
                self.model.eval()
                with torch.no_grad():
                    self.test(epoch)
                self.model.train()

            # if epoch % self.config.save_fre == 0:
            #     torch.save(self.model.state_dict(), f"{self.config.save_path}/{epoch}.pth")


    def test(self, epoch):
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
                if self.DatasetWithSeg and self.DiffusedPoses: 
                    interact_poses = self.pose_prompt_dataset.get_interaction_poses(index)
                    kwargs = {'interact_poses': interact_poses}
                else:
                    kwargs = dict()
                
                output = self.model(joints_input[:, :, :23], scene_points, gazes, **kwargs)
                joints_predict = output['Motion']

                loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])
                
                test_metrics.update("Traj-path", loss_trans_gcn[:, self.config.input_seq_len:].mean(), gazes.shape[0])
                test_metrics.update("Traj-interaction", loss_des_trans_gcn, gazes.shape[0])
                test_metrics.update("MPJPE", mpjpe_gcn[:, self.config.input_seq_len:].mean(), gazes.shape[0])
                test_metrics.update("MPJPE-interaction", des_mpjpe_gcn, gazes.shape[0])

            current_metrics = test_metrics.result()
            if self.is_better(current_metrics, self.best_metrics):
                self.best_metrics = current_metrics.copy()
                self.best_epoch = epoch
                # Optionally, log or save the model here
                torch.save(self.model.state_dict(), f"{self.config.save_path}/best.pth")
                    
            test_metrics.log(self.logger, epoch, train=False)
            test_metrics.reset()

    def calc_loss_gcn(self, poses_predict, poses_label, poses_input):
        poses_label = torch.cat([poses_input, poses_label], dim=1)
        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)
        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1)
        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()
    
    def is_better(self, current, best):
        better = 0
        worse = 0
        for key in self.metric_keys:
            if current[key] < best[key]:
                better += 1
            elif current[key] > best[key]:
                worse += 1
        return True if better > worse else False
    
    def show_best_result(self):
        print(self.best_metrics)


if __name__ == '__main__':
    model_based_list = ['CrossAttention', 'Transformer']
    model_based_select = model_based_list[0]

    if model_based_select == "CrossAttention":
        from config.motionbasedcaconfig import MotionBasedCAConfig
        config = MotionBasedCAConfig().parse_args()
        config.model_base = "CrossAttention"
        from model.DPM import DualPromptedMotionDecoder
        model = DualPromptedMotionDecoder
        
        
    evaluator = Motion_evalutor(config, model)
    evaluator.train()
    evaluator.show_best_result()