import torch
import torch.nn as nn
from model.base_cross_model import Mlp, TransformerEncoder, PositionwiseFeedForward, TransformerDecoder
from model.pointnet_plus2 import PointNet2SemSegSSGShape
from model.AffordanceNet.utils.builder import build_model
import numpy as np
import trimesh

# PS: afforancenet and scene parser need pre-trained.

# if you want to generate motion based on text.
# if not, you can generate motion based on action (that means action type)
affordance_to_action = {
    'grasp': 'the man grasps the object',
    'contain': 'the man puts something into the object',
    'lift': 'the man lifts the object up',
    'openable': 'the man opens the object',
    'layable': 'the man lies down on the object',
    'sittable': 'the man sits on the object',
    'support': 'the man leans against the object',
    'wrap_grasp': 'the man wraps fingers around the object',
    'pourable': 'the man pours something from the object',
    'move': 'the man moves the object',
    'display': 'the man look at the object on display',
    'pushable': 'the man pushes the object',
    'pull': 'the man pulls the object',
    'listen': 'the man puts the object to the ear and listen',
    'wear': 'the man wears the object',
    'press': 'the man presses the object with a finger',
    'cut': 'the man cut something using the object',
    'stab': 'the man stabs something with the object',
}

# ---------------- Module Definitions ----------------

class GazeNet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.linear_att = PositionwiseFeedForward(d_in=d_in, d_hid=4 * d_in)
        self.ln = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(0.1)
        self.mlp = Mlp(in_dim=d_in, out_dim=..., expansion=4, drop=0.1)

    def forward(self, x):
        residual = x
        x = self.linear_att(x)
        x = self.dropout(x)
        x = self.ln(x + residual)
        return self.mlp(x)

class ObjectRetrival(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.linear_att = PositionwiseFeedForward(d_in=d_in, d_hid=4 * d_in)
        self.ln = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(0.1)
        self.mlp = Mlp(in_dim=d_in, out_dim=..., expansion=2, drop=0.1)

    def forward(self, x):
        residual = x
        x = self.linear_att(x)
        x = self.dropout(x)
        x = self.ln(x + residual)
        return self.mlp(x)

class SceneParser(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder for CVPR2024 SoftGroup SceneParser implementation, or use the data we have processed.
        return

class GazeguidedAffordanceLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = config.output_seq_len
        self.seq_len = self.input_seq_len + self.output_seq_len
        self.distributed_train_affordance = False # whether you have trained the affordancenet on more than one gpu.
        self.npoints_together = 4096
        self.SaveAffordancePath = ""
        
        self.sceneparser = SceneParser()
        self.affordancenet = build_model().to(self.device)

        # Load affordance model weights
        if self.distributed_train_affordance:
            state_dict = torch.load(...) # the path of trained affordancenet model.
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.affordancenet.load_state_dict(new_state_dict)
        else:
            self.affordancenet.load_state_dict(torch.load(".../model.t7")) # the path of trained affordancenet model.

        self.gazenet = None  
        self.Count = 0
        # Placeholder for gaze network with external annotations, following Harmonizing Stochasticity and Determinism: Scene-responsive Diverse Human Motion Prediction

    def affordance_inference(self, model, instances):
        # we trained affordancenet as the official implementation
        with torch.no_grad():
            model.eval()
            visual_afford_pred_list = []
            textual_affordance_list = []
            for instance_dict in instances:
                point_cloud = next(iter(instance_dict.values()))  
                centroid = point_cloud.mean(dim=0, keepdim=True)  
                centered_point_cloud = point_cloud - centroid 
                instance = centered_point_cloud.unsqueeze(0).permute(0, 2, 1).float()
                afford_pred = torch.sigmoid(model(instance)) 
                afford_pred = afford_pred.permute(0, 2, 1).contiguous()
                score = afford_pred.squeeze(0) 
                avg_score_per_class = score.mean(dim=0)
                most_likely_affordance = torch.argmax(avg_score_per_class).item()
                visual_afford_pred_list.append(afford_pred)
                textual_affordance_list.append(affordance_to_action[list(affordance_to_action.keys())[most_likely_affordance]])

                points = point_cloud.cpu().numpy()  # [N, 3]
                affordance_map = afford_pred[0, :, 0].cpu().numpy()  # [N]
                affordance_map = (affordance_map - affordance_map.min()) / (affordance_map.max() - affordance_map.min())
                colors = np.zeros((points.shape[0], 4)) 
                colors[:, 0] = affordance_map
                colors[:, 1] = 1 - affordance_map 
                colors[:, 2] = 0 
                colors[:, 3] = 1.0
                
                pcd = trimesh.PointCloud(points, colors=colors)
                save_path = f".../affordance_vis_{self.Count}.ply"
                pcd.export(save_path)
                self.Count += 1
                
        return visual_afford_pred_list, textual_affordance_list
    
    def save_affordance(self, visual_affordance, textual_affordance):
        import numpy as np
        # Save textual affordance
        textual_affordance_path = f"{self.SaveAffordancePath}/textual_affordance_{self.Count}.txt"
        with open(textual_affordance_path, "w", encoding="utf-8") as f:
            for item in textual_affordance:
                f.write(item + "\n")
        processed_visual_affordance = []
        for pc in visual_affordance:
            npoints = pc.shape[1]
            if npoints < self.npoints_together:
                pad_size = self.npoints_together - npoints
                padded_pc = torch.nn.functional.pad(pc, (0, 0, 0, pad_size, 0, 0), mode='constant', value=0)
            elif npoints > self.npoints_together:
                padded_pc = pc[:, :self.npoints_together, :]
            else:
                padded_pc = pc
            processed_visual_affordance.append(padded_pc)
        visual_affordance_array = torch.stack(processed_visual_affordance).cpu().numpy()
        visual_affordance_path = f"{self.SaveAffordancePath}/visual_affordance_{self.Count}.npy"
        np.save(visual_affordance_path, visual_affordance_array)
        self.Count += 1

    def forward(self, gaze, scene_instances):
        gaze = gaze.to(self.device)
        scene_instances = [{k: torch.from_numpy(v).to(self.device) for k, v in d.items()} for d in scene_instances]
        gaze_center = gaze[:, [0], :, :]  # [B, 1, 1, 3]

        all_avg_distances = []
        all_instances = []
        batch_size = gaze_center.shape[0]

        for b in range(batch_size):
            current_gaze = gaze_center[b, 0, 0, :].unsqueeze(0)  # [1, 3]
            scene_dict = scene_instances[b]
            avg_distances = {}

            for instance_key, point_cloud in scene_dict.items():
                distances = torch.norm(point_cloud - current_gaze, dim=1)
                avg_distance = torch.mean(distances)
                avg_distances[instance_key] = avg_distance

            target_key = min(avg_distances, key=avg_distances.get)
            all_avg_distances.append(avg_distances[target_key])
            all_instances.append({target_key: scene_dict[target_key]})

        visual_affordance, textual_affordance = self.affordance_inference(self.affordancenet, all_instances)
        self.save_affordance(visual_affordance, textual_affordance)
        # with the viual_affordance and text_afforance, you can train the mdm by yourself, or just use the textual_affordance to ues the pretrained mdm.
        return visual_affordance, textual_affordance