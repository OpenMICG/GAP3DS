import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_cross_model import CrossAttentionLayer
from model.pointnet_plus2 import PointNet2SemSegSSGShape
from model.gcn import GCN
import torch.nn.init as init
import numpy as np

class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=4, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim * expansion)
        self.fc2 = nn.Linear(out_dim * expansion, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = self.drop(F.gelu(self.fc1(x)))
        out = self.drop(self.fc2(out))
        return out


class PoseEncoder(nn.Module):
    def __init__(self, f_p=128):
        super().__init__()
        self.f_p = f_p
        self.mlp = Mlp(in_dim=69, out_dim=self.f_p, expansion=2, drop=0.1)

    def forward(self, x):
        B, T, J, C = x.shape
        x = x.reshape(B, T, J * C)
        out = self.mlp(x)
        return out


class TrajEncoder(nn.Module):
    def __init__(self, f_r=64):
        super().__init__()
        self.f_r = f_r
        self.mlp = Mlp(in_dim=3, out_dim=self.f_r, expansion=2, drop=0.1)

    def forward(self, traj):
        B, T, _, C = traj.shape
        traj = traj.reshape(B, T, C)
        out = self.mlp(traj) 
        return out


class TransformerFilling(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        out = self.transformer_decoder(tgt, memory, tgt_mask, src_mask)
        return out


class EndingMotionGenerator(nn.Module):
    def __init__(self, d_in=64, d_hidden=128, d_out=69):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, z):
        B, _ = z.shape
        out = self.mlp(z)     
        out = out.view(B, 23, 3)
        return out

def InteractExtract(interact_poses, interact_dest):
    bs, _ = interact_poses.reshape(-1, 1).shape
    motion_length = 120
    dest_motion_length = 16
    dest_motion_len_index = [12, 24, 36, 48, 60, 72, 84, 96, 108, 119]
    dest_interact_poses_list = []
    for bsi in range(bs):
        interact_pose = interact_poses[bsi].reshape(motion_length, 22, 3)
        zero_pad = np.zeros((motion_length, 1, 3))  # if the mdm base on the smpl model, you need to use smplx
        interact_pose = np.concatenate([interact_pose, zero_pad], axis=1)
        interact_pose = interact_pose[dest_motion_len_index,]
        interact_dest_bsi = interact_dest[bsi, ].cpu().numpy()
        interact_pose = interact_pose - interact_pose[:,[0],:] + interact_dest_bsi
        dest_interact_poses_list.append(interact_pose)
    dest_interact_poses = np.stack(dest_interact_poses_list, axis=0)
    return torch.from_numpy(dest_interact_poses).float()

class DualPromptedMotionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.input_seq_len  = config.input_seq_len  # T
        self.output_seq_len = config.output_seq_len # ΔT
        self.seq_len = self.input_seq_len + self.output_seq_len
        self.joint_num = 23
        self.dimension = config.train_dimension
        
        # ========== Module Definitions ==========

        self.pose_encoder = PoseEncoder(f_p=self.dimension)
        self.scene_encoder = PointNet2SemSegSSGShape({'feat_dim': self.dimension})
        self.traj_encoder = TrajEncoder(f_r=self.dimension)

        self.f_s_cross_att = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=1, num_heads=1)
            for _ in range(config.N_k)
        ])
        self.xs_cross_att = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=self.dimension, num_heads=1)
            for _ in range(config.N_a)
        ])
        self.ts_cross_att = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=self.dimension, num_heads=1)
            for _ in range(config.N_b)
        ])
        self.fm_cross_att = nn.ModuleList([
            CrossAttentionLayer(num_q_channels=self.dimension, num_kv_channels=self.dimension, num_heads=1)
            for _ in range(config.N_b)
        ])
        
        # 5) Motion Type Output & Embedding
        self.motion_type_mlp = Mlp(in_dim=self.dimension , out_dim=config.motion_label_dim, expansion=2, drop=0.1)
        self.label_to_z = nn.Linear(config.motion_label_dim, config.d_z)  # (B, d_label) -> (B, d_z)

        self.ending_gen = EndingMotionGenerator(
            d_in=config.d_z,
            d_hidden=config.end_hidden_dim,
            d_out=69,  # 23×3
        )

        # 7) Pose Decoder and Traj Decoder
        self.pose_ref = TransformerFilling(d_model=self.dimension, nhead=8, num_layers=config.N_c)
        self.traj_decoder = TransformerFilling(d_model=self.dimension,  nhead=8, num_layers=config.N_e)

        # 8) General input/output projections
        self.pose_input_proj  = nn.Linear(69, self.dimension, bias=False).to(self.device)
        self.pose_output_proj = nn.Linear(self.dimension, 69, bias=False).to(self.device)

        self.traj_input_mlp   = nn.Linear(self.dimension, self.dimension, bias=False).to(self.device)
        self.traj_output_mlp  = nn.Linear(self.dimension, 3, bias=False).to(self.device)

        # generation and decoding
        self.motion_gen = GCN(config, node_n=69)
        self.motion_decoder = GCN(config, node_n=69)
    
    def forward(self, joints, scene_xyz, gazes, **kwargs):
        
        B, T, J, C = joints.shape
        _, N, _ = scene_xyz.shape
        assert T == self.input_seq_len, "The number of input frames must match config.input_seq_len"
        
        interact_dest = joints[:,-1, 0,:].clone()
        
        if 'interact_poses' in kwargs: # if you use the mdm to generate the poses.
            interact_poses = kwargs['interact_poses'][: ,0]
            future_mask_joints = InteractExtract(interact_poses,interact_dest).to(self.device)
        else:  # if not.
            future_mask_joints = torch.zeros((B, self.output_seq_len, J, C), device=self.device)
        
        X_masked = torch.cat([joints, future_mask_joints], dim=1)  # (B, T+ΔT, 23, 3)


        F_pose = self.pose_encoder(joints) 
        fp_features, bottleneck_feats = self.scene_encoder(scene_xyz.repeat(1,1,2)) # encode the scene.
        scene_feats = fp_features.transpose(1, 2)
        F_traj_1 = self.traj_encoder(gazes)
        F_traj_last = F_traj_1[:,[-1],:].repeat(1,10,1)
        F_traj = torch.cat([F_traj_1,F_traj_last],dim = 1)
        
        if "interaction_location" in kwargs: interaction_location = kwargs['interaction_location'] # if you use the gazenet to find the interaction_location
        else: interaction_location = gazes[:, 0, 0, :]
        
        dist = torch.norm(scene_xyz - interaction_location.unsqueeze(1), dim=-1, keepdim=True)
        dist_min, dist_max = dist.min(dim=1, keepdim=True)[0], dist.max(dim=1, keepdim=True)[0] 
        distance_map = (dist - dist_min) / (dist_max - dist_min + 1e-8)  # (B, N, 1)
        distance_map = 1 - distance_map
        
    
        for layer in self.f_s_cross_att:
            scene_feats = layer(scene_feats, distance_map)

        F_pose_enhenced = F_pose.clone()
        for layer in self.xs_cross_att:
            F_pose = layer(F_pose, scene_feats)  # (B, T+ΔT, 128)
            
        for layer in self.ts_cross_att:
            F_traj = layer(F_traj, scene_feats)  # (B, T, 64)
            
            
        F_pose = F_pose[:,6:,:]
        F_pose = torch.cat([F_pose_enhenced,F_pose],dim= 1)
                    
        
        motion_logits = self.motion_type_mlp(F_pose)
        motion_logits_last = motion_logits[:, -1, :]
        motion_label = torch.argmax(motion_logits_last, dim=-1)
        z = self.label_to_z(motion_logits_last) 

        ending_motion_3d = self.ending_gen(z)
        X_masked[:, -1, :, :] = ending_motion_3d[:]

        X_masked_flat = X_masked.reshape(B, self.seq_len, -1)
        X_embed = self.pose_input_proj(X_masked_flat).transpose(0, 1)

        pred_embed = self.pose_ref(X_embed, X_embed)
        pred_embed = pred_embed.transpose(0, 1)               # (B, T, 128)
        pred_flat  = self.pose_output_proj(pred_embed)  
        pred_motion = pred_flat.reshape(B, self.seq_len, self.joint_num, 3)


        H = scene_xyz[..., [1]]
        phi, _ = torch.max(H, dim=1, keepdim=True)
        phi /= 2
        M = (H > phi).float()
        F_SM = scene_feats * (1 - M)
        
        for layer in self.fm_cross_att:  
            F_traj = layer(F_traj, F_SM)
        
        F_traj = F_traj[:,6:,:]
        F_STS = torch.cat([F_traj_1,F_traj],dim = 1)

        Traj_embed = self.traj_input_mlp(F_STS).transpose(0, 1)
        pred_traj_embed = self.traj_decoder(Traj_embed, Traj_embed) 
        pred_traj_embed = pred_traj_embed.transpose(0, 1) 
        pred_traj = self.traj_output_mlp(pred_traj_embed)
        pred_traj = pred_traj.reshape(B, self.seq_len, 1, 3)
        predictions = pred_motion.clone()
        predictions[:, :6, :, :] = joints
        distance = predictions[:,:,[0],:] - pred_traj
        predictions =  predictions - distance
        predictions[:, :, [0], :] = pred_traj
        
        predictions = self.motion_gen(predictions)
        predictions[:, :6, :, :] = joints
        predictions = self.motion_decoder(predictions)
        
        return {
            "Motion":predictions,
            "Motion Label":motion_label
        }