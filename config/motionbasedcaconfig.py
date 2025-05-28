from argparse import ArgumentParser

# Motion prediction Based Cross-Attention Config
class MotionBasedCAConfig(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.input_configs = self.add_argument_group('input')
        self.input_configs.add_argument('--batch_size', default=16, type=int)
        self.input_configs.add_argument('--num_workers', default=4, type=int)
        self.input_configs.add_argument('--input_seq_len', default=6, type=int)
        self.input_configs.add_argument('--output_seq_len', default=10, type=int)
        self.input_configs.add_argument('--dataroot', default= ..., type=str) ### need to input
        self.input_configs.add_argument('--fps', default=2, type=int)
        self.input_configs.add_argument('--sample_points', default=4096, type=int)
        
        # the setting of dataset.
        self.dataset_configs = self.add_argument_group('dataset')
        self.dataset_configs.add_argument('--DatasetWithSeg', default=False, type=bool, help='Use dataset with segmentation')
        self.dataset_configs.add_argument('--DatasetWithAug', default=False, type=bool, help='Use dataset with augmentation')
        self.dataset_configs.add_argument('--Dataset', default=False, type=bool, help='Use the original dataset')

        # the name of N_k, etc.. is confusing, so we explained it with --help
        self.motion_configs = self.add_argument_group('motion_prediction')
        self.motion_configs.add_argument('--dropout', default=0.5, type=float)
        self.motion_configs.add_argument('--motion_hidden_dim', default=256, type=int)
        self.motion_configs.add_argument('--N_k', type=int, default=4, help='Number of cross-attention repeats')
        self.motion_configs.add_argument('--N_a', type=int, default=4, help='Number of some parameter A')
        self.motion_configs.add_argument('--N_b', type=int, default=8, help='Number of some parameter B')
        self.motion_configs.add_argument('--N_c', type=int, default=8, help='Number of transformer layers for pose')
        self.motion_configs.add_argument('--N_d', type=int, default=8, help='Number of some parameter D')
        self.motion_configs.add_argument('--N_e', type=int, default=8, help='Number of transformer layers for traj')
        self.motion_configs.add_argument('--d_z', type=int, default=128)
        self.motion_configs.add_argument('--end_hidden_dim', type=int, default=256)
        self.motion_configs.add_argument('--motion_label_dim', type=int, default=5, help='Dimension of motion labels')

        self.train_configs = self.add_argument_group('train')
        self.train_configs.add_argument('--save_path', type=str, default=...) ### need to input
        self.train_configs.add_argument('--save_fre', type=int, default=1)
        self.train_configs.add_argument('--val_fre', type=int, default=3)
        self.train_configs.add_argument('--load_cross_attention_model_dir', type=str, default=None)
        self.train_configs.add_argument('--load_optim_dir', type=str, default=None)
        self.train_configs.add_argument('--train_dimension',type=int, default=128)
        self.train_configs.add_argument('--epoch', type=int, default=1000)
        self.train_configs.add_argument('--lr', type=float, default=3e-4)
        self.train_configs.add_argument('--weight_decay', type=float, default=1e-4)
        self.train_configs.add_argument('--gamma', type=float, default=0.98)

        self.eval_configs = self.add_argument_group('eval')
        self.eval_configs.add_argument('--output_path', default='results', type=str)
        self.eval_configs.add_argument('--smplx_path', default= ..., type=str) ### need to input
        self.eval_configs.add_argument('--vposer_path', default= ..., type=str) ### need to input

    def get_configs(self):
        return self.parse_args()


if __name__ == '__main__':
    config = MotionBasedCAConfig()
    print(config.get_configs())