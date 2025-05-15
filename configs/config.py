import json
from pathlib import Path


class Config:
    """Config class

    Set using HPC to true in order to use appropriate paths for HPC
    """

    def __init__(self, remote, dataset=None, existing_model_tag=None, data_dir=None):

        # overwrite settings with arguments provided
        self.dataset = dataset if dataset is not None else 'Sonicom' # 'SONICOM'
        self.data_dir = data_dir if data_dir is not None else '/data/' + self.dataset

        if existing_model_tag is not None:
            self.start_with_existing_model = True
        else:
            self.start_with_existing_model = False

        self.existing_model_tag = existing_model_tag if existing_model_tag is not None else None

        # Data processing parameters
        self.gen_sofa_flag = True
        self.nbins_hrtf = 128  # make this a power of 2
        self.train_samples_ratio = 0.8
        self.hrir_samplerate = 48000.0
        self.normalize_input = False
        self.domain = 'magnitude_db'
        self.max_degree = 21
        self.max_num_points = 72 * 12
        self.hrtf_loader = "hrtfdata"

        # Data dirs
        if remote:
            self.data_dir_path = 'C:/Users/steph/Desktop/output'
            self.raw_hrtf_dir = Path('C:/Users/steph/Desktop/HRTF Datasets')
            self.amt_dir = 'C:/Users/steph/Desktop/amtoolbox-1.6.0'
        else:
            self.data_dir_path = '/Users/Downloads/shtHRTF'
            self.raw_hrtf_dir = Path('/Users/Downloads')

        self.result_folder = '/results'
        self.existing_model_path = f'{self.data_dir_path}{self.result_folder}/{self.existing_model_tag}'

        self.valid_recon_path = f'{self.data_dir_path}{self.result_folder}/valid_recon'
        self.valid_target_path = f'{self.data_dir_path}{self.data_dir}/valid_target'
        self.model_path = f'{self.data_dir_path}{self.result_folder}'

        self.baseline_dir = '/baseline_results/' + self.dataset
        self.barycentric_hrtf_dir = self.data_dir_path + self.baseline_dir + '/barycentric/valid'
        self.hrtf_selection_dir = self.data_dir_path + self.baseline_dir + '/hrtf_selection/valid'

        self.train_val_id_dir = self.data_dir_path + self.data_dir + '/train_val_id'

        self.mean_std_filename = self.data_dir_path + self.data_dir + '/mean_std_' + f'{self.dataset}.pickle'
        self.mean_std_coef_dir = self.data_dir_path + self.data_dir + '/coef_mean_std'
        self.train_sd_ild_mean_std_filename = self.data_dir_path + self.data_dir + f'/{self.dataset}_train_sd_ild_mean_std.pickle'

        self.log_path = f'{self.data_dir_path}/logs'
        self.checkpoint_path = f'{self.data_dir_path}/checkpoints'

        # Training hyperparams
        self.batch_size = 8
        self.num_workers = 1
        self.num_initial_points = 19
        self.optimizer = 'adam'
        self.num_epochs = 100  # was originally 250
        self.lr = 5e-4
        self.margin = 1.8670232e-08 # filter out negative values and make it non-zero when in magnitude domain
        self.apply_sht = True
        self.use_mse_loss = False
        self.save_interval = 150
        self.CosineAnnealingLR_period = 50
        self.use_nd_loss = True
        self.use_cos_loss = False

        # model parameters
        self.latent_dim = 128
        self.lr_pad_idx=0
        self.hidden_size = 1024
        self.num_encoder_transformer_layers = 1
        self.num_decoder_transformer_layers = 1
        self.decoder_initial_size = 16
        self.num_heads = 8
        self.num_groups = 2
        self.dropout = 0.
        self.norm_type = "token_scale"
        self.activation = "relu"

        # Loss function weight
        self.content_weight = 1
        self.mse_scale = 1

        self.ngpu = 1
        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = 'cpu'

    def save(self, n):
        self.raw_hrtf_dir = str(self.raw_hrtf_dir)
        j = {}
        for k, v in self.__dict__.items():
            j[k] = v
        with open(f'{self.data_dir_path}/configs/config_files/config_{n}.json', 'w') as f:
            json.dump(j, f)

    def load(self, n):
        with open(f'{self.data_dir_path}/configs/config_files/config_{n}.json', 'r') as f:
            j = json.load(f)
            for k, v in j.items():
                setattr(self, k, v)
        self.raw_hrtf_dir = Path(self.raw_hrtf_dir)

    def get_train_params(self):
        # deprecated
        params = {
            "domain": self.domain,
            "max sht degree": self.max_degree,
            "max num points": self.max_num_points,
            "hrtf loader": self.hrtf_loader,
            "data path": self.raw_hrtf_dir,
            "batch size": self.batch_size,
            "initial points": self.num_initial_points,
            "optimizer": self.optimizer,
            "epochs": self.num_epochs,
            "lr": self.lr,
            "sht": self.apply_sht,
            "mse loss": self.use_mse_loss,
            "latent dim": self.latent_dim,
            "hidden size": self.hidden_size,
            "encoder layer": self.num_encoder_transformer_layers,
            "decoder layer": self.num_decoder_transformer_layers,
            "decoder initial size": self.decoder_initial_size,
            "num heads": self.num_heads,
            "num groups": self.num_groups,
            "dropout": self.dropout,
            "norm type": self.norm_type,
            "activation": self.activation,
            "neighbor loss": self.use_nd_loss
        }
        return params
