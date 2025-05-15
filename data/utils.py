import torch
import torch.nn.functional as F
import pickle
import importlib

from configs.config import Config
from .hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
from .hartufo import HrirSpec

def load_mean_std(config, device):
    if config.normalize_input:
        mean_std_dir = config.mean_std_coef_dir
        mean_std_full = mean_std_dir + "/mean_std_full.pickle"
        with open(mean_std_full, "rb") as f:
            mean, std = pickle.load(f)
        mean = mean.float().to(device)
        std = std.float().to(device)
        return mean, std
    else:
        return None, None

def get_hrtf_loader_function(config: Config):
    hrtf_loader = config.hrtf_loader
    if hrtf_loader == "hartufo":
        imp = importlib.import_module('data.hartufo.full')
        dataset = config.dataset.capitalize() # Sonicom
    elif hrtf_loader == "hrtfdata":
        imp = importlib.import_module('data.hrtfdata.full')
        dataset = config.dataset.upper() # SONICOM
    else:
        raise ValueError(f"unrecognized hrtf loader: {hrtf_loader}")

    load_function = getattr(imp, dataset)
    return load_function
    

def get_dataset_info(config: Config):
    load_function = get_hrtf_loader_function(config)
    dataset = config.dataset.upper() # SONICOM
    data_dir = config.raw_hrtf_dir / dataset
    hrtf_loader = config.hrtf_loader
    if hrtf_loader == "hartufo":
        ds = load_function(data_dir, features_spec=HrirSpec(domain="magnitude", side="left", samplerate=config.hrir_samplerate), subject_ids='first')
        return ds.fundamental_angles, ds.orthogonal_angles, ds.radii
    elif hrtf_loader == "hrtfdata":
        ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                             'side': 'left', 'domain': config.domain}}, subject_ids='first')
        return ds.row_angles, ds.column_angles, ds.radii
    else:
        raise ValueError(f"unrecognized hrtf loader: {hrtf_loader}")

def inverse_sht(config, sr, masks):
    harmonics_list = []
    bs = sr.shape[0]
    num_row_angles = len(config.row_angles)
    num_column_angles = len(config.column_angles)
    num_radii = len(config.radii)
    for mask in masks:
        SHT = SphericalHarmonicsTransform(config.max_degree, config.row_angles, config.column_angles, config.radii, mask)
        harmonics = torch.from_numpy(SHT.get_harmonics()).float()
        harmonics_list.append(harmonics)
    harmonics_tensor = torch.stack(harmonics_list).to(sr.device)
    # compute recons and rearrange the shape to [batch_size, nbins, r, w, h]
    recons = (harmonics_tensor @ sr.permute(0, 2, 1)).view(bs, num_row_angles, num_column_angles, num_radii, -1).permute(0, 4, 3, 1, 2)
    if config.domain == "magnitude":
        recons = F.relu(recons) + config.margin # filter out negative values and make it non-zero
    if config.normalize_input:
        recons = unormalize(config, recons)
    return recons
    

def unormalize(config, x):
    return x * config.std + config.mean
