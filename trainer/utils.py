import torch
import numpy as np
import pickle
import math

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.dataset import CUDAPrefetcher, CPUPrefetcher, MergeHRTFDataset
from configs.config import Config
from configs.model_config import ModelConfig
from model.model import HRTF_Transformer, AutoEncoder, Encoder, Decoder
from model.res_encoder import ResEncoder
from model.DBPN import D_DBPN
from data.utils import get_hrtf_loader_function
from data.hartufo import HrirSpec
import importlib

def compute_sh_degree(config):
    # deprecated
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('data.hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'time'}}, subject_ids='first')
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)
    degree = max(1, int(np.sqrt(num_row_angles*num_col_angles*num_radii/config.upscale_factor) - 1)) 
    return degree


def load_hrtf(config: Config, mean=None, std=None):
    # imp = importlib.import_module('data.hrtfdata.full')
    # load_function = getattr(imp, config.dataset)
    load_function = get_hrtf_loader_function(config)

    id_file_dir = config.train_val_id_dir
    id_filename = id_file_dir + '/train_val_id.pickle'
    with open(id_filename, "rb") as file:
        train_ids, val_ids = pickle.load(file)

    # define transforms
    if mean is None or std is None:
        transform = None
    else:
        transform = (mean, std)

    domain = config.domain
    max_degree = config.max_degree
    apply_sht = config.apply_sht
    hrtf_loader = config.hrtf_loader

    data_dir = config.raw_hrtf_dir / config.dataset.upper()

    if hrtf_loader == "hartufo":
        left_train = load_function(data_dir, features_spec=HrirSpec(domain=domain, side='left', samplerate=config.hrir_samplerate), subject_ids=train_ids)
        right_train = load_function(data_dir, features_spec=HrirSpec(domain=domain, side='right', samplerate=config.hrir_samplerate), subject_ids=train_ids)
        left_val = load_function(data_dir, features_spec=HrirSpec(domain=domain, side='left', samplerate=config.hrir_samplerate), subject_ids=val_ids)
        right_val = load_function(data_dir, features_spec=HrirSpec(domain=domain, side='right', samplerate=config.hrir_samplerate), subject_ids=val_ids)
    elif hrtf_loader == "hrtfdata":
        left_train = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'left', 'domain': domain}}, subject_ids=train_ids)
        right_train = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'right', 'domain': domain}}, subject_ids=train_ids)
        left_val = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'left', 'domain': domain}}, subject_ids=val_ids)
        right_val = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'right', 'domain': domain}}, subject_ids=val_ids)
    else:
        raise ValueError(f"unrecognized hrtf loader: {hrtf_loader}")
    train_dataset = MergeHRTFDataset(hrtf_loader, left_train, right_train, config.num_initial_points, max_degree=max_degree, apply_sht=apply_sht, transform=transform)
    val_dataset = MergeHRTFDataset(hrtf_loader, left_val, right_val, config.num_initial_points, max_degree=max_degree, apply_sht=apply_sht, transform=transform)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    test_dataloader = DataLoader(val_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)
    
    # Place all data on the preprocessing data loader
    if torch.cuda.is_available() and config.ngpu > 0:
        device = torch.device(config.device_name)
        train_prefetcher = CUDAPrefetcher(train_dataloader, device)
        test_prefetcher = CUDAPrefetcher(test_dataloader, device)
    else:
        train_prefetcher = CPUPrefetcher(train_dataloader)
        test_prefetcher = CPUPrefetcher(test_dataloader)
    return train_prefetcher, test_prefetcher


def progress(i, batches, n, num_epochs, timed):
    """Prints progress to console

    :param i: Batch index
    :param batches: total number of batches
    :param n: Epoch number
    :param num_epochs: Total number of epochs
    :param timed: Time per batch
    """
    message = 'batch {} of {}, epoch {} of {}'.format(i, batches, n, num_epochs)
    print(f"Progress: {message}, Time per iter: {timed}")


def spectral_distortion_inner(input_spectrum, target_spectrum, domain="magnitude"):
    numerator = target_spectrum
    denominator = input_spectrum
    if domain == "magnitude": 
        return torch.mean((20 * torch.log10(numerator / denominator)) ** 2)
    else:
        return torch.mean((numerator - denominator) ** 2)

def spectral_distortion_inner_v1(input_spectrum, target_spectrum, domain="magnitude"):
    # this function is used for computing the spectral distortion for input with shape of [b, nbins, r, w, h]
    numerator = target_spectrum
    denominator = input_spectrum
    if domain == "magnitude": 
        return torch.mean((20 * torch.log10(numerator / denominator)) ** 2, dim=1)
    else:
        return torch.mean((numerator - denominator) ** 2, dim=1)

def spectral_distortion_metric(generated, target, reduction='mean', domain="magnitude"):
    """Computes the mean spectral distortion metric for a 5 dimensional tensor (N x C x P x W x H)
    Where N is the batch size, C is the number of frequency bins, P is the number of panels (usually 5),
    H is height, and W is width.

    Computes the mean over every HRTF in the batch"""
    batch_size = generated.size(0)
    num_panels = generated.size(2)
    width = generated.size(3)
    height = generated.size(4)
    total_positions = num_panels * height * width
    # total_sd_metric = 0
    # for b in range(batch_size):
    #     total_all_positions = 0
    #     for i in range(num_panels):
    #         for j in range(width):
    #             for k in range(height):
    #                 average_over_frequencies = spectral_distortion_inner(generated[b, :, i, j, k],
    #                                                                      target[b, :, i, j, k], domain)
    #                 total_all_positions += torch.sqrt(average_over_frequencies)
    #     sd_metric = total_all_positions / total_positions
    #     total_sd_metric += sd_metric
    average_over_frequencies = spectral_distortion_inner_v1(generated, target, domain)
    total_all_positions = torch.sum(torch.sqrt(average_over_frequencies))
    total_sd_metric = total_all_positions / total_positions
    if reduction == 'mean':
        output_loss = total_sd_metric / batch_size
    elif reduction == 'sum':
        output_loss = total_sd_metric
    else:
        raise RuntimeError("Please specify a valid method for reduction (either 'mean' or 'sum').")

    return output_loss


def ILD_metric_inner(config, input_spectrum, target_spectrum):
    input_left = input_spectrum[:config.nbins_hrtf]
    input_right = input_spectrum[config.nbins_hrtf:]
    target_left = target_spectrum[:config.nbins_hrtf]
    target_right = target_spectrum[config.nbins_hrtf:]
    if config.domain == "magnitude":
        input_ILD = torch.mean((20 * torch.log10(input_left / input_right)))
        target_ILD = torch.mean((20 * torch.log10(target_left / target_right)))
    else:
        input_ILD = torch.mean(input_left - input_right)
        target_ILD = torch.mean(target_left - target_right)
    return torch.abs(input_ILD - target_ILD)

def ILD_metric_inner_v1(nbins, input_spectrum, target_spectrum, domain="magnitude"):
    # this function is used for computing ild loss for input with shape of [b, nbins, r, w, h]
    input_left = input_spectrum[:,:nbins,...]
    input_right = input_spectrum[:,nbins:,...]
    target_left = target_spectrum[:,:nbins,...]
    target_right = target_spectrum[:,nbins:,...]
    if domain == "magnitude":
        input_ILD = torch.mean((20 * torch.log10(input_left / input_right)), dim=1)
        target_ILD = torch.mean((20 * torch.log10(target_left / target_right)), dim=1)
    else:
        input_ILD = torch.mean(input_left - input_right, dim=1)
        target_ILD = torch.mean(target_left - target_right, dim=1)
    return torch.abs(input_ILD - target_ILD)


def ILD_metric(nbins, generated, target, reduction="mean", domain="magnitude"):
    batch_size = generated.size(0)
    num_panels = generated.size(2)
    height = generated.size(3)
    width = generated.size(4)
    total_positions = num_panels * height * width

    # total_ILD_metric = 0
    # for b in range(batch_size):
    #     total_all_positions = 0
    #     for i in range(num_panels):
    #         for j in range(height):
    #             for k in range(width):
    #                 average_over_frequencies = ILD_metric_inner(config, generated[b, :, i, j, k], target[b, :, i, j, k])
    #                 total_all_positions += average_over_frequencies
    #     ILD_metric_batch = total_all_positions / total_positions
    #     total_ILD_metric += ILD_metric_batch

    average_over_frequencies = ILD_metric_inner_v1(nbins, generated, target, domain)
    total_ILD_metric = torch.sum(average_over_frequencies) / total_positions

    if reduction == 'mean':
        output_loss = total_ILD_metric / batch_size
    elif reduction == 'sum':
        output_loss = total_ILD_metric
    else:
        raise RuntimeError("Please specify a valid method for reduction (either 'mean' or 'sum').")

    return output_loss

def sd_ild_loss(config: Config, generated, target, sd_mean, sd_std, ild_mean, ild_std):
    """Computes the mean sd/ild loss for a 5 dimensional tensor (N x C x P x W x H)
    Where N is the batch size, C is the number of frequency bins, P is the number of panels (usually 5),
    H is height, and W is width.

    Computes the mean over every HRTF in the batch"""

    # calculate SD and ILD metrics
    sd_metric = spectral_distortion_metric(generated, target, domain=config.domain)
    ild_metric = ILD_metric(config.nbins_hrtf, generated, target, domain=config.domain)
    # with open("log.txt", "a") as f:
    #     f.write(f"sd nan? {torch.isnan(sd_metric).any()}")
    #     f.write(f"ild nan? {torch.isnan(ild_metric).any()}")

    # normalize SD and ILD based on means/standard deviations passed to the function
    sd_norm = torch.div(torch.sub(sd_metric, sd_mean), sd_std)
    ild_norm = torch.div(torch.sub(ild_metric, ild_mean), ild_std)

    # add normalized metrics together
    sum_norms = torch.add(sd_norm, ild_norm)

    # un-normalize
    sum_std = (sd_std ** 2 + ild_std ** 2) ** 0.5
    sum_mean = sd_mean + ild_mean

    output = torch.add(torch.mul(sum_norms, sum_std), sum_mean)

    return output


def cos_similarity_loss(generated, target):
    # cos_similarity_criterion = nn.CosineSimilarity(dim=2)
    # avg_cos_loss_over_frequency = ((1-cos_similarity_criterion(generated, target))**2).mean(1)
    # # take square root and average over batch size
    # cos_loss = torch.sqrt(avg_cos_loss_over_frequency).mean()
    # return cos_loss
    
    # simplified version 0402
    cos_sim = F.cosine_similarity(generated, target, dim=-1)
    return (1 - cos_sim).mean()
    

def compute_gradient_penalty(discriminator, real_data, fake_data):
    batch_size = real_data.size(0)
    device = real_data.device
    alpha = torch.rand(batch_size, 1, 1).to(device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)

    # Set requires_grad=True for interpolates
    interpolates.requires_grad_(True)

    # Calculate critic scores for interpolates
    interpolates_scores = discriminator(interpolates)

    # Compute gradients of critic with respect to interpolates
    gradients = torch.autograd.grad(outputs=interpolates_scores,
                                    inputs=interpolates,
                                    grad_outputs=torch.ones_like(interpolates_scores),
                                    create_graph=True,
                                    retain_graph=True)[0]

    # Compute the norm of gradients
    gradients_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()  # Penalty term
    return gradient_penalty

def plot_hrtf(generated, target, path, filename):
    x = generated[0, 0, 0, :]
    y = target[0, 0, 0, :]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(x)
    ax1.set_title('recon')
    ax2.plot(y)
    ax2.set_title('original')
    plt.savefig(f"{path}/{filename}.png")
    plt.close()

def plot_losses(losses, labels, colors, path, filename, title="Loss Curves"):
    """Plot loss curves"""
    params = {
        'axes.labelsize': 10,
        'font.size': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': [6, 4.5]
    }
    plt.rcParams.update(params)
    plt.figure()
    plt.grid(ls='dashed', axis='y', color='0.8')
    for i in range(len(losses)):
        plt.plot(losses[i], label=labels[i], linewidth=2, color=colors[i])
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt_legend = plt.legend()
    frame = plt_legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')
    plt.savefig(f'{path}/{filename}.png')
    plt.close()

def plot_test_sample_hrtf(path, ir_id, ori_hrtf, recon_hrtf, is_min=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    if is_min:
        title = "min diff"
    else:
        title = "max diff"
    ax1.plot(ori_hrtf[ir_id])
    ax1.set_title(f'{title} original (ID: {ir_id})')
    ax2.plot(recon_hrtf[ir_id])
    ax2.set_title(f'{title} recon (ID {ir_id})')

    # plt.show()
    plt.savefig(f"{path}/tf_{title}_{ir_id}.png")
    plt.close()

def convert_num_points_to_num_coeff(num_points):
    # minimum coefficients is set to 4
    return max(4, math.floor(math.sqrt(num_points)) ** 2)

def get_model(config: Config):
    ngpu = config.ngpu
    device = torch.device(config.device_name if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    nbins = config.nbins_hrtf * 2 # left and right
    if config.apply_sht:
        # max num coeff
        target_size = (config.max_degree + 1) ** 2
        # initial num coeff
        lr_size = convert_num_points_to_num_coeff(config.num_initial_points)
    else:
        lr_size = config.num_initial_points
        target_size = config.max_num_points
    encoder_config = ModelConfig(nbins=nbins,
                                 hidden_size=config.hidden_size,
                                 num_transformer_layers=config.num_encoder_transformer_layers,
                                 num_heads=config.num_heads,
                                 num_groups=config.num_groups,
                                 dropout=config.dropout,
                                 initial_size=lr_size,
                                 target_size=target_size,
                                 latent_dim=config.latent_dim,
                                 apply_sht=config.apply_sht,
                                 norm_type=config.norm_type,
                                 activation=config.activation)
    
    decoder_config = ModelConfig(nbins=nbins,
                                 hidden_size=config.hidden_size,
                                 num_transformer_layers=config.num_decoder_transformer_layers,
                                 num_heads=config.num_heads,
                                 num_groups=config.num_groups,
                                 dropout=config.dropout,
                                 initial_size=config.decoder_initial_size,
                                 target_size=target_size,
                                 latent_dim=config.latent_dim,
                                 apply_sht=config.apply_sht,
                                 norm_type=config.norm_type,
                                 activation=config.activation)
    
    # model initialization
    # hrtf_transformer = HRTF_Transformer(encoder_config, decoder_config).to(device)

    # model = AutoEncoder(nbins=nbins, initial_size=lr_size, latent_dim=config.latent_dim, base_channels=512, target_size=target_size).to(device)
    # model = ResEncTranDec(encoder_config, decoder_config).to(device)
    # model = AutoEncoder(ResEncoder, encoder_config, Decoder, decoder_config).to(device)
    # model = AutoEncoder(Encoder, encoder_config, D_DBPN, decoder_config).to(device)
    model = AutoEncoder(Encoder, encoder_config, Decoder, decoder_config).to(device)

    return model

def magnitude_neighbor_dissim_loss(recons, target, reduction='mean'):
    # straight forward version of neighor dissmiliarity loss, but not efficient
    recons = F.pad(recons, pad=(0,0,1,1,0,0), mode='circular')
    target = F.pad(target, pad=(0,0,1,1,0,0), mode='circular')
    r, w, h = recons.shape[-3:] # 74, 12
    total_loss = 0.
    total_positions = r * (w-2) * (h-2)

    def compute_neighbor_diff(hrtf, center):
        # expected input shape: [b nbins r w h]
        # center: (x, y)
        x, y = center
        # diff = 0
        # for offset_w, offset_h in [(0,1), (0,-1), (1,0), (-1,0)]:
        #     diff += hrtf[...,x,y] - hrtf[..., x + offset_w, y + offset_h]

        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = torch.stack([hrtf[...,x+offset_w, y+offset_h] for offset_w, offset_h in offsets], dim=0)
        avg_neighbors = torch.mean(neighbors, dim=0)

        return hrtf[...,x,y] - avg_neighbors
    
    for i in range(1, w-1):
        for j in range(1, h-1):
            deriv_recons = compute_neighbor_diff(recons, (i, j))
            deriv_target = compute_neighbor_diff(target, (i, j))
            total_loss += torch.mean((deriv_recons - deriv_target) ** 2, dim=(1,2)) # average over frequency dim, get rid of radius dim
    neighbor_dissim_loss = total_loss / total_positions
    print(neighbor_dissim_loss.shape)
    if reduction == 'mean':
        output_loss = torch.mean(neighbor_dissim_loss)
    elif reduction == 'sum':
        output_loss = torch.sum(neighbor_dissim_loss)
    else:
        raise RuntimeError("Please specify a valid method for reduction (either 'mean' or 'sum').")
    return output_loss

def neighbor_dissim_metric(recons, target, reduction='mean', domain="magnitude"):
    # if domain == "magnitude_db":
    #     recons_mag = 10 ** (recons / 20)
    #     target_mag = 10 ** (target / 20)
    # else:
    #     recons_mag = recons
    #     target_mag = target
    recons_mag = recons
    target_mag = target
    recons_mag = F.pad(recons_mag, (0,0,1,1,0,0), mode='circular')
    target_mag = F.pad(target_mag, (0,0,1,1,0,0), mode='circular')
    r, w, h = recons_mag.shape[-3:] # 1, 74, 12
    total_positions = r * (w-2) * (h-2)

    def compute_diff(hrtf):
        unfolded = F.unfold(hrtf.reshape(-1, r, w, h),
                            kernel_size=3,
                            padding=0,
                            stride=1) # [b*nbins, 9, n_patches]
        
        center = unfolded[:, 4:5, :] # [b*nbins, 1, n_patches]
        neighbors = unfolded[:, [1,3,5,7], :] # [b*nbins, 4, n_patches]
        avg_neighbors = torch.mean(neighbors, dim=1, keepdim=True)

        diff = center - avg_neighbors
        return diff.reshape(hrtf.shape[0], hrtf.shape[1], -1)
    
    diff_recons = compute_diff(recons_mag)
    diff_target = compute_diff(target_mag)

    loss = torch.sum(torch.mean((diff_recons - diff_target) ** 2, dim=1))
    loss /= total_positions
    if reduction == 'mean':
        return loss / recons.shape[0]
    elif reduction == 'sum':
        return loss
    else:
        raise RuntimeError("Please specify a valid method for reduction (either 'mean' or 'sum').")