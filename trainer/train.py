import pickle
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime
import time

from trainer.utils import *
from data.utils import get_dataset_info, load_mean_std, inverse_sht

from configs.config import Config
from configs.model_config import  ModelConfig
from model.model import HRTF_Transformer

def get_model_and_optimizer(config: Config):
    # model initialization
    hrtf_transformer = get_model(config)

    # optimizer
    if config.optimizer == "adam":
        optimizer = optim.Adam(hrtf_transformer.parameters(), lr=config.lr)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(hrtf_transformer.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0001)
    else:
        raise ValueError(f"unrecognized optimizer: {config.optimizer}")


    return hrtf_transformer, optimizer

def train(config: Config, model, optimizer, train_prefetcher):
    """ Train the transformer model

    Args:
        config: Config object containing model hyperparameters
        model: transformer model instance
        train_prefetcher: prefetcher for training data
    """
    domain = config.domain
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(config.log_path, str(config.num_initial_points), current_time)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "log.txt")
    training_params = config.get_train_params()
    with open(log_file_path, "a") as f:
        f.write("=== Training Parameters ===\n")
        for k,v in training_params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
    plot_dir = os.path.join(log_dir, "plots", "train")
    os.makedirs(plot_dir, exist_ok=True)
    checkpoint_dir = os.path.join(config.checkpoint_path, str(config.num_initial_points), current_time)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # get data distribution info (row angles, column angles, radii) for latter use
    config.row_angles, config.column_angles, config.radii = get_dataset_info(config)

    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)

    # Assign torch device
    ngpu = config.ngpu

    device = torch.device(config.device_name if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    print(f'Using {ngpu} GPUs. ')
    print(device, " will be used.\n")
    cudnn.benchmark = True

    # set lr sheduler
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.CosineAnnealingLR_period)

    # loss functions
    cos_similarity_criterion = cos_similarity_loss
    content_criterion = sd_ild_loss
    mse_loss_fn = nn.MSELoss()

    # mean and std for ILD and SD, which are used for normalization
    # computed based on average ILD and SD for training data, when comparing each individual
    # to every other individual in the training data
    with open(config.train_sd_ild_mean_std_filename, 'rb') as f:
        # sd_mean: 6.12656831741333, sd_std: 0.4705064594745636, ild_mean: 1.9910638332366943, ild_std: 0.4973623752593994
        # old: sd_mean: 5.916355609893799, sd_std: 0.45490849018096924, ild_mean: 1.7631410360336304, ild_std: 0.4395105242729187
        sd_mean, sd_std, ild_mean, ild_std = pickle.load(f)
    # sd_mean = 7.387559253346883
    # sd_std = 0.577364154400081
    # ild_mean = 3.6508303231127868
    # ild_std = 0.5261339271318863

    if config.normalize_input:
        mean, std = load_mean_std(config, device)

    train_loss_list = []
    grad_norm_list = []
    sd_loss_list = []
    if config.use_nd_loss:
        neighbor_dissim_loss_list = []
    if config.apply_sht:
        train_content_loss_list = []
        train_sh_coeff_mse_list = []
        train_sh_coeff_cos_list = []

    # initialize min_loss
    min_loss = float('inf')

    for epoch in range(config.num_epochs):
        with open(log_file_path, "a") as f:
            f.write(f"\nEpoch: {epoch}\n")

        times = []
        train_loss = 0.
        sd_loss = 0.
        if config.use_nd_loss:
            neighbor_loss = 0.
        if config.apply_sht:
            train_content_loss = 0.
            train_sh_coeff_mse_loss = 0.
            train_sh_coeff_cos_loss = 0.

        # Initialize the number of data batches to print logs on the terminal
        batch_index = 0

        # Initialize the data loader and load the first batch of data
        train_prefetcher.reset()
        batch_data = train_prefetcher.next()

        while batch_data is not None:
            if ('cuda' in str(device)) and (ngpu > 1):
                start_overall = torch.cuda.Event(enable_timing=True)
                end_overall = torch.cuda.Event(enable_timing=True)
                start_overall.record()
            else:
                start_overall = time.time()

            # Transfer in-memory data to CUDA devices to speed up training
            if config.apply_sht:
                # lr_coefficient shape: [b, num_initial_coefficients, nbins]
                lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                                 non_blocking=True, dtype=torch.float)
                # hr_coefficient shape: [b, nbins, num_coefficients]
                hr_coefficient = batch_data["hr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                                 non_blocking=True, dtype=torch.float)
                # hrtf shape: [b, nbins, r, w, h]
                hrtf = batch_data["hrtf"].to(device=device, memory_format=torch.contiguous_format,
                                             non_blocking=True, dtype=torch.float)
                masks = batch_data["mask"]
                
                sr = model(lr_coefficient)
                # cosine similarity loss
                if not config.use_mse_loss and config.use_cos_loss:
                    sh_coeff_cos_loss = cos_similarity_criterion(sr, hr_coefficient)
                    sh_coeff_mse_loss = ((sr - hr_coefficient) ** 2).mean()
                recons = inverse_sht(config, sr, masks)
            else:
                # lr_hrtf shape: [b, num_initial_points, nbins]
                lr_hrtf = batch_data["lr_hrtf"].to(device=device, memory_format=torch.contiguous_format,
                                                   non_blocking=True, dtype=torch.float)
                # hrtf shape: [b, nbins, r, w, h]
                hrtf = batch_data["hr_hrtf"].to(device=device, memory_format=torch.contiguous_format,
                                                   non_blocking=True, dtype=torch.float)
                # recons shape: [b, nbins, num_points]
                recons = model(lr_hrtf)
                recons = recons.reshape(hrtf.shape)

            # monitor training sd
            x = recons.detach().clone()
            y = hrtf.detach().clone()
            spectral_distorion = spectral_distortion_metric(x, y, domain=config.domain).item()
            sd_loss += spectral_distorion

            # during every 25th epoch and last epoch, save filename for mag spectrum plot
            if epoch % 25 == 0 or epoch == (config.num_epochs - 1):
                generated = recons[0].permute(2, 3, 1, 0)  # w x h x r x nbins
                target = hrtf[0].permute(2, 3, 1, 0)
                id = batch_data['id'][0].item()
                filename = f"magnitude_{id}_{epoch}"
                plot_hrtf(generated.detach().cpu(), target.detach().cpu(), plot_dir, filename)

            # loss
            if config.use_mse_loss:
                loss = mse_loss_fn(recons, hrtf) * config.mse_scale
            else:
                unweighted_content_loss = content_criterion(config, recons, hrtf, sd_mean, sd_std, ild_mean, ild_std)
                content_loss = config.content_weight * unweighted_content_loss
                if config.apply_sht and config.use_cos_loss:
                    loss = content_loss + sh_coeff_cos_loss
                else:
                    loss = content_loss
            neighbor_dissim_loss = None
            if config.use_nd_loss:
                neighbor_dissim_loss = neighbor_dissim_metric(recons, hrtf, domain=config.domain)
                neighbor_loss += neighbor_dissim_loss.item()
                loss += neighbor_dissim_loss
            
            train_loss += loss.item()
            if config.apply_sht and not config.use_mse_loss:
                train_content_loss += content_loss.item()
                if config.use_cos_loss:
                    train_sh_coeff_cos_loss += sh_coeff_cos_loss.item()
                    train_sh_coeff_mse_loss += sh_coeff_mse_loss.item()
            
            # backward
            loss.backward()
            
            # gradient clipping
            # total_norm = torch.nn.utils.clip_grad_norm_(
            #     model.parameters(),
            #     max_norm=10.0,
            #     norm_type=2
            # )

            # compute grad norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norm_list.append(total_norm)
            # print(f'Total gradient norm: {total_norm}')

            # optimizer
            optimizer.step()
            optimizer.zero_grad()

            # lr scheduler
            # lr_scheduler.step()

            with open(log_file_path, "a") as f:
                f.write(f"{batch_index}/{len(train_prefetcher)}\n")
                f.write(f"loss: {loss.item()}\n")
                f.write(f"grad_norm: {grad_norm_list[-1]}\n")
                f.write(f"sd: {spectral_distorion}\n")
                if config.use_nd_loss and neighbor_dissim_loss is not None:
                    f.write(f"neighbor dissimilarity loss: {neighbor_dissim_loss.item()}\n")
                if config.apply_sht and not config.use_mse_loss:
                    if config.use_cos_loss:
                        f.write(f"sh cos: {sh_coeff_cos_loss.item()}, sh mse: {sh_coeff_mse_loss.item()}\n")
                    f.write(f"content loss: {content_loss.item()}\n\n")
                
            
            if ('cuda' in str(device)) and (ngpu > 1):
                end_overall.record()
                torch.cuda.synchronize()
                times.append(start_overall.elapsed_time(end_overall))
            else:
                end_overall = time.time()
                times.append(end_overall - start_overall)

            # Every 0th batch log useful metrics
            if batch_index == 0:
                with torch.no_grad():
                    # if epoch % config.save_interval == 0 or epoch == (config.num_epochs - 1):
                    #     torch.save(model.state_dict(), f'{checkpoint_dir}/transformer_{epoch}.pt')
                    progress(batch_index, batches, epoch, config.num_epochs, timed=np.mean(times))
                    times = []

            # Preload the next batch of data
            batch_data = train_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1
        train_loss_list.append(train_loss / len(train_prefetcher))
        sd_loss_list.append(sd_loss / len(train_prefetcher))
        if config.use_nd_loss:
            neighbor_dissim_loss_list.append(neighbor_loss / len(train_prefetcher))
        if config.apply_sht and not config.use_mse_loss:
            train_content_loss_list.append(train_content_loss / len(train_prefetcher))
            train_sh_coeff_cos_list.append(train_sh_coeff_cos_loss / len(train_prefetcher))
            train_sh_coeff_mse_list.append(train_sh_coeff_mse_loss / len(train_prefetcher))
        print(f"Average epoch loss: {train_loss_list[-1]}")
        print(f"grad norm: {grad_norm_list[-1]}")
        print(f"sd loss: {sd_loss_list[-1]}")
        if config.use_nd_loss:
            print(f"neighbor dissimilarity loss: {neighbor_dissim_loss_list[-1]}")
        if config.apply_sht and not config.use_mse_loss:
            print(f"Average content loss: {train_content_loss_list[-1]}")
            if config.use_cos_loss:
                print(f"Aberage sh mse loss: {train_sh_coeff_mse_list[-1]}, sh cos loss: {train_sh_coeff_cos_list[-1]}")

        if train_loss_list[-1] < min_loss:
            msg = f"better result obtained, new checkpoint saved at epoch {epoch}, cur: {train_loss_list[-1]}, prev: {min_loss}"
            print(msg)
            min_loss = train_loss_list[-1]
            with open(log_file_path, "a") as f:
                f.write(msg)
            torch.save(model.state_dict(), f'{checkpoint_dir}/transformer.pt')
    # plot loss curves
    plot_path = os.path.join(plot_dir, "losses")
    os.makedirs(plot_path, exist_ok=True)
    plot_losses([train_loss_list], ['Training loss'], ['red'], path=plot_path, filename='loss', title="Training Loss")
    plot_losses([grad_norm_list], ['grad norm'], ['green'], path=plot_path, filename='grad_norm', title="Grad Norm")
    plot_losses([sd_loss_list], ['training sd loss'], ['blue'], path=plot_path, filename='training_sd', title="Training sd")
    if config.use_nd_loss and neighbor_dissim_loss_list:
        plot_losses([neighbor_dissim_loss_list], ['neighbor dissimilarity loss'], ['cyan'], path=plot_path, filename='neighbor_loss', title="neighbor loss")
    if config.apply_sht and not config.use_mse_loss:
        if config.use_cos_loss:
            plot_losses([train_sh_coeff_mse_list],['SH mse loss'],['blue'], path=plot_path, filename='SH_mse_loss', title="SH mse loss")
            plot_losses([train_sh_coeff_cos_list],['SH cos loss'],['blue'], path=plot_path, filename='SH_cos_loss', title="SH cos loss")
            plot_losses([train_loss_list, train_content_loss_list, train_sh_coeff_cos_list],
                        ['Training loss', 'Content loss', 'coefficient sim loss'],
                        ['green', 'purple', 'red'],
                        path=plot_path, filename='loss_curves', title="Training loss curves")
        else:
            plot_losses([train_loss_list, train_content_loss_list],
                        ['Training loss', 'Content loss'],
                        ['green', 'purple'],
                        path=plot_path, filename='loss_curves', title="Training loss curves")
        with open(f'{log_dir}/train_losses.pickle', "wb") as file:
            pickle.dump((train_loss_list, train_content_loss_list, train_sh_coeff_cos_list, train_sh_coeff_mse_list), file)
    else:
        with open(f'{log_dir}/train_losses.pickle', "wb") as file:
            pickle.dump((train_loss_list), file)
    print("TRAINING FINISHED")
    
def train_model(config: Config):
    if config.normalize_input:
        print("normalize input")
        mean_std_dir = config.mean_std_coef_dir
        mean_std_full = mean_std_dir + "/mean_std_full.pickle"
        with open(mean_std_full, "rb") as f:
            mean_full, std_full = pickle.load(f)

        mean_std_lr = mean_std_dir + f"/mean_std_{config.upscale_factor}.pickle"
        with open(mean_std_lr, "rb") as f:
            mean_lr, std_lr = pickle.load(f)
        mean = (mean_lr, mean_full)
        std = (std_lr, std_full)
    else:
        mean, std = None, None
    train_prefetcher, _ = load_hrtf(config, mean, std)
    print("train prefetcher: ", len(train_prefetcher))

    hrtf_transformer, optimizer = get_model_and_optimizer(config)
    print("------Start training!--------")
    train(config, hrtf_transformer, optimizer, train_prefetcher)


