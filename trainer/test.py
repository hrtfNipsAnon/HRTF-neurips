import os
import pickle
import scipy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from data.utils import get_dataset_info, load_mean_std, inverse_sht

from trainer.utils import *
from configs.config import Config

def test(config: Config, checkpoint):
    domain = config.domain

    if config.normalize_input:
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
    _, test_prefetcher = load_hrtf(config, mean, std)
    print("test set loadded successfully")

    ngpu = config.ngpu
    device = torch.device(config.device_name if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # get data distribution info (row angles, column angles, radii) for latter use
    config.row_angles, config.column_angles, config.radii = get_dataset_info(config)

    checkpoint_path = os.path.dirname(checkpoint)
    recon_mag_dir = checkpoint_path + '/mag'
    recon_db_dir = checkpoint_path + '/db'
    os.makedirs(recon_mag_dir, exist_ok=True)
    os.makedirs(recon_db_dir, exist_ok=True)

    nbins = config.nbins_hrtf * 2

    # model initialization
    model = get_model(config)
    print("Build hrtf transformer model successfully.")
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    print(f"Load hrtf transformer model weights '{checkpoint} successfully.'")

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_param_mb = param_size / 1024 ** 2
    size_buffer_mb = buffer_size / 1024 ** 2
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('param size: {:.3f}MB'.format(size_param_mb))
    print('buffer size: {:.3f}MB'.format(size_buffer_mb))
    print('model size: {:.3f}MB'.format(size_all_mb))

    # Start the verification mode of the model.
    model.eval()

    # Initialize the data loader and load the first batch of data
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()

    plot_min_max_diff = True
    count = 0
    avg_lsd = []
    while batch_data is not None:
        sample_id = batch_data["id"].item()
        print(f"test {count + 1} / {len(test_prefetcher)}")
        if config.apply_sht:
            lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True, dtype=torch.float)
            hrtf = batch_data["hrtf"].detach().cpu()
            mask = batch_data["mask"]
            # upsample lr coefficient
            with torch.no_grad():
                sr = model(lr_coefficient)
            recon = inverse_sht(config, sr, mask)[0].detach().cpu()
        else:
            lr_hrtf = batch_data["lr_hrtf"].to(device=device, memory_format=torch.contiguous_format,
                                               non_blocking=True, dtype=torch.float)
            hrtf = batch_data["hr_hrtf"].detach().cpu()
            with torch.no_grad():
                recon = model(lr_hrtf)
            recon = recon.reshape(hrtf.shape)[0].detach().cpu() # nbins x r x w x h
            
        # save reconstructed hrtfs into pickle files
        file_name = '/' + f"{config.dataset}_{sample_id}.pickle"
        if domain == "magnitude_db":
            with open(recon_db_dir + file_name, "wb") as file:
                recon_db = recon.permute(2, 3, 1, 0) # nbins x r x w x h -> w x h x r x nbins
                pickle.dump(recon_db, file)
            with open(recon_mag_dir + file_name, "wb") as file:
                recon_mag = 10 ** (recon / 20)
                recon_mag = recon_mag.permute(2, 3, 1, 0) # nbins x r x w x h -> w x h x r x nbins
                pickle.dump(recon_mag, file)
        elif domain == "magnitude":
            with open(recon_mag_dir + file_name, "wb") as file:
                recon_mag = recon.permute(2, 3, 1, 0) # nbins x r x w x h -> w x h x r x nbins
                pickle.dump(recon_mag, file)
            with open(recon_db_dir + file_name, "wb") as file:
                recon_db = 20 * torch.log10(recon)
                recon_db = recon_db.permute(2, 3, 1, 0)
                pickle.dump(recon_db, file)

        ir_id = 0
        max_value = None
        max_id = None
        min_value = None
        min_id = None
        recon = recon.view(nbins, -1).T
        original_hrtf = hrtf[0].view(nbins, -1).T
        total_all_position = 0
        total_positions = len(recon)
        total_sd_metric = 0
        print("subject: ", sample_id)
        for original, generated in zip(original_hrtf, recon):
            if domain == "magnitude_db":
                original = 10 ** (original / 20)
                generated = 10 ** (generated / 20)

            if domain == "magnitude_db" or domain == "magnitude":
                average_over_frequencies = spectral_distortion_inner(generated, original)
            elif domain == "time":
                nbins = config.nbins_hrtf
                ori_tf_left = abs(scipy.fft.rfft(original[:nbins], nbins*2)[1:])
                ori_tf_right = abs(scipy.fft.rfft(original[nbins:], nbins*2)[1:])
                gen_tf_left = abs(scipy.fft.rfft(generated[:nbins], nbins*2)[1:])
                gen_tf_right = abs(scipy.fft.rfft(generated[nbins:], nbins*2)[1:])

                ori_tf = np.ma.concatenate([ori_tf_left, ori_tf_right])
                gen_tf = np.ma.concatenate([gen_tf_left, gen_tf_right])

                average_over_frequencies = spectral_distortion_inner(gen_tf, ori_tf)
            else:
                raise ValueError(f"Domain '{domain}' is not recognized. Expected 'magnitude_db', 'magnitude', or 'time'.")
            total_all_position += np.sqrt(average_over_frequencies)

            if max_value is None or np.sqrt(average_over_frequencies) > max_value:
                max_value = np.sqrt(average_over_frequencies)
                max_id = ir_id
            if min_value is None or np.sqrt(average_over_frequencies) < min_value:
                min_value = np.sqrt(average_over_frequencies)
                min_id = ir_id
            ir_id += 1
        
        sd_metric = total_all_position / total_positions
        total_sd_metric += sd_metric
        avg_lsd.append(sd_metric)
        print("Log SD (across all positions): ", float(sd_metric))

        if plot_min_max_diff:
            plot_test_sample_hrtf(checkpoint_path, min_id, original_hrtf, recon, is_min=True)
            plot_test_sample_hrtf(checkpoint_path, max_id, original_hrtf, recon, is_min=False)
            plot_min_max_diff = False

        # Preload the next batch of data
        batch_data = test_prefetcher.next()
        count += 1
    print("lsd for all test subject: ", avg_lsd)
    mean_lsd = np.mean(avg_lsd)
    print("avg lsd: ", mean_lsd)
