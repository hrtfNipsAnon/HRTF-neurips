import torch
import numpy as np
from torch.utils.data import Dataset

from .hrtfdata.transforms.hrirs import SphericalHarmonicsTransform

def get_sample_coords(num_initial_points):
    """
    row = [-180., -175., -170., -165., -160., -155., -150., -145., -140., -135., -130., -125.,
            -120., -115., -110., -105., -100.,  -95.,  -90.,  -85.,  -80.,  -75.,  -70.,  -65.,
            -60.,  -55.,  -50.,  -45.,  -40.,  -35.,  -30.,  -25.,  -20.,  -15.,  -10.,   -5.,
                0.  ,  5.  , 10. ,  15.  , 20. ,  25.  , 30.  , 35.  , 40. ,  45.,   50.,   55.,
            60. ,  65.  , 70. ,  75. ,  80. ,  85.  , 90.  , 95. , 100. , 105.,  110. , 115.,
            120. , 125. , 130. , 135. , 140. , 145. , 150. , 155. , 160.,  165.,  170.,  175.,]

    col = [-45., -30., -20., -10.,   0.,  10.,  20.,  30.,  45.,  60.,  75.,  90.]
    """
    if num_initial_points == 100:
        # [-180., -160., -140., -120., -100.,  -80.,  -60.,  -40.,  -20.,   -5., 5.,   20.,   40.,   60.,   80.,  100.,  120.,  140.,  160.,  175.]
        row_idx = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35, 37, 40, 44, 48, 52, 56, 60, 64, 68, 71]
        col_idx = [1, 3, 4, 6, 8] # [-30, -10, 0, 20, 45]
        return [(i, j) for i in row_idx for j in col_idx]
    
    if num_initial_points == 27:
        row_idx = [0, 8, 16, 24, 32, 40, 48, 56, 64] #[-180.0, -140.0, -100.0, -60.0, -20.0, 20.0, 60.0, 100.0, 140.0]
        col_idx = [0, 4, 8]    #[-45.0, 0.0, 45.0]
        return [(i, j) for i in row_idx for j in col_idx]

    if num_initial_points == 19:
        row_idx = [[18, 27, 36, 45, 54], [12, 21, 27, 36, 45, 51, 60], [18, 27, 36, 45, 54], [24, 48]]
        col_idx = [1, 4, 7, 9]
        return [(row, col) for col, rows in zip(col_idx, row_idx) for row in rows]
    
    if num_initial_points == 18:
        row_idx = [0, 12, 24, 36, 48, 60] #[-180, -120, -60, 0, 60, 120]
        col_idx = [1, 4, 8]  # [-30, 0, 45]
        return [(i, j) for i in row_idx for j in col_idx]
    
    if num_initial_points == 8:
        row_idx = [0, 18, 36, 54]   # [-180.0, -90.0, 0.0, 90.0]
        col_idx = [2, 8]   # [-20, 45]
        return [(i, j) for i in row_idx for j in col_idx]
    
    if num_initial_points == 5:
        return [(24, 2), (24, 8), (36, 4), (48, 2), (48, 8)] # (-60,-20), (-60,45), (0,0), (60,-20), (60,45)
    
    if num_initial_points == 3:
        return [(24, 2), (36, 8), (48, 2)] # (-60,-20), (0,45), (60,-20)
    
    raise ValueError(f"the num_initial_points {num_initial_points} is not predefined!")
    
class MergeHRTFDataset(Dataset):
    def __init__(self, hrtf_loader, left_hrtf, right_hrtf, num_initial_points, max_degree=21, apply_sht=True, transform=None):
        super(MergeHRTFDataset, self).__init__()
        self.left_hrtf = left_hrtf
        self.right_hrtf = right_hrtf
        self.apply_sht = apply_sht
        if apply_sht:
            self.num_initial_points = num_initial_points
            if hrtf_loader == 'hrtfdata':
                self.row_angles, self.column_angles = left_hrtf.row_angles, left_hrtf.column_angles
            elif hrtf_loader == 'hartufo':
                self.row_angles, self.column_angles = left_hrtf.fundamental_angles, left_hrtf.orthogonal_angles
            else:
                raise ValueError(f"unrecognized hrtf loader: {hrtf_loader}")
            self.num_row_angles, self.num_col_angles = len(self.row_angles), len(self.column_angles)
            self.num_radii = len(self.left_hrtf.radii)
            self.degree = max(1, int(np.sqrt(num_initial_points) - 1))
            self.max_degree = max_degree
        self.transform = transform
        self.selected_coords = get_sample_coords(num_initial_points)

    def __getitem__(self, index: int):
        try:
            left = self.left_hrtf[index]['features'][:, :, :, 1:]
            right = self.right_hrtf[index]['features'][:, :, :, 1:]
            sample_id = self.left_hrtf.subject_ids[index]
            merge = np.ma.concatenate([left, right], axis=3)
            selected_rows = [coord[0] for coord in self.selected_coords]
            selected_cols = [coord[1] for coord in self.selected_coords]
            
            if self.apply_sht:
                original_mask = np.all(np.ma.getmaskarray(left), axis=3)
                mask = np.ones((self.num_row_angles, self.num_col_angles, self.num_radii), dtype=bool)
                mask[selected_rows, selected_cols, :] = original_mask[selected_rows, selected_cols, :]
                lr_SHT = SphericalHarmonicsTransform(self.degree, self.row_angles,
                                                    self.column_angles,
                                                    self.left_hrtf.radii,
                                                    mask)
                lr_coefficient = torch.from_numpy(lr_SHT(merge)) # [num_coefficients, nbins]
                hr_SHT = SphericalHarmonicsTransform(self.max_degree, self.row_angles,
                                                    self.column_angles,
                                                    self.left_hrtf.radii,
                                                    original_mask)
                hr_coefficient = torch.from_numpy(hr_SHT(merge).T)

                if self.transform is not None:
                    mean_lr, mean_full = self.transform[0]
                    std_lr, std_full = self.transform[1]
                    lr_coefficient = (lr_coefficient - mean_lr) / std_lr
                    hr_coefficient = (hr_coefficient - mean_full) / std_full
                
                merge = torch.from_numpy(merge.data).permute(3, 2, 0, 1)  # nbins x r x w x h
                return {"lr_coefficient": lr_coefficient, "hr_coefficient": hr_coefficient,
                        "hrtf": merge, "mask": original_mask, "id": sample_id}
            else:
                merge = torch.from_numpy(merge.data).permute(3, 2, 0, 1)  # nbins x r x w x h
                lr_hrtf = merge[:, :, selected_rows, selected_cols]
                lr_hrtf = lr_hrtf.reshape(lr_hrtf.shape[0], -1).T # [num_points, nbins]
                return {"lr_hrtf": lr_hrtf, "hr_hrtf": merge, "id": sample_id}
        except Exception as e:
            print(f"[ERROR] Index {index} failed in __getitem__: {e}")
            raise
    def __len__(self):
        return len(self.left_hrtf)
    
class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)
    

class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        if self.batch_data is None:
            return None
        
        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v) and k not in {'mask', 'id'}:
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)