from abc import ABC
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from .general_eval import MVSDataset
from .blended_dataset import BlendedMVSDataset
from .dtu_yao import DTUMVSDataset
from .tt_dataset import TanksandTemplesDataset
from .eth3d_dataset import ETH3DDataset

np.random.seed(1234)


class DTULoader(ABC):
    def __init__(self):
        super().__init__()

    def get(self, data_path, data_list, mode, num_srcs, num_depths, interval_scale=1.0,
                 shuffle=True, seq_size=49, batch_size=1, fix_res=False, max_h=None, max_w=None,
                 dataset_eval='dtu', num_stages=3, num_workers=4, **kwargs):
        if (mode == 'train') or (mode == 'val'):
            self.mvs_dataset = DTUMVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                             shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, num_stages=num_stages)
        else:
            self.mvs_dataset = MVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                          shuffle=shuffle, seq_size=seq_size, batch_size=batch_size,
                                          max_h=max_h, max_w=max_w, fix_res=fix_res, dataset=dataset_eval, num_stages=num_stages)
        drop_last = True if mode == 'train' else False
        # super().__init__(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
        #                  num_workers=num_workers, pin_memory=True, drop_last=drop_last)

        return DataLoader(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, drop_last=drop_last)

    #     self.n_samples = len(self.mvs_dataset)

    # def get_num_samples(self):
    #     return len(self.mvs_dataset)


class BlendedLoader(ABC):
    def __init__(self):
        super().__init__()
    
    def get(self, data_path, data_list, mode, num_srcs, num_depths, interval_scale=1.0,
                 shuffle=True, seq_size=49, batch_size=1, fix_res=False, max_h=None, max_w=None, 
                 num_workers=4, num_stages=3, high_res=False, random_image_scale=False):
        if (mode == 'train') or (mode == 'val'):
            self.mvs_dataset = BlendedMVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                                 shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, 
                                                 num_stages=num_stages, high_res=high_res, random_image_scale=random_image_scale)
        else:
            self.mvs_dataset = MVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                          shuffle=shuffle, seq_size=seq_size, batch_size=batch_size,
                                          max_h=max_h, max_w=max_w, fix_res=fix_res, dataset='dtu')
        drop_last = True if mode == 'train' else False
        # super().__init__(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
        #                  num_workers=num_workers, pin_memory=True, drop_last=drop_last)

        return DataLoader(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, drop_last=drop_last)


class BlendedHighResLoader(ABC):
    def __init__(self):
        super().__init__()
    
    def get(self, data_path, data_list, mode, num_srcs, num_depths, interval_scale=1.0,
                 shuffle=True, seq_size=49, batch_size=1, fix_res=False, max_h=None, max_w=None, 
                 num_workers=4, num_stages=3, high_res=False, random_image_scale=False):
        if (mode == 'train') or (mode == 'val'):
            self.mvs_dataset = BlendedMVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                                 shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, 
                                                 num_stages=num_stages, high_res=high_res, random_image_scale=random_image_scale)
        else:
            self.mvs_dataset = MVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                          shuffle=shuffle, seq_size=seq_size, batch_size=batch_size,
                                          max_h=max_h, max_w=max_w, fix_res=fix_res, dataset='dtu')
        drop_last = True if mode == 'train' else False
        # super().__init__(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
        #                  num_workers=num_workers, pin_memory=True, drop_last=drop_last)

        return DataLoader(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, drop_last=drop_last)


class TanksandTemplesLoader(ABC):
    def __init__(self):
        super().__init__()
    
    def get(self, data_path, data_list, mode, num_srcs, num_depths, interval_scale=1.0,
                 shuffle=True, seq_size=49, batch_size=1, fix_res=False, max_h=None, max_w=None, num_workers=4, num_stages=3):
        if (mode == 'train') or (mode == 'val'):
            self.mvs_dataset = TanksandTemplesDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                                 shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, num_stages=num_stages)
        else:
            self.mvs_dataset = MVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                          shuffle=shuffle, seq_size=seq_size, batch_size=batch_size,
                                          max_h=max_h, max_w=max_w, fix_res=fix_res, dataset='dtu')
        drop_last = True if mode == 'train' else False
        # super().__init__(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
        #                  num_workers=num_workers, pin_memory=True, drop_last=drop_last)

        return DataLoader(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, drop_last=drop_last)


class ETH3DLoader(ABC):
    def __init__(self):
        super().__init__()
    
    def get(self, data_path, data_list, mode, num_srcs, num_depths, interval_scale=1.0,
                 shuffle=True, seq_size=49, batch_size=1, fix_res=False, max_h=None, max_w=None, num_workers=4, num_stages=3):
        if (mode == 'train') or (mode == 'val'):
            self.mvs_dataset = ETH3DDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                                 shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, num_stages=num_stages)
        else:
            self.mvs_dataset = MVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                          shuffle=shuffle, seq_size=seq_size, batch_size=batch_size,
                                          max_h=max_h, max_w=max_w, fix_res=fix_res, dataset='dtu')
        drop_last = True if mode == 'train' else False
        # super().__init__(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
        #                  num_workers=num_workers, pin_memory=True, drop_last=drop_last)

        return DataLoader(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, drop_last=drop_last)
