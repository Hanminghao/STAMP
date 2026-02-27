import h5py
import pytorch_lightning as pl
import re
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import os
import numpy as np
import torch
from math import ceil
from os.path import join
from torch.distributed import get_rank
from collections import defaultdict
from PIL import Image  
import random
from typing import List, Dict, Any


class BERTH5Dataset(Dataset):
    def __init__(self, file_paths: list, columns: list):
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.file_paths.sort(key=lambda x: int(re.search(r'tokens-(\d+)', x).group(1)))
        self.columns = columns

        self.lengths = []
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                self.lengths.append(len(f[self.columns[0]]))
        self.cum_lengths = np.cumsum(self.lengths)

        self.h5_files = {}

    def __len__(self):
        return int(self.cum_lengths[-1])
    
    def _get_file_and_index(self, idx):
        file_idx = int(np.searchsorted(self.cum_lengths, idx, side='right'))
        in_file_idx = idx if file_idx == 0 else idx - self.cum_lengths[file_idx-1]
        return self.file_paths[file_idx], in_file_idx

    def __getitem__(self, idx):
        file_path, in_file_idx = self._get_file_and_index(idx)
        if file_path not in self.h5_files:
            # 在每个 worker 内部缓存文件句柄
            self.h5_files[file_path] = h5py.File(file_path, 'r')
        f = self.h5_files[file_path]
        
        sample = {}
        for col in self.columns:
            if col == 'tokenized_gene':
                # 读取单个样本转换为 tensor
                sample[col] = torch.from_numpy(f[col][in_file_idx]).clone()
            # 如果有其他列，也可以在此添加
        return sample


class CLIPH5Dataset(Dataset):
    def __init__(self, file_path: List[str], columns: List[str], image_processor, ref_processor, vision_model_name):
        self.file_paths = file_path if isinstance(file_path, list) else [file_path]
        try:
            self.file_paths.sort(key=lambda x: int(re.search(r'tokens-(\d+)', x).group(1)))
        except:
            print('No need to sort')
        self.columns = columns
        self.image_processor = image_processor
        self.ref_processor = ref_processor
        self.vision_model_name = vision_model_name

        # 计算每个文件的样本数及累积长度
        self.lengths = []
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                self.lengths.append(len(f[self.columns[0]]))
        self.cum_lengths = np.cumsum(self.lengths)

        self.h5_files = {}

        self.paired_indices = self._create_paired_indices()

    def __len__(self):
        return int(self.cum_lengths[-1])

    def _get_file_and_index(self, idx):
        """通过索引快速定位文件和对应的内部索引。"""
        file_idx = int(np.searchsorted(self.cum_lengths, idx, side='right'))
        in_file_idx = idx if file_idx == 0 else idx - self.cum_lengths[file_idx - 1]
        return self.file_paths[file_idx], in_file_idx
    
    def _create_paired_indices(self):
        """生成每个样本的配对样本索引。"""
        slide_to_indices = defaultdict(list)
        total_idx = 0

        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as f:
                # 一次性读取整个 batch_slide_id
                slide_ids = f['batch_slide_id'][:]

            # 遍历当前文件中的所有 slide_id，避免频繁打开文件
            for in_file_idx, slide_id in enumerate(slide_ids):
                slide_to_indices[slide_id].append(total_idx)
                total_idx += 1

        return slide_to_indices

    def _load_sample(self, file_path, in_file_idx):
        """加载一个样本数据。"""
        if file_path not in self.h5_files:
            self.h5_files[file_path] = h5py.File(file_path, 'r')
        f = self.h5_files[file_path]

        return_dic = {}
        for col in self.columns:
            if col == 'images':
                image_data = f[col][in_file_idx]
                image_data = np.transpose(image_data, (1, 2, 0))
                image_data = Image.fromarray(image_data)
                return_dic[col] = self.image_processor(image_data)
                return_dic['ref'] = self.ref_processor(image_data)

            elif col == 'images_aug':
                image_data = f[col][in_file_idx]
                image_data = np.transpose(image_data, (1, 2, 0))
                image_data = Image.fromarray(image_data)
                return_dic[col] = self.image_processor(image_data)

            elif col == 'tokenized_gene':
                return_dic[col] = torch.from_numpy(f[col][in_file_idx]).clone()

            elif col in ['batch_slide_id', 'batch_dataset_id']:
                data_item = f[col][in_file_idx]
                return_dic[col] = torch.from_numpy(data_item) if isinstance(data_item, np.ndarray) else torch.tensor(data_item)

            elif col == 'normalized_expression':
                data_item = f[col][in_file_idx]
                expressions = torch.from_numpy(data_item) if isinstance(data_item, np.ndarray) else torch.tensor(data_item)
                return_dic[col] = expressions[self.hvg_idx]

            elif col == 'pos_label':
                data_item = f[col][in_file_idx]
                return_dic[col] = torch.from_numpy(data_item) if isinstance(data_item, np.ndarray) else torch.tensor(data_item)

        return return_dic
    
    def __getitem__(self, idx):
        # 获取主样本
        file_path, in_file_idx = self._get_file_and_index(idx)
        sample_1 = self._load_sample(file_path, in_file_idx)
        batch_slide_id = sample_1['batch_slide_id']
        # 获取配对样本（根据提前生成的索引）
        paired_idx = random.choice(self.paired_indices[batch_slide_id.item()])
        paired_file_path, paired_file_idx = self._get_file_and_index(paired_idx)
        sample_2 = self._load_sample(paired_file_path, paired_file_idx)

        # 返回主样本和配对样本
        for key in sample_2.keys():
            sample_1[key] = torch.stack([sample_1[key], sample_2[key]], dim=0)

        return sample_1    

class DownstreamCLIPH5Dataset(Dataset):
    def __init__(self, file_path: List[str], columns: List[str], image_processor, vision_model_name):
        self.file_paths = file_path if isinstance(file_path, list) else [file_path]
        try:
            self.file_paths.sort(key=lambda x: int(re.search(r'tokens-(\d+)', x).group(1)))
        except:
            print('No need to sort')
        self.columns = columns
        self.image_processor = image_processor
        self.vision_model_name = vision_model_name

        # 计算每个文件的样本数及累积长度
        self.lengths = []
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                self.lengths.append(len(f[self.columns[0]]))
        self.cum_lengths = np.cumsum(self.lengths)

        # 缓存每个文件句柄，适配多线程读取
        self.h5_files = {}

    def __len__(self):
        return int(self.cum_lengths[-1])

    def _get_file_and_index(self, idx):
        """通过索引快速定位文件和对应的内部索引。"""
        file_idx = int(np.searchsorted(self.cum_lengths, idx, side='right'))
        in_file_idx = idx if file_idx == 0 else idx - self.cum_lengths[file_idx - 1]
        return self.file_paths[file_idx], in_file_idx

    
    def _load_sample(self, file_path, in_file_idx):
        """加载一个样本数据。"""
        if file_path not in self.h5_files:
            self.h5_files[file_path] = h5py.File(file_path, 'r')
        f = self.h5_files[file_path]

        return_dic = {}
        for col in self.columns:
            if col == 'images':
                image_data = f[col][in_file_idx]
                image_data = np.transpose(image_data, (1, 2, 0))
                image_data = Image.fromarray(image_data)
                return_dic[col] = self.image_processor(image_data)

            elif col == 'tokenized_gene':
                return_dic[col] = torch.from_numpy(f[col][in_file_idx]).clone()

            elif col in ['batch_slide_id', 'batch_dataset_id']:
                data_item = f[col][in_file_idx]
                return_dic[col] = torch.from_numpy(data_item) if isinstance(data_item, np.ndarray) else torch.tensor(data_item)

            elif col == 'normalized_expression':
                data_item = f[col][in_file_idx]
                expressions = torch.from_numpy(data_item) if isinstance(data_item, np.ndarray) else torch.tensor(data_item)
                return_dic[col] = expressions[self.hvg_idx]

        return return_dic
    
    def __getitem__(self, idx):
        # 获取主样本
        file_path, in_file_idx = self._get_file_and_index(idx)
        sample = self._load_sample(file_path, in_file_idx)
        return sample


class DataModuleDistributed(pl.LightningDataModule):
    def __init__(
            self,
            path: str,
            columns: List[str],
            batch_size: int,
            world_size: int,
            sub_sample_frac: float = 1.0,
            image_processor=None,
            ref_processor=None,
            task_name: str = 'pretrain',
            vision_model_name: str = 'phikon', 
            sample_list: List[str] = None,
            num_workers: int = 8
    ):
        super().__init__()
        self.columns = columns
        self.task_name = task_name
        self.image_processor = image_processor
        self.ref_processor = ref_processor
        train_path = os.path.join(path, 'train')
        val_path = os.path.join(path, 'val')
        # 获取训练和验证数据文件路径
        self.file_paths_train = self._get_h5_files(train_path, world_size, sub_sample_frac, sample_list)
        self.file_paths_val = self._get_h5_files(val_path, world_size, sub_sample_frac, sample_list)
        self.batch_size = batch_size
        self.vision_model_name = vision_model_name
        self.num_workers = num_workers

    def _get_h5_files(self, base_path: str, world_size: int, sub_sample_frac: float = 1, sample_list: List[str] = None):

        files_devices = []
        files_list = os.listdir(base_path)
        if sample_list is not None:
            files_list = [file for file in files_list if file.split('.')[0] in sample_list]

        for device in range(world_size):
            files = [file for file in files_list if (file.endswith('.h5')) and ((int(file.split('.')[0].split('-')[1]) % world_size)==device)]
            files = [join(base_path, file) for file in sorted(files, key=lambda x: int(x.split('.')[0].split('-')[1]))]
            files.sort(reverse=True)
            files_devices.append(files[:ceil(sub_sample_frac * len(files))])

            
        return files_devices

    def train_dataloader(self):
        if self.task_name == 'align':
            datasets = [CLIPH5Dataset(fp, self.columns, self.image_processor, self.ref_processor, self.vision_model_name) for fp in self.file_paths_train]
        elif self.task_name == 'pretrain':
            datasets = [BERTH5Dataset(fp, self.columns) for fp in self.file_paths_train]
        return DataLoader(datasets[get_rank()], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        if self.task_name == 'align':
            datasets = [CLIPH5Dataset(fp, self.columns, self.image_processor, self.ref_processor, self.vision_model_name) for fp in self.file_paths_val]
        elif self.task_name == 'pretrain':
            datasets = [BERTH5Dataset(fp, self.columns) for fp in self.file_paths_val]
        return DataLoader(datasets[get_rank()], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)


class DownstreamDataModuleDistributed(pl.LightningDataModule):
    def __init__(
            self,
            path: str,
            columns: List[str],
            batch_size: int,
            image_processor=None,
            vision_model_name: str = 'phikon', 
            num_workers: int = 8,
            sample_list: List[str] = None,
            test_sample: str = '',
    ):
        super().__init__()
        self.columns = columns
        self.num_workers = num_workers
        self.vision_model_name = vision_model_name
        self.image_processor = image_processor
        self.batch_size = batch_size
        self.file_paths = self._get_h5_files(path, sample_list)
        if test_sample == '':
            self.datasets = DownstreamCLIPH5Dataset(self.file_paths, self.columns, self.image_processor, self.vision_model_name)
            self.train_size = int(0.9 * len(self.datasets))
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.datasets, [self.train_size, len(self.datasets) - self.train_size])
        else:
            val_file_path = self._get_h5_files(path, [test_sample])
            self.train_dataset = DownstreamCLIPH5Dataset(self.file_paths, self.columns, self.image_processor, self.vision_model_name)
            self.val_dataset = DownstreamCLIPH5Dataset(val_file_path, self.columns, self.image_processor, self.vision_model_name)
        

    def _get_h5_files(self, base_path: str, sample_list: List):

        files = [file for file in os.listdir(base_path) if (file.endswith('.h5')) and file.split('.h5')[0] in sample_list]
        files = [join(base_path, file) for file in files]
        return files

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=False) 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True,sampler=sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False) 
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, sampler=sampler, num_workers=self.num_workers)

class DownstreamDataset(Dataset):
    def __init__(
            self,
            file_paths: str,
            columns: List[str],
            image_processor,
            ):
        super().__init__()
        self.columns = columns
        self.image_processor = image_processor
        if isinstance(file_paths, str):
            self.h5_file = h5py.File(file_paths, 'r')
        elif isinstance(file_paths, list):
            self.h5_file = self.concat_h5_files(file_paths)

    def concat_h5_files(self, file_paths):
        total_rows = {col: 0 for col in self.columns}
        sample_shapes = {}
        for fp in file_paths:
            with h5py.File(fp, 'r') as f:
                for col in self.columns:
                    ds = f[col]
                    total_rows[col] += ds.shape[0]
                    if col not in sample_shapes:
                        sample_shapes[col] = ds.shape[1:]

        with h5py.File(file_paths[0], 'r') as f:
            dtypes = {col: f[col].dtype for col in self.columns}

        data_dict = {}
        for col in self.columns:
            data_dict[col] = np.empty((total_rows[col],) + sample_shapes[col], dtype=dtypes[col])

        current_index = {col: 0 for col in self.columns}
        for fp in file_paths:
            with h5py.File(fp, 'r') as f:
                for col in self.columns:
                    data = f[col][:]
                    nrows = data.shape[0]
                    data_dict[col][current_index[col]:current_index[col] + nrows] = data
                    current_index[col] += nrows

        return data_dict


    def __len__(self):
        return len(self.h5_file[self.columns[0]])
        
    def __getitem__(self, idx):

        return_dic = {}
        for col in self.columns:
            if col == 'images':
                image_data = self.h5_file[col][idx]
                image_data = np.transpose(image_data, (1, 2, 0))
                image_data = Image.fromarray(image_data)
                return_dic[col] = self.image_processor(image_data)

            elif col == 'tokenized_gene':
                return_dic[col] = torch.from_numpy(self.h5_file[col][idx]).clone()

            elif col == 'spot_label':
                try:
                    data_item = self.h5_file[col][idx]
                except:
                    data_item = -1
                return_dic[col] = np.array(data_item, dtype=int)

            elif col == 'spot_names':
                return_dic[col] = self.h5_file[col][idx].decode('utf-8')
        return return_dic