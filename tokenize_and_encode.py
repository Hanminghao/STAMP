"""
Pipeline script that tokenizes a raw ST h5ad file and then encodes the
tokenized genes and paired image patches with the pretrained Stamp model.

It blends the data preparation logic from `tokenize_downstream.py` and the
model/weight handling from `train_stamp.py` into a single, runnable entry
point.
"""

import argparse
import copy
import os
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import cv2
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from PIL import Image

from config_files._config_finetune_stamp import sweep_config as clip_cfg
from config_files._config_finetune_stamp import spot_config as spot_cfg
from config_files._config_finetune_stamp import visual_config as visual_cfg
from models._stamp import Stamp
from models._utils import sf_normalize, tokenize_data, set_seed, get_safe_region, adjust_crop
from data.datamodules import DownstreamDataset

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize raw ST data then encode with Stamp")
    parser.add_argument('--sample', type=str, default='MISC1', help='Sample id (e.g., SPA123)')
    parser.add_argument('--raw_h5ad', type=str, default=None, help='Path to raw h5ad; defaults to ROOT_DIR/hest/st/<sample>.h5ad')
    parser.add_argument('--wsi', type=str, default=None, help='Path to WSI .tif; defaults to ROOT_DIR/hest/wsis/<sample>.tif')
    parser.add_argument('--meta_csv', type=str, default='HEST_v1_0_2.csv', help='CSV with magnification column')
    parser.add_argument('--root_dir', type=str, default='./hest', help='Base dir containing hest/{st,wsis}')
    parser.add_argument('--gene_dict', type=str, default='gene_name_id_dict.pkl', help='Pickle mapping gene_name->gene_id')
    parser.add_argument('--h5ad_head', type=str, default='model.h5ad', help='Vocabulary template h5ad')
    parser.add_argument('--mean_npy', type=str, default='Visium_mean.npy', help='Mean expression per gene')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for encoding')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers for encoding')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cuda, cuda:0, cpu); defaults to cuda with CPU fallback if unavailable')
    return parser.parse_args()


def load_tokenization_assets(args):
    with open(args.gene_dict, 'rb') as f:
        gene_name_id_dict = pickle.load(f)
    h5ad_head = sc.read_h5ad(args.h5ad_head)
    h5ad_head_vars = h5ad_head.var_names

    mean = np.load(args.mean_npy)
    mean = np.nan_to_num(mean)
    rounded_values = np.where((mean % 1) >= 0.5, np.ceil(mean), np.floor(mean))
    mean = np.where(mean == 0, 1, rounded_values)
    return gene_name_id_dict, h5ad_head_vars, mean


def compute_patch_size(meta_csv, sample_id, default_size=112):
    if not meta_csv or not os.path.exists(meta_csv):
        return default_size
    meta = pd.read_csv(meta_csv, index_col=1)
    if sample_id not in meta.index:
        return default_size
    magnification = meta.loc[sample_id, 'magnification']
    try:
        mag_int = int(str(magnification).rstrip('xX'))
    except ValueError:
        return default_size
    return default_size * mag_int / 20


def map_gene_ids(adata, gene_name_id_dict):
    current_var_names = adata.var_names
    new_var_names = list(current_var_names.copy())
    for i, gene_name in enumerate(current_var_names):
        if gene_name in gene_name_id_dict:
            new_var_names[i] = gene_name_id_dict[gene_name]
    adata.var_names = new_var_names
    if adata.var_names.duplicated().any():
        adata.var_names_make_unique()
    return adata


def tokenize_single_sample(args, gene_name_id_dict, h5ad_head_vars, mean, patch_size):
    sample_id = args.sample
    adata_path = args.raw_h5ad or os.path.join(args.root_dir, 'st', f'{sample_id}.h5ad')
    wsi_path = args.wsi or os.path.join(args.root_dir, 'wsis', f'{sample_id}.tif')

    adata = sc.read_h5ad(adata_path)
    adata.X = adata.X.astype('float32')
    if not sp.isspmatrix_csr(adata.X):
        if sp.isspmatrix(adata.X):
            adata.X = adata.X.tocsr()
        elif isinstance(adata.X, np.ndarray):
            adata.X = csr_matrix(adata.X)

    adata = map_gene_ids(adata, gene_name_id_dict)

    adata_filtered = adata[:, adata.var_names.isin(h5ad_head_vars)]
    h5ad_head = sc.read_h5ad(args.h5ad_head)
    adata = ad.concat([h5ad_head, adata_filtered], join='outer', axis=0)
    adata = adata[1:]  # drop head template row
    adata = adata[:, h5ad_head_vars]
    if not sp.isspmatrix_csr(adata.X):
        adata.X = csr_matrix(adata.X)
    adata.X = adata.X.astype('float32')
    adata.obs['spot_id'] = adata.obs.index
    adata.obs.reset_index(drop=True, inplace=True)
    adata_coords = adata.obsm['spatial']

    x = adata.X
    x = np.nan_to_num(x)
    x = sf_normalize(x)
    median_counts_per_gene = mean.copy()
    median_counts_per_gene += median_counts_per_gene == 0
    if adata.X.shape[1] != median_counts_per_gene.shape[0]:
        raise ValueError(f"Gene dimension mismatch: expression {adata.X.shape[1]} vs mean {median_counts_per_gene.shape[0]}")
    out = x / median_counts_per_gene.reshape((1, -1))
    tokenized_idx = tokenize_data(out, 4096, 30)

    wsi = cv2.imread(wsi_path)
    if wsi is None:
        raise FileNotFoundError(f"WSI not found: {wsi_path}")
    wsi_width, wsi_height = wsi.shape[1], wsi.shape[0]

    image_arrays, image_aug_arrays, pos_labels = [], [], []
    spot_names, spot_pos = [], []

    for i in tqdm(range(len(adata.obs)), ncols=0, total=len(adata.obs), desc=f'Tokenizing {sample_id}'):
        spot_id = adata.obs['spot_id'][i]
        row, col = adata.obs['array_row'][i], adata.obs['array_col'][i]
        x_center, y_center = adata_coords[i]
        top_left = (int(x_center - patch_size), int(y_center - patch_size))
        bottom_right = (int(x_center + patch_size), int(y_center + patch_size))
        top_left_aug, bottom_right_aug, pos_label = get_safe_region(*(x_center, y_center), patch_size, wsi_width, wsi_height)

        img = wsi[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        img_aug = wsi[top_left_aug[1]:bottom_right_aug[1], top_left_aug[0]:bottom_right_aug[0]]
        
        if img.size == 0:
            print(f"Warning: Patch out of image bounds for barcode {spot_id}. Skipping.")
            continue
        if img_aug.size == 0:
            print(f"Warning: AUG Patch out of image bounds for barcode {spot_id}. Adjusting.")
            top_left_aug, bottom_right_aug = adjust_crop(top_left_aug, bottom_right_aug, wsi.shape[0], wsi.shape[1], 3 * patch_size)
            img_aug = wsi[top_left_aug[1]:bottom_right_aug[1], top_left_aug[0]:bottom_right_aug[0]]

        img = cv2.resize(img, (224, 224))
        img_aug = cv2.resize(img_aug, (224, 224))
        img_array = np.transpose(np.array(img), (2, 0, 1))
        img_aug_array = np.transpose(np.array(img_aug), (2, 0, 1))

        image_arrays.append(img_array)
        image_aug_arrays.append(img_aug_array)
        pos_labels.append(pos_label)
        spot_names.append(spot_id)
        spot_pos.append((row, col))

    image_arrays = np.stack(image_arrays)
    image_aug_arrays = np.stack(image_aug_arrays)
    pos_labels_np = np.array(pos_labels, dtype=int)
    tokenized_idx = np.array(tokenized_idx)
    spot_names_np = np.array(spot_names, dtype='S')
    spot_pos_np = np.array(spot_pos, dtype=int)

    return tokenized_idx, image_arrays, spot_names_np, spot_pos_np, image_aug_arrays, pos_labels_np



def load_stamp_model(args, device):
    clip_config = copy.deepcopy(clip_cfg)
    spot_config = copy.deepcopy(spot_cfg)
    visual_config = copy.deepcopy(visual_cfg)

    model = Stamp(spot_config=spot_config,
                  visual_config=visual_config,
                  dim_output=clip_config['dim_output'],
                  temperature=clip_config['temperature'],
                  extract_layers=clip_config['extract_layers'],
                  function_layers=clip_config['function_layers'],
                  lr=clip_config['lr'],
                  warmup=clip_config['warmup'],
                  max_epochs=clip_config['max_epochs'],
                  pool=clip_config['pool'],
                  without_context=clip_config['without_context'],
                  margin=clip_config['margin'],
                  p=clip_config['p'],
                  eps=clip_config['eps'])

    ckpt_path = clip_config.get('pretrained_path')
    if not ckpt_path:
        raise ValueError("pretrained_path is not set in _config_finetune_stamp.py; please update it or supply a checkpoint.")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    missing_layers = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if missing_layers.missing_keys:
        print('Warning: missing keys when loading checkpoint:', missing_layers.missing_keys)

    model.eval()
    model.to(device)

    image_processor = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return model, image_processor


class _MemoryDataset(Dataset):
    """In-memory dataset for tokenized genes and images."""

    def __init__(self, tokens, images, spot_names, image_processor):
        self.tokens = tokens
        self.images = images
        self.spot_names = spot_names
        self.image_processor = image_processor

    def __len__(self):
        return len(self.spot_names)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image)
        return {
            'tokenized_gene': torch.from_numpy(self.tokens[idx]).clone(),
            'images': self.image_processor(image),
            'spot_names': self.spot_names[idx].decode('utf-8') if isinstance(self.spot_names[idx], bytes) else self.spot_names[idx],
        }


def encode_embeddings(token_source, model, image_processor, device, batch_size, num_workers, sample):
    # token_source can be a path (str) or a tuple of (tokens, images, spot_names)
    if isinstance(token_source, str):
        columns = ['images', 'tokenized_gene', 'spot_names']
        dataset = DownstreamDataset(token_source, columns, image_processor)
    else:
        # allow optional extra returns (spot_pos, aug images, pos_labels), we only use first three
        tokens, images, spot_names = token_source[:3]
        dataset = _MemoryDataset(tokens, images, spot_names, image_processor)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    gene_emb_list, img_emb_list, spot_name_list = [], [], []

    for batch in tqdm(loader, ncols=0, desc=f'Encoding {sample}'):
        spot_name_list.extend(batch['spot_names'])
        tokenized_gene = batch['tokenized_gene'].to(device, non_blocking=True).long()
        images = batch['images'].to(device, non_blocking=True)

        # align sequence length with context length used during training
        tokenized_gene = tokenized_gene[:, :model.spot_backbone.hparams.context_length]

        batch_for_model = {
            'tokenized_gene': tokenized_gene,
            'images': images,
        }

        with torch.no_grad():
            gene_feat = model.encode_gene(batch_for_model)
            patch_feat = model.encode_visual(batch_for_model, multi_scale=False)

            gene_emb = F.normalize(model.spot_projection(gene_feat), dim=-1).cpu()
            img_emb = F.normalize(model.patch_projection(patch_feat), dim=-1).cpu()

        gene_emb_list.append(gene_emb)
        img_emb_list.append(img_emb)

    gene_emb = torch.cat(gene_emb_list, dim=0).numpy()
    img_emb = torch.cat(img_emb_list, dim=0).numpy()
    spot_names_np = np.array(spot_name_list, dtype='S')

    return gene_emb, img_emb, spot_names_np


def main():
    args = parse_args()
    set_seed(42)

    if args.device:
        if 'cuda' in args.device and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    gene_name_id_dict, h5ad_head_vars, mean = load_tokenization_assets(args)

    patch_size = compute_patch_size(args.meta_csv, args.sample)
    token_source = tokenize_single_sample(args, gene_name_id_dict, h5ad_head_vars, mean, patch_size)


    model, image_processor = load_stamp_model(args, device)
    gene_emb, img_emb, spot_names = encode_embeddings(token_source, model, image_processor, device, args.batch_size, args.num_workers, args.sample)
    print(f'Generated embeddings — gene: {gene_emb.shape}, image: {img_emb.shape}, spots: {len(spot_names)}')


if __name__ == '__main__':
    main()
