from sklearn.utils import sparsefuncs
from scipy.sparse import issparse, csr_matrix
import numpy as np
import os
import torch
import random
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

def vit_grid_pooling(vit_output, grid_size=(3, 3), kernel_size=6, stride=4):
    batch_size, num_tokens, hidden_dim = vit_output.shape

    # (B, 196, D) -> (B, D, 14, 14)
    vit_output = vit_output.reshape(batch_size, 14, 14, hidden_dim).permute(0, 3, 1, 2)

    # (B, D, H_out * W_out * kernel_size * kernel_size)
    patches = vit_output.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    
    # (B, D, grid_h * grid_w, kernel_size * kernel_size)
    patches = patches.contiguous().view(batch_size, hidden_dim,
                                        grid_size[0], grid_size[1],
                                        -1)  # -1 automatically infers patch size

    # (B, D, grid_h * grid_w)
    pooled = patches.mean(dim=-1)

    return pooled.view(batch_size, -1, hidden_dim)  # (B, grid_h * grid_w, D)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def complete_masking(batch, p, n_tokens):
    padding_token = 1
    cls_token = 3

    indices = batch['tokenized_gene']

    indices = torch.where(indices == 0, torch.tensor(padding_token), indices) # 0 is originally the padding token, we change it to 1
    batch['tokenized_gene'] = indices

    mask = 1 - torch.bernoulli(torch.ones_like(indices), p) # mask indices with probability p
    spatial_mask = 1 - torch.bernoulli(torch.ones_like(indices), 1)
    
    masked_indices = indices * mask # masked_indices 
    masked_indices = torch.where(indices != padding_token, masked_indices, indices) # we just mask non-padding indices
    mask = torch.where(indices == padding_token, torch.tensor(padding_token), mask) # in the model we evaluate the loss of mask position 0
    spatial_mask = torch.where(indices == padding_token, torch.tensor(padding_token), spatial_mask) # in the model we evaluate the loss of mask position 0
    # so we make the mask of all PAD tokens to be 1 so that it's not taken into account in the loss computation
    
    # Notice for the following 2 lines that masked_indices has already not a single padding token masked
    masked_indices = torch.where(indices != cls_token, masked_indices, indices) # same with CLS, no CLS token can be masked
    mask = torch.where(indices == cls_token, torch.tensor(padding_token), mask) # we change the mask so that it doesn't mask any CLS token
    spatial_mask = torch.where(indices == cls_token, torch.tensor(padding_token), spatial_mask) # we change the mask so that it doesn't mask any CLS token
    
    # 80% of masked indices are masked
    # 10% of masked indices are a random token
    # 10% of masked indices are the real token

    random_tokens = torch.randint(10, n_tokens, size=masked_indices.shape, device=masked_indices.device)
    random_tokens = random_tokens * torch.bernoulli(torch.ones_like(random_tokens)*0.1).type(torch.int64) 

    masked_indices = torch.where(masked_indices == 0, random_tokens, masked_indices) # put random tokens just in the previously masked tokens

    same_tokens = indices.clone()
    same_tokens = same_tokens * torch.bernoulli(torch.ones_like(same_tokens) * 0.1).type(torch.int64)

    masked_indices = torch.where(masked_indices == 0, same_tokens, masked_indices) # put same tokens just in the previously masked tokens

    batch['masked_indices'] = masked_indices
    batch['mask'] = mask
    batch['spatial_mask'] = spatial_mask
    attention_mask = (masked_indices == padding_token)
    batch['attention_mask'] = attention_mask.type(torch.bool)

    return batch


def set_seed(seed):
    """
    Sets the seed for all libraries used.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

def sf_normalize(X):
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    # avoid zero devision error
    counts += counts == 0.
    # normalize to 10000. counts
    scaling_factor = 10000. / counts

    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X


def _sub_tokenize_data(x: csr_matrix, max_seq_len: int = -1, aux_tokens: int = 30):
    scores_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    
    for i in range(x.shape[0]):
        start_idx = x.indptr[i]  # Start of the non-zero elements for row i
        end_idx = x.indptr[i + 1]  # End of the non-zero elements for row i
        nonzero_indices = x.indices[start_idx:end_idx]  # Indices of non-zero elements
        nonzero_data = x.data[start_idx:end_idx]  # Values of non-zero elements
        
        # sorted_indices = nonzero_indices[np.argsort(-nonzero_data)][:max_seq_len]
        sorted_idx = np.argsort(-nonzero_data)[:max_seq_len]
        sorted_indices = nonzero_indices[sorted_idx]  # Get indices following the sorted order
        sorted_indices = sorted_indices + aux_tokens  # Adjust for auxiliary tokens
        
        if max_seq_len > 0:
            scores = np.zeros(max_seq_len, dtype=np.int32)
        else:
            scores = np.zeros(x.shape[1], dtype=np.int32)

        scores[:len(sorted_indices)] = sorted_indices.astype(np.int32)
        
        scores_final[i, :] = scores
    
    return scores_final


def tokenize_data(x: np.array, max_seq_len: int = None, aux_tokens: int = None):
    """Tokenize the input gene vector to a vector of 32-bit integers."""
    if type(x) == np.matrix:
        x = csr_matrix(x)
    scores_final = _sub_tokenize_data(x.tocsr(), max_seq_len, aux_tokens)

    return scores_final.astype('i4')


def group_spots_into_unique_batches(adata_slide, batch_size=9):
    array_row = np.array(adata_slide.obs['array_row'])
    array_col = np.array(adata_slide.obs['array_col'])

    spatial_coords = np.column_stack((array_row, array_col)).astype(int)
    n_spots = spatial_coords.shape[0]
    unassigned = set(range(n_spots))
    batches = []

    # Build a NearestNeighbors object with enough neighbors
    nbrs = NearestNeighbors(n_neighbors=n_spots)
    nbrs.fit(spatial_coords)

    # Greedy grouping: as long as the remaining points are enough for a full mini-batch,
    # strictly group them without repetition
    while len(unassigned) >= batch_size:
        # Pick a seed point from the unassigned set
        seed = unassigned.pop()
        group = [seed]

        # Get the ordering of all points relative to the seed (sorted by distance: near to far)
        distances, indices = nbrs.kneighbors([spatial_coords[seed]], n_neighbors=n_spots)
        indices = indices.tolist()

        # Add nearest unassigned points until reaching batch_size
        for idx in indices[0]:
            if len(group) >= batch_size:
                break
            if idx in unassigned:
                group.append(idx)
        # Remove all points in this group from unassigned to guarantee uniqueness
        for idx in group:
            unassigned.discard(idx)
        # Compute each point's mean distance to others and find the most central point
        group_coords = spatial_coords[group]
        mean_distances = np.mean(
            np.sqrt(((group_coords[:, None, :] - group_coords[None, :, :]) ** 2).sum(axis=2)),
            axis=1,
        )
        most_central_idx = np.argmin(mean_distances)  # Find the point with the smallest mean distance
        group.insert(0, group.pop(most_central_idx))  # Move the most central point to the first position
        batches.append(group)

    # For the remaining spots fewer than batch_size (cannot satisfy uniqueness for 9):
    if unassigned:
        # Now the remaining count is insufficient to form a full mini-batch by itself.
        # We relax the constraint: pick a seed and fill up to 9 with nearest neighbors (allow duplicates)
        remaining_list = list(unassigned)
        # Fill around the first remaining point as center
        seed = remaining_list[0]
        distances, indices = nbrs.kneighbors([spatial_coords[seed]], n_neighbors=batch_size)
        indices = indices.tolist()
        # First, include all remaining points (each appears at least once)
        group = list(unassigned)
        # If still fewer than batch_size, traverse indices to fill up to 9 (may reuse existing points)
        for idx in indices[0]:
            if len(group) >= batch_size:
                break
            if idx not in group:
                group.append(idx)
        # Compute each point's mean distance to others and find the most central point
        group_coords = spatial_coords[group]
        mean_distances = np.mean(
            np.sqrt(((group_coords[:, None, :] - group_coords[None, :, :]) ** 2).sum(axis=2)),
            axis=1,
        )
        most_central_idx = np.argmin(mean_distances)  # Find the point with the smallest mean distance
        group.insert(0, group.pop(most_central_idx))  # Move the most central point to the first position
        batches.append(group)

    return batches


def draw_mini(combined_adata, all_batches, slide_id):
    plt.clf()
    plt.close()
    # Extract spatial coordinates for all spots
    array_row = combined_adata.obs['array_row']
    array_col = combined_adata.obs['array_col']

    # Create a DataFrame to store each spot's spatial coordinates and its mini-batch info
    spots_data = []
    for batch_idx, batch in enumerate(all_batches):
        for spot in batch:
            x = int(array_row[spot])
            y = int(array_col[spot])  # Extract spatial coordinate
            spots_data.append({'x': x, 'y': y, 'batch': batch_idx})

    # Convert to DataFrame
    spots_df = pd.DataFrame(spots_data)
    center_spots_df = spots_df.groupby('batch').first().reset_index()
    # Generate a high-contrast color palette (use seaborn discrete palettes)
    num_batches = len(all_batches)
    palette = sns.color_palette("tab20", num_batches)  # Other palettes available, e.g., "tab20b", "hsv", etc.

    # Randomly shuffle color assignment to avoid adjacent batches having similar colors
    np.random.seed(42)  # Fix random seed for reproducibility
    shuffled_colors = np.random.permutation(palette)

    # Create a color dict mapping each batch to a unique color
    batch_colors = {batch: shuffled_colors[batch] for batch in range(num_batches)}

    # Plot each spot using its mini-batch color
    plt.figure(figsize=(12, 10))
    for batch, group in spots_df.groupby('batch'):
        plt.scatter(group['x'], group['y'], color=batch_colors[batch], label=f'Mini-batch {batch}', s=10)
    
    for batch, group in center_spots_df.groupby('batch'):
        plt.scatter(group['x'], group['y'], marker='*', color=batch_colors[batch], s=100) 

    plt.title('Spatial Distribution of Mini-batches')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')  # Place legend on the right
    plt.tight_layout()
    save_dir = 'distribution'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/mini_batch_distribution_{slide_id}.png')


def filter_batches_by_distance(all_batches, combined_adata, max_distance=10):
    """
    Filter out mini-batches where any pairwise distance exceeds max_distance.
    
    Args:
        all_batches: list, each element is an index list for a mini-batch.
        spatial_coords: ndarray, spatial coordinates of all spots, shape (n_spots, 2).
        max_distance: float, maximum allowed distance.
    
    Returns:
        filtered_batches: list, the filtered mini-batches.
    """
    filtered_batches = []
    max_distance2 = 0
    for batch in all_batches:
        # Extract spatial coordinates of this mini-batch
        array_row = combined_adata.obs['array_row']
        array_col = combined_adata.obs['array_col']
        coords = np.column_stack((array_row[batch], array_col[batch])).astype(int)
        
        # Compute pairwise Euclidean distances within this mini-batch
        distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
        
        # Keep the batch if all pairwise distances are <= max_distance
        if distances.max() > max_distance2:
            max_distance2 = distances.max()
        if np.all(distances <= max_distance):
            filtered_batches.append(batch)
    
    return filtered_batches


def get_safe_region(x_center, y_center, patch_size, image_w, image_h):
    valid_grids = []
    
    # Precompute all possible 3x3 grid positions
    for grid_id in range(9):
        row = grid_id // 3
        col = grid_id % 3
        
        # Compute candidate region coordinates
        x1 = x_center - (col * 2 + 1) * patch_size
        y1 = y_center - (row * 2 + 1) * patch_size
        x2 = x1 + 6 * patch_size
        y2 = y1 + 6 * patch_size
        
        # Boundary compliance check
        if x1 >= 0 and y1 >= 0 and x2 <= image_w and y2 <= image_h:
            valid_grids.append(grid_id)
    
    # Safe selection logic
    if not valid_grids:
        # Fallback: return the largest feasible region
        safe_x1 = max(0, x_center - 3*patch_size)
        safe_y1 = max(0, y_center - 3*patch_size)
        safe_x2 = min(image_w, x_center + 3*patch_size)
        safe_y2 = min(image_h, y_center + 3*patch_size)
        return (safe_x1, safe_y1), (safe_x2, safe_y2), -1  # -1 indicates a non-standard region
    
    # Randomly choose a valid position
    chosen = random.choice(valid_grids)
    row = chosen // 3
    col = chosen % 3
    
    # Final coordinate computation
    final_x1 = x_center - (col * 2 + 1) * patch_size
    final_y1 = y_center - (row * 2 + 1) * patch_size
    return (int(final_x1), int(final_y1)), (int(final_x1+6*patch_size), int(final_y1+6*patch_size)), chosen


def adjust_crop(top_left, bottom_right, img_width, img_height, patch_size):
    x_min, y_min = top_left
    x_max, y_max = bottom_right

    # Ensure x_min and y_min are not less than 0
    if x_min < 0:
        x_max = int(min(x_max - (x_min - 0), img_width))
        x_min = 0
    if y_min < 0:
        y_max = int(min(y_max - (y_min - 0), img_height))
        y_min = 0

    # Ensure x_max and y_max do not exceed image width and height
    if x_max > img_width:
        x_min = int(max(x_min - (x_max - img_width), 0))
        x_max = img_width
    if y_max > img_height:
        y_min = int(max(y_min - (y_max - img_height), 0))
        y_max = img_height


    return (x_min, y_min), (x_max, y_max)
