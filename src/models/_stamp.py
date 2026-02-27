import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from typing import List
from torch import optim
from ._utils import complete_masking, vit_grid_pooling
from ._visiumformer_saptial import CosineWarmupScheduler, VisiumformerSatial
import torch.nn.functional as F
import numpy as np
import timm  

CLS_TOKEN = 2

class Stamp(pl.LightningModule):
    def __init__(self, 
                 spot_config: dict,
                 visual_config: dict,
                 dim_output: int,
                 temperature: float,
                 extract_layers: List[int],
                 function_layers: str,
                 lr: float, 
                 warmup: int, 
                 max_epochs: int,
                 pool: int = 'mean',
                 without_context: bool = True,
                 margin: float = 0.5,
                 p: int = 2,
                 eps: float = 1e-6,
                 ):
        """
        Args:
            backbone (pl.LightningModule): pretrained model
            baseline (bool): just for wandb logger to know it's baseline; baseline here means non-trained Transformer
            extract_layers (int): which hidden representations use as input for the linear layer
            function_layers (str): which function use to combine the hidden representations used
            lr (float): learning rate
            warmup (int): number of steps that the warmup takes
            max_epochs (int): number of steps until the learning rate reaches 0
            pool (str): could be None, 'cls' or 'mean'. CLS adds a token that gathers info of the sequence, mean just averages all tokens

        """
        super().__init__()
        self.spot_backbone = VisiumformerSatial(dim_model=spot_config['dim_model'], 
                                                nheads=spot_config['nheads'], 
                                                dim_feedforward=spot_config['dim_feedforward'], 
                                                nlayers=spot_config['nlayers'],
                                                dropout=spot_config['dropout'],
                                                batch_first=spot_config['batch_first'], 
                                                n_tokens=spot_config['n_tokens'],
                                                context_length=spot_config['context_length'],
                                                autoregressive=spot_config['autoregressive'],
                                                pool=spot_config['pool'],
                                                learnable_pe=spot_config['learnable_pe'],
                                                spatial_aware=spot_config['spatial_aware'],
                                                masking_p=0.0)
        
        self.spot_backbone.hparams.masking_p = 0.0
        self.spot_projection = nn.Linear(self.spot_backbone.hparams.dim_model, dim_output)

        if spot_config['pretrained_path'] is not None:
            checkpoint = torch.load(spot_config['pretrained_path'], map_location='cpu')
            model_state_dict = self.spot_backbone.state_dict()
            filtered_state_dict = {
                k: v for k, v in checkpoint['state_dict'].items()
                if k in model_state_dict and model_state_dict[k].shape == v.shape
            }
            self.spot_backbone.load_state_dict(filtered_state_dict, strict=False)
            print("Did not load the following keys:", set(model_state_dict.keys()) - set(filtered_state_dict.keys()))

        else:
            print("No pretrained path provided for spot backbone, using random initialization.")
            
        if visual_config['model_name'] == 'uni':
            visual_model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
            if visual_config['pretrained_path'] is not None:
                visual_model.load_state_dict(torch.load(visual_config['pretrained_path'], map_location="cpu"), strict=True)
            from models._prompt_learner import VisionPromptLearnerUni
            self.visual_backbone = VisionPromptLearnerUni(visual_model)
            self.patch_projection = nn.Linear(visual_model.embed_dim, dim_output)
            self.region_projection = nn.Linear(visual_model.embed_dim, dim_output)
            self.positioning_projection = nn.Linear(visual_model.embed_dim, dim_output)

        self.visual_backbone.train()
        self.visual_backbone_name = visual_config['model_name']
        
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.save_hyperparameters(ignore=['backbone'])
        
    def encode_gene(self, batch):
        # x -> size: batch x (context_length) x 1
        batch = complete_masking(batch, 0.0, self.spot_backbone.hparams.n_tokens+5)
        masked_indices = batch['masked_indices'].to(self.spot_backbone.device)
        attention_mask = batch['attention_mask'].to(self.spot_backbone.device)
        token_embedding = self.spot_backbone.embeddings(masked_indices)

        if self.spot_backbone.hparams.learnable_pe:
            pos_embedding = self.spot_backbone.positional_embedding(self.spot_backbone.pos.to(token_embedding.device))
            embeddings = self.spot_backbone.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.spot_backbone.positional_embedding(token_embedding)

        hidden_repr = []

        for i in range(len(self.spot_backbone.encoder.layers)):
            layer = self.spot_backbone.encoder.layers[i]
            embeddings = layer(embeddings, is_causal=self.spot_backbone.autoregressive, src_key_padding_mask=attention_mask) # bs x seq_len x dim
            if i in self.hparams.extract_layers:
                hidden_repr.append(embeddings)

        if self.hparams.function_layers == "mean":
            combined_tensor = torch.stack(hidden_repr, dim=-1)
            transformer_output = torch.mean(combined_tensor, dim=-1)  # bs x seq_len x dim
        if self.hparams.function_layers == "sum":
            combined_tensor = torch.stack(hidden_repr, dim=-1)
            transformer_output = torch.sum(combined_tensor, dim=-1)  # bs x seq_len x dim
        if self.hparams.function_layers == "concat":
            transformer_output = torch.cat(hidden_repr, dim=2)
                        

        if self.hparams.without_context:
            cls_prediction = transformer_output[:, 3:, :].mean(1)
        else:
            cls_prediction = transformer_output.mean(1)

        return cls_prediction
            
    def encode_visual(self, batch, multi_scale=False):
        # x -> size: batch x (context_length) x 1
        image = batch['images']
        
        patch_features, _ = self.visual_backbone(image)
        
        if multi_scale:
            image_aug = batch['images_aug']
            ref = batch['ref']
            region_features, ref_features, region_tokens = self.visual_backbone(image_aug, ref)
            return patch_features, region_features, ref_features, region_tokens
        
        else:
            return patch_features
    
    def forward(self, batch):
        spot_features = self.encode_gene(batch)

        patch_features, region_features, ref_features, region_tokens = self.encode_visual(batch, multi_scale=True)
        
        spot_embeddings = self.spot_projection(spot_features)
        patch_embeddings = self.patch_projection(patch_features)
        region_embeddings = self.region_projection(region_features)

        pooled_region_tokens = vit_grid_pooling(region_tokens)
        ref_embeddings = self.positioning_projection(ref_features)
        pooled_region_embeddings = self.positioning_projection(pooled_region_tokens)

        return spot_features, spot_embeddings, \
               patch_features, patch_embeddings, \
               region_embeddings, ref_embeddings, pooled_region_embeddings
    
    def training_step(self, batch, *args, **kwargs):
        # get the embeddings & features [embeddings: after projection, features: before projection]
        spot_features, spot_embeddings, patch_features, patch_embeddings, \
        region_embeddings, ref_embeddings, pooled_region_embeddings = self.forward(batch)
        # normalized features
        spot_embeddings = F.normalize(spot_embeddings, dim=-1)
        patch_embeddings = F.normalize(patch_embeddings, dim=-1)
        region_embeddings = F.normalize(region_embeddings, dim=-1)

        # cosine similarity as logits, patch <-> spot
        logit_scale = self.logit_scale.exp()
        logits_per_patch = logit_scale * patch_embeddings @ spot_embeddings.t()
        logits_per_spot = logits_per_patch.t()
        labels1 = torch.arange(logits_per_patch.shape[0], device=self.device, dtype=torch.long)
        patch_spot_loss = (
            F.cross_entropy(logits_per_patch, labels1) +
            F.cross_entropy(logits_per_spot, labels1)
        ) / 2
        self.log('train_patch_spot_loss', patch_spot_loss.mean(), sync_dist=True, prog_bar=False, reduce_fx='mean')

        # cosine similarity as logits, patch -> region
        logits_patch_region = logit_scale * patch_embeddings @ region_embeddings.t()
        labels2 = torch.arange(logits_patch_region.shape[0], device=self.device, dtype=torch.long)
        patch_region_loss = F.cross_entropy(logits_patch_region, labels2)
        self.log('train_patch_region_loss', patch_region_loss.mean(), sync_dist=True, prog_bar=False, reduce_fx='mean')

        # cosine similarity as logits, region <-> spot
        logits_per_region2 = logit_scale * region_embeddings @ spot_embeddings.t()
        logits_per_spot2 = logits_per_region2.t()
        labels3 = torch.arange(logits_per_region2.shape[0], device=self.device, dtype=torch.long)
        region_spot_loss = (
            F.cross_entropy(logits_per_region2, labels3) +
            F.cross_entropy(logits_per_spot2, labels3)
        ) / 2
        self.log('train_region_spot_loss', region_spot_loss.mean(), sync_dist=True, prog_bar=False, reduce_fx='mean')

        # patch positioning loss (consine simlarity with CE loss)
        logits_patch_position = logit_scale * torch.sum(F.normalize(pooled_region_embeddings, dim=-1) * F.normalize(ref_embeddings, dim=-1).unsqueeze(1), dim=2) 
        labels4 = batch['pos_label']
        positioning_loss = F.cross_entropy(logits_patch_position, labels4)
        self.log('train_positioning_loss', positioning_loss.mean(), sync_dist=True, prog_bar=False, reduce_fx='mean')

        loss = patch_spot_loss + patch_region_loss + region_spot_loss + positioning_loss
        self.log('train_loss', loss.mean(), sync_dist=True, prog_bar=True, reduce_fx='mean')

        
        return loss.mean()
    
    def validation_step(self, batch, *args, **kwargs):
        # get the embeddings & features [embeddings: after projection, features: before projection]
        spot_features, spot_embeddings, patch_features, patch_embeddings, \
        region_embeddings, ref_embeddings, pooled_region_embeddings = self.forward(batch)
        # normalized features
        spot_embeddings = F.normalize(spot_embeddings, dim=-1)
        patch_embeddings = F.normalize(patch_embeddings, dim=-1)
        region_embeddings = F.normalize(region_embeddings, dim=-1)


        # cosine similarity as logits, patch <-> spot
        logit_scale = self.logit_scale.exp()
        logits_per_patch = logit_scale * patch_embeddings @ spot_embeddings.t()
        logits_per_spot = logits_per_patch.t()
        labels1 = torch.arange(logits_per_patch.shape[0], device=self.device, dtype=torch.long)
        patch_spot_loss = (
            F.cross_entropy(logits_per_patch, labels1) +
            F.cross_entropy(logits_per_spot, labels1)
        ) / 2
        self.log('val_patch_spot_loss', patch_spot_loss.mean(), sync_dist=True, prog_bar=False, reduce_fx='mean')

        # cosine similarity as logits, patch -> region
        logits_patch_region = logit_scale * patch_embeddings @ region_embeddings.t()
        labels2 = torch.arange(logits_patch_region.shape[0], device=self.device, dtype=torch.long)
        patch_region_loss = F.cross_entropy(logits_patch_region, labels2)
        self.log('val_patch_region_loss', patch_region_loss.mean(), sync_dist=True, prog_bar=False, reduce_fx='mean')

        # cosine similarity as logits, region <-> spot
        logits_per_region2 = logit_scale * region_embeddings @ spot_embeddings.t()
        logits_per_spot2 = logits_per_region2.t()
        labels3 = torch.arange(logits_per_region2.shape[0], device=self.device, dtype=torch.long)
        region_spot_loss = (
            F.cross_entropy(logits_per_region2, labels3) +
            F.cross_entropy(logits_per_spot2, labels3)
        ) / 2
        self.log('val_region_spot_loss', region_spot_loss.mean(), sync_dist=True, prog_bar=False, reduce_fx='mean')

        # patch positioning loss (consine simlarity with CE loss)
        logits_patch_position = logit_scale * torch.sum(F.normalize(pooled_region_embeddings, dim=-1) * F.normalize(ref_embeddings, dim=-1).unsqueeze(1), dim=2) 
        labels4 = batch['pos_label']
        positioning_loss = F.cross_entropy(logits_patch_position, labels4)
        self.log('val_positioning_loss', positioning_loss.mean(), sync_dist=True, prog_bar=False, reduce_fx='mean')

        loss = patch_spot_loss + patch_region_loss + region_spot_loss + positioning_loss
        self.log('val_loss', loss.mean(), sync_dist=True, prog_bar=True, reduce_fx='mean')

        
        return loss.mean()
    
    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                batch[key] = value.view(-1, *value.shape[2:])
        
        data_key = 'tokenized_gene'

        if self.hparams.pool == 'cls': # Add cls token at the beginning of the set
            x = batch[data_key]
            cls = torch.ones((x.shape[0], 1), dtype=torch.int32, device=x.device)*CLS_TOKEN # CLS token is index 2
            x = torch.cat((cls, x), dim=1) # add CLS
            batch[data_key] = x

        batch['tokenized_gene'] = batch['tokenized_gene'][:, :self.spot_backbone.hparams.context_length]
        
        return batch
    
    def configure_optimizers(self):
        
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.001)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_epochs=self.hparams.max_epochs)
        
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        
    def initialize_weights(self):

        for name, param in self.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean=0.0, std=0.02)
    
    def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()