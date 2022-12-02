# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from models.modules.attention_layers import CrossAttentionLayer, FFNLayer, SelfAttentionLayer


class SSLTransformerDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim: int,
                 nheads: int,
                 nqueries: int,
                 dim_feedforward: int,
                 dec_layers: int,
                 pre_norm: bool,
                 clusters: Tensor) -> None:
        super().__init__()

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # Using clusters as initial queries
        self.nqueries = nqueries
        self.query_feat = nn.Embedding.from_pretrained(clusters, freeze=False)

    def get_initial_queries(self, input_feature):
        _, bs, _ = input_feature.size()
        i = torch.arange(self.nqueries, device=input_feature.device)
        # [K, C] -> [B, K, C] -> [K, B, C]
        cluster_prototypes = self.query_feat(i)
        cluster_prototypes = cluster_prototypes.unsqueeze(1).repeat(1, bs, 1)
        return cluster_prototypes

    def forward(self, feature):
        """Compute prototype feature for clusters and prediction of prototype

        Args:
            feature: image feature (B x C x H x W)

        Returns:
            cluster prototypes (B x nqueries x nqueries)
        """
        # [B, C, H, W] -> [B, C, H*W] -> [H*W, B, C]
        input_feature = feature.flatten(2)
        input_feature = input_feature.permute(2, 0, 1)

        cluster_prototypes = self.get_initial_queries(input_feature)
        attn_mask, meaningless_clusters = self.get_attention_region(
            cluster_prototypes, input_feature)

        # [B, Q] -> [B, h, Q]
        multihead_attn_mask = meaningless_clusters.unsqueeze(
            1).repeat(1, self.num_heads, 1)
        # [B, h, Q] -> [B, h, Q, HW] -> [B*h, Q, HW]
        cross_attn_mask = multihead_attn_mask.unsqueeze(3).repeat(
            1, 1, 1, input_feature.size(dim=0)).flatten(0, 1)
        # [B, h, Q] -> [B, h, Q, Q] -> [B*h, Q, Q]
        self_attn_mask = multihead_attn_mask.unsqueeze(
            3).repeat(1, 1, 1, self.nqueries).flatten(0, 1)

        for i in range(self.num_layers):
            # attention: cross-attention first
            # here we prevent the model use the clusters directly for class prediction
            is_skip_connection = (i == 0)

            cluster_prototypes = self.transformer_cross_attention_layers[i](
                cluster_prototypes, input_feature,
                # is_skip_connection=is_skip_connection,
                # memory_mask=cross_attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
            )

            cluster_prototypes = self.transformer_self_attention_layers[i](
                cluster_prototypes,
                # tgt_mask=self_attn_mask,
                tgt_key_padding_mask=None,
            )

            cluster_prototypes = self.transformer_ffn_layers[i](
                cluster_prototypes
            )

        cluster_prototypes = self.decoder_norm(cluster_prototypes)
        # [K, B, C] -> [B, K, C]
        cluster_prototypes = cluster_prototypes.permute(1, 0, 2)

        return cluster_prototypes

    def get_attention_region(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [B, Q, C]
        features = mask_features.permute(1, 2, 0)  # [B, Q, HW]
        outputs_mask = torch.einsum(
            "bqc,bcf->bqf", decoder_output, features)  # f = hw

        # [B, Q, HW] -> [B, Q] -> [B, Q, HW] -> [B, h, Q, HW] -> [B*h, Q, HW]
        attn_mask = (outputs_mask.sigmoid().flatten(2) < 0.5).bool()
        attn_mask = attn_mask

        meaningless_clusters = torch.all(attn_mask, dim=2)
        meaningless_clusters = meaningless_clusters

        return attn_mask, meaningless_clusters


class AttentionDecoder(nn.Module):
    def __init__(self,
                 nqueries,
                 initial_clusters) -> None:
        super().__init__()
        self.nqueries = nqueries
        self.query_feat = nn.Embedding.from_pretrained(
            initial_clusters, freeze=False)
    def get_initial_queries(self, input_feature):
        _, bs, _ = input_feature.size()
        i = torch.arange(self.nqueries, device=input_feature.device)
        # [K, C] -> [B, K, C] 
        cluster_prototypes = self.query_feat(i)
        cluster_prototypes = cluster_prototypes.unsqueeze(0).repeat(bs, 1, 1)
        return cluster_prototypes

    def forward(self, input_features):
        pass


def build_decoder(initial_clusters, args, dec_type='trans'):
    if (dec_type == 'trans'):
        return SSLTransformerDecoder(args.feature_dim,
                                     args.feature_dim,
                                     args.nheads,
                                     args.num_queries,
                                     args.dim_feedforward,
                                     args.dec_layers,
                                     args.pre_norm,
                                     clusters=initial_clusters)
    if (dec_type == 'attn'):
        return AttentionDecoder(clusters=initial_clusters)
