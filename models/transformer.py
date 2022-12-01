# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F



class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, is_skip_connection,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        if (is_skip_connection):
            tgt = tgt + self.dropout(tgt2)
        else:
            tgt = self.dropout(tgt2)

        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory, is_skip_connection,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        if (is_skip_connection):
            tgt = tgt + self.dropout(tgt2)
        else:
            tgt = self.dropout(tgt2)    

        return tgt

    def forward(self, tgt, memory, 
                is_skip_connection: True,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, is_skip_connection, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, is_skip_connection, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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

        # project input to smaller dimension reduce complexity
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)
    
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
        input_feature = self.input_proj(feature).flatten(2)
        input_feature = input_feature.permute(2, 0, 1)

        cluster_prototypes = self.get_initial_queries(input_feature)
        attn_mask, meaningless_clusters = self.get_attention_region(cluster_prototypes, input_feature)
        # [B, Q] -> [B, h, Q]
        multihead_attn_mask = meaningless_clusters.unsqueeze(1).repeat(1, self.num_heads, 1)
        # [B*h, Q, HW]
        cross_attn_mask = multihead_attn_mask.unsqueeze(2).repeat(1, 1, 1, input_feature.size(dim=0)).flatten(0, 1)
        # [B*h, Q, Q]
        self_attn_mask = multihead_attn_mask.unsqueeze(2).repeat(1, 1, 1, self.nqueries).flatten(0, 1)

        print(cross_attn_mask)
        print(self_attn_mask)
        for i in range(self.num_layers):
            # attention: cross-attention first
            is_skip_connection = (i == 0) # here we prevent the model use the clusters directly for class prediction
            
            cluster_prototypes = self.transformer_cross_attention_layers[i](
                cluster_prototypes, input_feature,
                is_skip_connection=is_skip_connection,
                memory_mask=cross_attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
            )

            cluster_prototypes = self.transformer_self_attention_layers[i](
                cluster_prototypes,
                tgt_mask=self_attn_mask,
                tgt_key_padding_mask=None,
            )
        
        cluster_prototypes = self.decoder_norm(cluster_prototypes)
        # [K, B, C] -> [B, K, C]
        cluster_prototypes = cluster_prototypes.permute(1, 0, 2)

        return cluster_prototypes
    
    def get_attention_region(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1) #[B, Q, C]
        features = mask_features.permute(1, 2, 0) #[B, Q, HW]
        print(decoder_output.size())
        print(features.size())
        outputs_mask = torch.einsum("bqc,bcf->bqf", decoder_output, features) # f = hw

        # [B, Q, HW] -> [B, Q] -> [B, Q, HW] -> [B, h, Q, HW] -> [B*h, Q, HW]
        attn_mask = (outputs_mask.sigmoid().flatten(2) > 0.5).bool()
        attn_mask = attn_mask.detach()

        meaningless_clusters = torch.all(attn_mask, dim=2)
        meaningless_clusters = meaningless_clusters.detach()

        return attn_mask, meaningless_clusters


def build_transformer_decoder(initial_clusters, args):
    return SSLTransformerDecoder(args.feature_dim,
                                args.feature_dim,
                                args.nheads,
                                args.num_queries,
                                args.dim_feedforward,
                                args.dec_layers,
                                args.pre_norm,
                                clusters=initial_clusters)
