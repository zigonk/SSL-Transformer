from typing import Optional

import torch
from torch import nn, Tensor
from models.loss import SSLTLoss
from models.transformer import build_decoder


class SSLTNet(nn.Module):
    def __init__(self, args, base_encoder,
                 criterion: Optional[SSLTLoss] = None,
                 initial_clusters: Optional[Tensor] = None) -> None:
        super().__init__()
        self.backbone = base_encoder
        self.decoder = build_decoder(initial_clusters, args)

        # output FFNs
        self.class_embed = nn.Linear(args.feature_dim, args.num_queries)
        self.criterion = criterion

    def forward(self, inputs):
        """Foward pass of the network

        Args:
            inputs: batched images, of shape [batch_size x 3 x H x W]

        Returns a dict with following keys:
            pred_logits: prediction of prototypes [B x K x K]
            cluster_prototypes: prototype features [B x K x C]
        """
        features = self.backbone(inputs)
        if self.training:
            k = 0.8
        else:
            k = 0
        cluster_prototypes = self.decoder(features, k)
        pred_logits = self.class_embed(cluster_prototypes)

        if self.training:
            return self.criterion(pred_logits, features)
        
        CAM_batch = torch.einsum('bqc,bchw->bqhw', cluster_prototypes, features)

        out = {
            'pred_logits': pred_logits,
            'cluster_prototypes': cluster_prototypes,
            'cam': CAM_batch,
        }

        return out
