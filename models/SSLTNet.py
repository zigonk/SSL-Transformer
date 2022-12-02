from typing import Optional

from torch import nn
from models.transformer import build_decoder

class SSLTNet(nn.Module):
    def __init__(self, args, base_encoder,
                 initial_clusters,
                 criterion) -> None:
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
        cluster_prototypes = self.decoder(features)
        pred_logits = self.class_embed(cluster_prototypes)

        if self.training:
            return self.criterion(pred_logits, features)

        out = {
            'pred_logits': pred_logits,
            'cluster_prototypes': cluster_prototypes,
        }

        return out
