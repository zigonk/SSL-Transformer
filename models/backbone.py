import torch
from transformer import TransformerDecoder, TransformerDecoderLayer

class SLDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False) -> None:
        super().__init__()
        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = torch.nn.LayerNorm(d_model)
    
    def foward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

class SLNet(nn.Module):
    def __init__(self, base_encoder, cluster, args) -> None:
        super().__init__()
        self.encoder = base_encoder(low_dim=args.feature_dim, head_type=args.head_type)
        self.clusters = torch.nn.Embedding.from_pretrained(cluster)
        
        self.transformer_decoder = SLDecoder
    def forward(self, inputs):
        """Foward pass of the network

        Args:
            inputs: batched images, of shape [batch_size x 3 x H x W]
        
        Returns predictions of all queries:
            output: the classification logits for all queries.
                            Shape= [batch_size x num_queries x num_classes]
        """
        features = self.backbone(inputs)

