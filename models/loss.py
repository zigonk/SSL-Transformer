import torch
from torch import nn
import torch.nn.functional as F

class SSLTLoss(nn.Module):
  """SSLoss computes the loss for self-supervised
  The final loss contains two losses:
    - CE loss: compute cross entropy loss with the target is pseudolabel
    - Contrastive learning loss: contrastive learning between each pair of prototype features of inter-image and intra-image
  """
  def __init__(self, num_classes, weight_dict) -> None:
    super().__init__()
    self.num_classes = num_classes
    self.weight_dict = weight_dict
  
  def loss_ce(self, preds):
    bs, k, _ = preds.shape
    # Create pseudo-label: ith cluster belongs to class i [B, K, K]
    tgt = F.one_hot(torch.arange(k).unsqueeze(0).repeat(bs, 1), k)
    loss_ce = F.cross_entropy(preds, tgt)
    return loss_ce
  def loss_contrast(self, preds):
    return 0
  def forward(self, preds, cluster_prototypes):
    losses = {}
    losses['ce'] = self.loss_ce(preds)
    losses['contrast'] = self.loss_contrast(cluster_prototypes)
    total_loss = 0
    for key in self.weight_dict.key():
      total_loss += self.weight_dict[key] * losses[key]
    return total_loss