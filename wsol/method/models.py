"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, dim, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone
        self.backbone_dim = dim
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x, labels=None, return_cam=False):
        logits = self.backbone(x)
        features = self.contrastive_head(logits['pre_logits'])
        features = F.normalize(features, dim = 1)
        feat_classes = logits['logits']

        if return_cam:
            return self.backbone(x, labels, return_cam)

        return features, feat_classes

