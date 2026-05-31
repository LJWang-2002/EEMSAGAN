import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

@LOSS_REGISTRY.register()
class MultiScaleSobelEdgeLoss(nn.Module):

    def __init__(self, scales=[1, 2, 4, 8], loss_weight=1.0,
                 loss_type='l1', normalized=True, eps=1e-6):
        super(MultiScaleSobelEdgeLoss, self).__init__()

        self.scales = scales
        self.loss_weight = loss_weight
        self.loss_type = loss_type.lower()
        self.normalized = normalized
        self.eps = eps

        sobel_kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_kernel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_kernel_x = sobel_kernel_x
        self.sobel_kernel_y = sobel_kernel_y

        self.register_buffer('sobel_x', self.sobel_kernel_x)
        self.register_buffer('sobel_y', self.sobel_kernel_y)

    def forward(self, pred, target, weight=None, **kwargs):
        edge_loss = 0.0
        batch_size = pred.shape[0]

        for scale in self.scales:
            if scale > 1:
                target_scaled = F.interpolate(target,
                                              scale_factor=1 / scale, mode='bilinear', align_corners=False)
                pred_scaled = F.interpolate(pred,
                                            scale_factor=1 / scale, mode='bilinear', align_corners=False)
            else:
                target_scaled = target
                pred_scaled = pred

            target_edges = self.compute_edges(target_scaled)
            pred_edges = self.compute_edges(pred_scaled)

            if self.loss_type == 'l1':
                scale_loss = F.l1_loss(pred_edges, target_edges, reduction='none')
            elif self.loss_type == 'l2':
                scale_loss = F.mse_loss(pred_edges, target_edges, reduction='none')

            if weight is not None:
                if weight.size(2) != scale_loss.size(2) or weight.size(3) != scale_loss.size(3):
                    weight = F.interpolate(weight, size=(scale_loss.size(2), scale_loss.size(3)),
                                           mode='bilinear', align_corners=False)
                scale_loss = scale_loss * weight

            edge_loss = edge_loss + scale_loss.mean()

        edge_loss = edge_loss / len(self.scales)

        return self.loss_weight * edge_loss

    def compute_edges(self, x):
        edges_list = []
        for c in range(x.shape[1]):
            x_channel = x[:, c:c + 1, :, :]

            grad_x = F.conv2d(x_channel, self.sobel_x, padding=1, stride=1)
            grad_y = F.conv2d(x_channel, self.sobel_y, padding=1, stride=1)

            edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)
            edges_list.append(edges)

        edges = torch.cat(edges_list, dim=1)

        if self.normalized:
            edges_max = edges.view(edges.shape[0], -1).max(dim=1)[0]
            edges_max = edges_max.view(-1, 1, 1, 1)
            edges = edges / (edges_max + self.eps)

        return edges

    def get_edge_maps(self, x):
        edge_maps = {}
        for scale in self.scales:
            if scale > 1:
                x_scaled = F.interpolate(x, scale_factor=1 / scale,
                                         mode='bilinear', align_corners=False)
            else:
                x_scaled = x
            edge_maps[f'scale_{scale}'] = self.compute_edges(x_scaled)

        return edge_maps


@LOSS_REGISTRY.register()
class AdaptiveEdgeLoss(MultiScaleSobelEdgeLoss):
    def __init__(self, scales=[1, 2, 4, 8], loss_weight=1.0,
                 loss_type='l1', normalized=True, eps=1e-6,
                 adaptive_weight=True, min_weight=0.1, max_weight=2.0):
        super().__init__(scales, loss_weight, loss_type, normalized, eps)

        self.adaptive_weight = adaptive_weight
        self.min_weight = min_weight
        self.max_weight = max_weight

    def forward(self, pred, target, weight=None, **kwargs):
        batch_size = pred.shape[0]

        if self.adaptive_weight and batch_size > 0:
            target_edges = self.compute_edges(target)
            edge_strength = target_edges.mean(dim=[1, 2, 3])

            adaptive_weights = torch.clamp(
                edge_strength * 5.0,
                min=self.min_weight,
                max=self.max_weight
            ).view(-1, 1, 1, 1)

            if weight is not None:
                weight = weight * adaptive_weights
            else:
                weight = adaptive_weights

        return super().forward(pred, target, weight, **kwargs)


@LOSS_REGISTRY.register()
class RemoteSensingEdgeLoss(MultiScaleSobelEdgeLoss):
    def __init__(self, scales=[1, 2, 4, 8], loss_weight=1.0,
                 loss_type='l1', normalized=True, eps=1e-6,
                 use_diagonal=True, edge_threshold=0.1):
        super().__init__(scales, loss_weight, loss_type, normalized, eps)

        self.use_diagonal = use_diagonal
        self.edge_threshold = edge_threshold

        if use_diagonal:
            sobel_kernel_45 = torch.tensor([
                [0, 1, 2],
                [-1, 0, 1],
                [-2, -1, 0]
            ], dtype=torch.float32).view(1, 1, 3, 3)

            sobel_kernel_135 = torch.tensor([
                [-2, -1, 0],
                [-1, 0, 1],
                [0, 1, 2]
            ], dtype=torch.float32).view(1, 1, 3, 3)

            self.register_buffer('sobel_45', sobel_kernel_45)
            self.register_buffer('sobel_135', sobel_kernel_135)

    def compute_edges(self, x):
        edges_list = []

        for c in range(x.shape[1]):
            x_channel = x[:, c:c + 1, :, :]

            grad_x = F.conv2d(x_channel, self.sobel_x, padding=1, stride=1)
            grad_y = F.conv2d(x_channel, self.sobel_y, padding=1, stride=1)

            if self.use_diagonal:
                grad_45 = F.conv2d(x_channel, self.sobel_45, padding=1, stride=1)
                grad_135 = F.conv2d(x_channel, self.sobel_135, padding=1, stride=1)

                edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_45 ** 2 + grad_135 ** 2 + self.eps)
            else:
                edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)

            if self.edge_threshold > 0:
                edges = torch.where(edges > self.edge_threshold, edges, torch.zeros_like(edges))

            edges_list.append(edges)

        edges = torch.cat(edges_list, dim=1)

        if self.normalized:
            edges_max = edges.view(edges.shape[0], -1).max(dim=1)[0]
            edges_max = edges_max.view(-1, 1, 1, 1)
            edges = edges / (edges_max + self.eps)

        return edges