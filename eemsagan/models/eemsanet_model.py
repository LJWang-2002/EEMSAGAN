import numpy as np
import random
import torch
import torch.nn.functional as F
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class RemoteSensingEEMSANetModel(SRModel):

    def __init__(self, opt):
        super(RemoteSensingEEMSANetModel, self).__init__(opt)
        self.gt_usm = opt.get('gt_usm', False)
        if self.gt_usm:
            self.usm_sharpener = USMSharp().cuda()
        else:
            self.usm_sharpener = None

        self.queue_size = opt.get('queue_size', 180)
        self.scale = opt.get('scale', 4)
        self.debug_mode = opt.get('debug_mode', False)


    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        b, c, h, w = self.lq.size()
        if self.queue_size == 0:
            return

        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0

        if self.queue_ptr == self.queue_size:
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()
            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train and self.opt.get('use_degradation', True):
            self.lq = data['lq'].to(self.device)
            self.gt = data['gt'].to(self.device)

            self._dequeue_and_enqueue()

            if self.gt_usm and self.usm_sharpener is not None:
                self.gt = self.usm_sharpener(self.gt)
                if self.debug_mode:
                    print(f"USM")

            self.lq = self.lq.contiguous()

        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                if self.opt.get('gt_usm', False) and hasattr(self, 'usm_sharpener'):
                    self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        original_is_train = self.is_train
        self.is_train = False
        super(RemoteSensingEEMSANetModel, self).nondist_validation(
            dataloader, current_iter, tb_logger, save_img)
        self.is_train = original_is_train