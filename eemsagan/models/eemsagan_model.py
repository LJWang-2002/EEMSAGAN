import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import USMSharp, get_root_logger
from collections import OrderedDict
from torch.nn import functional as F
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.losses import build_loss

@MODEL_REGISTRY.register()
class RemoteSensingEEMSAGANModel(SRGANModel):

    def __init__(self, opt):
        self.logger = get_root_logger()

        self._defer_weight_loading = True

        super(RemoteSensingEEMSAGANModel, self).__init__(opt)

        train_opt = self.opt.get('train', {})

        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None

        self._defer_weight_loading = False

        self.l1_gt_usm = opt.get('l1_gt_usm', False)
        self.percep_gt_usm = opt.get('percep_gt_usm', False)
        self.gan_gt_usm = opt.get('gan_gt_usm', False)
        self.edge_gt_usm = opt.get('edge_gt_usm', False)
        self.queue_size = opt.get('queue_size', 180)
        self.scale = opt.get('scale', 4)
        self.debug_mode = opt.get('debug_mode', False)

        if any([self.l1_gt_usm, self.percep_gt_usm, self.gan_gt_usm]):
            self.usm_sharpener = USMSharp().to(self.device)
        else:
            self.usm_sharpener = None

        self.safe_load_pretrained_weights()

    def safe_load_pretrained_weights(self):
        if not hasattr(self, 'opt') or 'path' not in self.opt:
            return

        path_opt = self.opt['path']
        if 'pretrain_network_g' in path_opt and path_opt['pretrain_network_g'] is not None:
            load_path = path_opt['pretrain_network_g']
            param_key = path_opt.get('param_key_g', 'params_ema')
            strict_load = path_opt.get('strict_load_g', True)

            self.load_network(self.net_g, load_path, strict_load, param_key)

    def load_network(self, network, load_path, strict=True, param_key='params'):
        if not hasattr(self, 'logger'):
            import logging
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(levelname)s: %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

        if load_path is None:
            return

        try:

            state_dict = torch.load(load_path, map_location=lambda storage, loc: storage)

            if param_key is not None and param_key in state_dict:
                state_dict = state_dict[param_key]

            network.load_state_dict(state_dict, strict=strict)

        except Exception as e:
            if strict:
                raise e
            else:
                self.logger.warning('Failed to load netwoek')

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            if self.queue_size == 0:
                return
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0

        if self.queue_size == 0:
            return

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

            if 'gt_path' in data:
                self.gt_path = data['gt_path']
            if 'lq_path' in data:
                self.lq_path = data['lq_path']

            self._dequeue_and_enqueue()

            if self.usm_sharpener is not None:
                self.gt_usm = self.usm_sharpener(self.gt)
            else:
                self.gt_usm = self.gt.clone()

            self.l1_gt = self.gt_usm if self.l1_gt_usm else self.gt
            self.percep_gt = self.gt_usm if self.percep_gt_usm else self.gt
            self.gan_gt = self.gt_usm if self.gan_gt_usm else self.gt
            self.edge_gt = self.gt_usm if self.edge_gt_usm else self.gt

        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                if self.usm_sharpener is not None:
                    self.gt_usm = self.usm_sharpener(self.gt)
                else:
                    self.gt_usm = self.gt.clone()

                self.l1_gt = self.gt_usm if self.l1_gt_usm else self.gt
                self.percep_gt = self.gt_usm if self.percep_gt_usm else self.gt
                self.gan_gt = self.gt_usm if self.gan_gt_usm else self.gt
                self.edge_gt = self.gt_usm if self.edge_gt_usm else self.gt


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        original_is_train = self.is_train
        self.is_train = False
        super(RemoteSensingEEMSAGANModel, self).nondist_validation(
            dataloader, current_iter, tb_logger, save_img)
        self.is_train = original_is_train

    def optimize_parameters(self, current_iter):

        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()

        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            if hasattr(self, 'cri_edge') and self.cri_edge is not None:
                l_g_edge = self.cri_edge(self.output, self.edge_gt)
                l_g_total += l_g_edge
                loss_dict['l_g_edge'] = l_g_edge

            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()

        real_d_pred = self.net_d(self.gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())

        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        l_d_total = (l_d_real + l_d_fake) / 2
        l_d_total.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        if self.opt['dist']:
            return self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):

        is_train = self.is_train
        self.is_train = False

        result = super().nondist_validation(dataloader, current_iter, tb_logger, save_img)

        self.is_train = is_train

        return result

    def save_training_state(self, epoch, current_iter):
        if current_iter % self.opt['logger']['save_checkpoint_freq'] == 0:
            self.save_network(self.net_g, 'net_g', current_iter)
            self.save_network(self.net_d, 'net_d', current_iter)

            super().save_training_state(epoch, current_iter)

        self.logger.info(f"save training: epoch={epoch}, iter={current_iter}")

    def save(self, epoch, current_iter):
        self.logger.info(f"save model: epoch={epoch}, iter={current_iter}")
        return super().save(epoch, current_iter)