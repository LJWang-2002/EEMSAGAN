import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from scipy import signal

@DATASET_REGISTRY.register()
class RemoteSensingEEMSAGANDataset(data.Dataset):

    def __init__(self, opt):
        super(RemoteSensingEEMSAGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(' ')[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        self.use_hflip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)
        self.use_vflip = opt.get('use_vflip', True)

        degradation_params = opt.get('degradation_params', {})

        self.scale = opt.get('scale', 4)

        blur_params = degradation_params.get('blur', {})
        self.kernel_size = blur_params.get('kernel_size', 11)
        self.sig_min = blur_params.get('sig_min', 0.1)
        self.sig_max = blur_params.get('sig_max', 1.2)
        self.rate_iso = blur_params.get('rate_iso', 0.9)

        noise_params = degradation_params.get('noise', {})
        self.variance_min = noise_params.get('variance_min', 0.0001)
        self.variance_max = noise_params.get('variance_max', 0.0009)
        self.noise_amplitude = noise_params.get('amplitude', 1.0)
        self.rate_cln = noise_params.get('rate_cln', 0.4)

        self.pulse_tensor = torch.zeros(11, 11).float()
        self.pulse_tensor[5, 5] = 1

    def isotropic_gaussian_kernel(self, kernel_size, sigma):
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        return kernel / np.sum(kernel)

    def anisotropic_gaussian_kernel(self, kernel_size, sigma_x, sigma_y, angle):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        sigma_matrix = np.array([
            [sigma_x ** 2 * cos_angle ** 2 + sigma_y ** 2 * sin_angle ** 2,
             (sigma_x ** 2 - sigma_y ** 2) * cos_angle * sin_angle],
            [(sigma_x ** 2 - sigma_y ** 2) * cos_angle * sin_angle,
             sigma_x ** 2 * sin_angle ** 2 + sigma_y ** 2 * cos_angle ** 2]
        ])

        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        xy = np.stack([xx, yy], axis=2)
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, axis=2))
        return kernel / np.sum(kernel)

    def random_gaussian_kernel(self):
        if random.random() < self.rate_iso:
            sigma = random.uniform(self.sig_min, self.sig_max)
            kernel = self.isotropic_gaussian_kernel(self.kernel_size, sigma)
            kernel_type = 'isotropic'
            sigma_x = sigma
        else:
            sigma_x = random.uniform(self.sig_min, self.sig_max)
            sigma_y = sigma_x * random.uniform(0.9, 1.1)
            angle = random.uniform(0, np.pi)
            kernel = self.anisotropic_gaussian_kernel(self.kernel_size, sigma_x, sigma_y, angle)
            kernel_type = 'anisotropic'

        return kernel, kernel_type, sigma_x

    def apply_blur(self, image, kernel):
        blurred = np.zeros_like(image)
        for i in range(3):
            blurred[:, :, i] = signal.convolve2d(
                image[:, :, i], kernel, mode='same', boundary='symm'
            )
        return blurred

    def bicubic_downsample(self, image, scale):
        h, w = image.shape[:2]
        new_h, new_w = h // scale, w // scale

        if image.dtype == np.float32:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        downsampled = cv2.resize(image_uint8, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return downsampled.astype(np.float32) / 255.0

    def add_enhanced_gaussian_noise(self, image):
        variance = random.uniform(self.variance_min, self.variance_max)
        std_dev = math.sqrt(variance) * self.noise_amplitude

        noise = np.random.normal(0, std_dev, image.shape)
        noisy_image = image + noise

        degradation_info = {
            'noise_variance': variance,
            'noise_amplitude': self.noise_amplitude,
            'effective_sigma': std_dev
        }

        return np.clip(noisy_image, 0, 1), degradation_info

    def apply_remote_sensing_degradation(self, hr_image):
        degradation_info = {
            'degradation_steps': 2
        }

        kernel, kernel_type, sigma = self.random_gaussian_kernel()
        blurred = self.apply_blur(hr_image, kernel)
        degradation_info.update({
            'kernel_type': kernel_type,
            'blur_sigma': sigma,
            'kernel_size': self.kernel_size
        })

        downsampled = self.bicubic_downsample(blurred, self.scale)
        degradation_info.update({
            'downsample_method': 'bicubic',
            'scale_factor': self.scale
        })

        if random.random() > self.rate_cln:
            final_image, noise_info = self.add_enhanced_gaussian_noise(downsampled)
            degradation_info.update(noise_info)
            degradation_info['noise_type'] = 'enhanced_gaussian'
        else:
            final_image = downsampled
            degradation_info.update({
                'noise_type': 'none',
                'noise_variance': 0.0,
                'noise_amplitude': 0.0,
                'effective_sigma': 0.0
            })

        return final_image, degradation_info

    def enhanced_augment(self, image):
        if self.use_hflip and random.random() < 0.5:
            image = image[:, ::-1, :]

        if self.use_vflip and random.random() < 0.5:
            image = image[::-1, :, :]

        if self.use_rot and random.random() < 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k, axes=(0, 1))

        return image

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                break
            except (IOError, OSError) as e:
                logger = get_root_logger()
                if retry > 1:
                    index = random.randint(0, self.__len__() - 1)
                    gt_path = self.paths[index]
                    time.sleep(1)
                retry -= 1
        else:
            raise IOError(f'no document: {gt_path}')

        img_gt = imfrombytes(img_bytes, float32=True)

        img_gt = self.enhanced_augment(img_gt)

        crop_size = self.opt.get('crop_size', 256)
        h, w = img_gt.shape[0:2]

        if img_gt.shape[0] < crop_size or img_gt.shape[1] < crop_size:
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w,
                                        cv2.BORDER_REFLECT_101)

        if img_gt.shape[0] > crop_size or img_gt.shape[1] > crop_size:
            h, w = img_gt.shape[0:2]
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)
            img_gt = img_gt[top:top + crop_size, left:left + crop_size, ...]

        lr_image, degradation_info = self.apply_remote_sensing_degradation(img_gt)

        img_gt_tensor = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        lr_image_tensor = img2tensor([lr_image], bgr2rgb=True, float32=True)[0]

        kernel = degradation_info.get('kernel', np.ones((self.kernel_size, self.kernel_size)))
        kernel_tensor = torch.FloatTensor(kernel)

        return {
            'gt': img_gt_tensor,
            'lq': lr_image_tensor,
            'kernel1': kernel_tensor,
            'kernel2': self.pulse_tensor,
            'sinc_kernel': self.pulse_tensor,
            'gt_path': gt_path,
            'degradation_info': degradation_info
        }

    def __len__(self):
        return len(self.paths)