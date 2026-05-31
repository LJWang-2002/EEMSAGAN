import importlib
from basicsr.utils import scandir
from os import path as osp
from .eemsanet_arch import EEMSANet

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
_arch_modules = [importlib.import_module(f'eemsagan.archs.{file_name}') for file_name in arch_filenames]
