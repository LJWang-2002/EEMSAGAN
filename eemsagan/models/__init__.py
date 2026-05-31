import importlib
from basicsr.utils import scandir
from os import path as osp

model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
_model_modules = [importlib.import_module(f'eemsagan.models.{file_name}') for file_name in model_filenames]
