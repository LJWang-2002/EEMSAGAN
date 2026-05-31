import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os.path as osp
from basicsr.train import train_pipeline

import eemsagan.archs
import eemsagan.data
import eemsagan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
