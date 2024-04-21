import copy
import pickle

import numpy as np
from skimage import io

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate

from pathlib import Path

class WeatherShiftDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, mode=None, weather='all'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        self.weather = weather
        if mode == None:
            pass
        else:
            self.mode = mode

        self.file_list = []
        self.include_kitti_data(self.mode)

    def include_weathershift_data(self, mode):
        path2spilit = self.root_path.joinpath('splits').joinpath(mode).joinpath(self.weather+'.txt')
        with open(path2spilit, 'r') as f:
            lines = f.readlines()
            self.file_list = [l.replace(',','_')+'.bin' for l in lines]
            self.file_list.sort()

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self, index):
        return super().__getitem__(index)