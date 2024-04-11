import copy
import pickle

import numpy as np
from skimage import io

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate

from pathlib import Path
from .....utils import print_warning
from .....utils.point_cloud import read_pcd
from .weathershift_utils import *

class WeatherShiftDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, mode=None, weather='all', lidar='lidar_hdl64_strongest'):
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
        self.root_path = (root_path if root_path is not None else Path(dataset_cfg['DATA_PATH']))
        self.class_names = class_names if class_names is not None else dataset_cfg['CLASS']
        self.mode = mode if mode is not None else self.mode

        self.weather = weather
        self.lidar = lidar

        self.file_list = []
        self.include_weathershift_data(self.mode)

        self.len = len(self.file_list)

    def include_weathershift_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')

        file_list = []
        path2spilit = self.root_path.joinpath('splits').joinpath(mode).joinpath(self.weather+'.txt')
        with open(path2spilit, 'r') as f:
            lines = f.readlines()
            for l in lines:
                filename = l.replace(',','_')
                path2pointcloud = self.root_path.joinpath(self.weather).joinpath(self.lidar).joinpath(filename+'.bin')
                path2label = self.root_path.joinpath('gt_labels/lidar3D').joinpath(filename+'.txt')
                if path2pointcloud.exists() and path2label.exists():
                    file_list.append(filename)
                else:
                    continue
        file_list.sort()

        self.file_list = file_list
        self.class_names = self.dataset_cfg['CLASS']

        if self.logger is not None:
            self.logger.info('Total samples for WeatherShift dataset of %s %s: %d' % (self.weather, self.lidar, len(file_list)))

    @property
    def mode(self):
        return 'train' if self.training else 'valid'

    def __getitem__(self, index):
        index = index % self.len
        filename = self.file_list[index]

        path2pointcloud = self.root_path.joinpath(self.weather).joinpath(self.lidar).joinpath(filename+'.bin')
        points = read_pcd(path2pointcloud) ## The Structure of the points

        path2label = self.root_path.joinpath('gt_labels/lidar3D').joinpath(filename+'.txt')
        label = read_label(path2label)


        input_dict = {
            'points': points,
            'frame_id': filename,
            'gt_boxes': label[:]['box'],
            'gt_names': label[:]['name']
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def __len__(self):
        return self.len
    
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        print_warning('Undone Function generate_prediction_dicts is used')
        return pred_dicts


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='./config.yaml', help='specify the config of dataset')
    args = parser.parse_args()

    try:
        yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
    except:
        yaml_config = yaml.safe_load(open(args.cfg_file))
    dataset_cfg = EasyDict(yaml_config)

    dataset = WeatherShiftDataset(
        dataset_cfg=dataset_cfg, 
        class_names=None,
        root_path=None,
        logger=common_utils.create_logger(),
        training=True,
        weather='dense_fog_day'
    )

    item = dataset[1]
    print(item)