import os
import sys
from os.path import join, dirname
import sys
# sys.path.insert(0, join(dirname(__file__), '.'))
sys.path.insert(0, '/data2/wanghejun/WeatherShift/main')
sys.path.append('/data2/wanghejun/WeatherShift/')

import numpy as np
import random
import copy

from OpenPCDet.pcdet.datasets.dataset import DatasetTemplate
from OpenPCDet.pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from OpenPCDet.pcdet.datasets.processor.data_processor import DataProcessor
from OpenPCDet.pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

# from ..dataset import DatasetTemplate
from pathlib import Path
from OpenPCDet.pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from main.utils import print_warning
from main.utils.point_cloud import read_pcd
from OpenPCDet.pcdet.datasets.weathershift.weathershift_utils import *
from OpenPCDet.pcdet.datasets.weathershift.weathershift_evaluation import *


class WeatherShiftDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, mode=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:depth_downsample_factor
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=None, training=training, root_path=root_path, logger=logger
        )
        self.root_path = (root_path if root_path is not None else Path(dataset_cfg['DATA_PATH']))
        self.class_names = class_names if class_names is not None else dataset_cfg['CLASS']

        # print(dataset_cfg['weather'])
        self.file_list = []

        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.include_weathershift_data(self.mode)

        self.grid_size = self.data_processor.grid_size

        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None

        # print(self.data_augmentor)

    def include_weathershift_data(self, mode):
        self.add_to_logger('Loading WeatherShift dataset')

        self.add_to_logger('Loading Real Data of %s' % (self.weather))
        
        path2spilit = self.root_path.joinpath('splits').joinpath(mode).joinpath(self.weather+'.txt')
        self.real_file_list = load_data(path2spilit, root=self.root_path, lidar=self.lidar, fake=self.fake, weather=self.weather, real=True)

        self.add_to_logger('Real samples for WeatherShift dataset of %s %s: %d' % (self.weather, self.lidar, len(self.real_file_list)))

        self.add_to_logger('Loading Fake Data of %s' % (self.weather))
        
        path2spilit = self.root_path.joinpath('splits').joinpath(mode).joinpath('clear_day'+'.txt')
        self.fake_file_list = load_data(path2spilit, root=self.root_path, lidar=self.lidar,fake=self.fake, weather=self.weather, real=False)

        self.add_to_logger('Fake samples for WeatherShift dataset of %s %s: %d' % (self.weather, self.lidar, len(self.fake_file_list)))

        self.add_to_logger('Total samples for WeatherShift dataset of %s %s: %d' % (self.weather, self.lidar, len(self.real_file_list)+len(self.fake_file_list)))

        self.class_names = self.dataset_cfg['CLASS']
        self.add_to_logger('There exist(s) %d categories' % (len(self.class_names)))

    
    @property
    def weather(self):
        return self.dataset_cfg['weather']
    
    @property
    def lidar(self):
        return self.dataset_cfg['lidar']
    
    @property
    def point_cloud_range(self):
        return np.array(self.dataset_cfg['POINT_CLOUD_RANGE'])
        
    
    @property
    def real_in_all(self):
        if self.training:
            return self.dataset_cfg['real_in_all']
        else:
            return 1
        
        
    @property
    def used_class(self):
        return self.dataset_cfg['CLASS']
    
    @property
    def fake(self):
        return self.dataset_cfg['fake']
    
    def add_to_logger(self, line):
        if self.logger is not None:
            self.logger.info(line)

    def __getitem__(self, index):
        fake_or_real = random.random() > self.real_in_all
        # fake_or_real = False

        if fake_or_real: # fake
            file_list = self.fake_file_list
        else: # real
            file_list = self.real_file_list

        index = index % len(file_list)
        filename = file_list[index]

        if fake_or_real: # fake
            path2pointcloud = self.root_path.joinpath(self.fake).joinpath(self.weather).joinpath(self.lidar).joinpath(filename+'.bin')
        else: # real
            path2pointcloud = self.root_path.joinpath('cloud').joinpath(self.lidar).joinpath(filename+'.bin')
        
        points = read_pcd(path2pointcloud) ## The Structure of the points

        path2label = self.root_path.joinpath('gt_labels/lidar3D').joinpath(filename+'.txt')
        labels = read_label(path2label)

        gt_boxes = [label['box'] for label in labels]
        gt_names = [label['name'] for label in labels]

        # if len(gt_boxes)>0:
        #     gt_boxes.append([0,0,0,0,0,0,0])
        #     gt_boxes.append('None')

        if len(gt_boxes)>0:
            input_dict = {
                'points': points,
                'frame_id': filename,
                'gt_boxes': gt_boxes,
                'gt_names': gt_names
            }
        else:
            input_dict = {
                'points': points,
                'frame_id': filename
            }
        data_dict = self.prepare_data(data_dict=input_dict)
        # print(data_dict['frame_id'])
        # print(len(data_dict['points']))

        return data_dict

    def __len__(self):
        if self.real_in_all <= 0:
            return len(self.fake_file_list)
        elif self.real_in_all >=1:
            return len(self.real_file_list)
        else:
            return len(self.real_file_list)+len(self.fake_file_list)
    
    def evaluation(self, det_annos, class_names, **kwargs):

        eval_det_annos = copy.deepcopy(det_annos)
        ap_result_str, ap_dict = get_official_eval_result(eval_det_annos, class_names)

        return ap_result_str, ap_dict
    
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path):
        """
        Args:
            batch_dict:
                'points'
                'frame_id'
                'gt_boxes'
                'gt_names'
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:
            pred_dict:
                'name'
                'alpha'
                'score'
                'boxes_lidar'

        """
        def generate_single_sample_dict(batch_index, box_dict, gt_boxes):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()


            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0])
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            if gt_boxes is None:
                return pred_dict
            
            gt_boxes = gt_boxes.cpu().numpy()
            pred_dict['gt_boxes'] = gt_boxes[:, :-1]
            pred_dict['gt_name'] = np.array(class_names)[gt_boxes[:, -1].astype(int) - 1]

            return pred_dict
    
        annos = []
        for index, box_dict in enumerate(pred_dicts):
            # print(box_dict)
            frame_id = batch_dict['frame_id'][index]
            gt_boxes = batch_dict['gt_boxes'][index] if 'gt_boxes' in batch_dict.keys() else None
            # print(gt_boxes)
            single_pred_dict = generate_single_sample_dict(index, box_dict, gt_boxes)

            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    print('name alpha score boxeslidar(x,y,z,dx,dy,dz,heading)',file=f)
                    name = single_pred_dict['name']
                    alpha = single_pred_dict['alpha']
                    score = single_pred_dict['score']
                    boxes_lidar = single_pred_dict['boxes_lidar']
                    
                    print('prediction:', file=f)
                    for idx in range(len(name)):
                        content = '%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' \
                              % (name[idx], alpha[idx],score[idx],
                                boxes_lidar[idx][0],boxes_lidar[idx][1],
                                boxes_lidar[idx][2],boxes_lidar[idx][3],
                                boxes_lidar[idx][4],boxes_lidar[idx][5],boxes_lidar[idx][6])
                        print(content, file=f)

                    print('ground truth:', file=f)
                    if 'gt_boxes' in single_pred_dict.keys():
                        gt_boxes = single_pred_dict['gt_boxes']
                        gt_name = single_pred_dict['gt_name']
                        for idx in range(len(gt_name)):
                            content = '%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' \
                                    % (gt_name[idx], 0 , 0,
                                        gt_boxes[idx][0],gt_boxes[idx][1],
                                        gt_boxes[idx][2],gt_boxes[idx][3],
                                        gt_boxes[idx][4],gt_boxes[idx][5],gt_boxes[idx][6])
                            print(content, file=f)


        return annos

    def get_infos(self, mode='train'):
        import concurrent.futures as futures

        infos = {}
        id = 0

        path2spilit = self.root_path.joinpath('splits').joinpath(mode).joinpath(self.weather+'.txt')
        real_file_list = load_data(path2spilit, root=self.root_path, lidar=self.lidar, weather=self.weather, real=True)

        path2spilit = path2spilit = self.root_path.joinpath('splits').joinpath(mode).joinpath('clear_day'+'.txt')
        fake_file_list = load_data(path2spilit, root=self.root_path, lidar=self.lidar, weather=self.weather, real=False)

        real_info = []
        for filename in real_file_list:
            info = {
                'path2pointcloud': self.root_path.joinpath('cloud').joinpath(self.lidar).joinpath(filename+'.bin'),
                'path2label': self.root_path.joinpath('gt_labels/lidar3D').joinpath(filename+'.txt'),
                'sample_id': id,
                'is_real' : True
            }
            id += 1
            real_info.append[info]

        fake_info = []
        for filename in fake_file_list:
            info = {
                'path2pointcloud': self.root_path.joinpath(self.fake).joinpath(self.weather).joinpath(self.lidar).joinpath(filename+'.bin'),
                'path2label': self.root_path.joinpath('gt_labels/lidar3D').joinpath(filename+'.txt'),
                'sample_id': id,
                'is_real' : False
            }
            id += 1
            fake_info.append[info]

        infos = {
            'real': real_info,
            'fake': fake_info,
            'all': real_info+fake_info,
        }

        return infos
    
    def create_groundtruth_database(self, save_path):
        import torch

        db_info_save_path = Path(save_path) / ('dbinfos.pkl')
        database_save_path = Path(save_path) / ('gt')


        database_save_path.mkdir(parents=True, exist_ok=True)
        save_path.mkdir(parents=True, exist_ok=True)

        all_db_infos = {}
        real = []
        fake = []
        infos = []
        id = 0

        for mode in ['train', 'valid', 'test']:
            path2spilit = self.root_path.joinpath('splits').joinpath(mode).joinpath(self.weather+'.txt')
            real = real + load_data(path2spilit, root=self.root_path, lidar=self.lidar, weather=self.weather, real=True)

            path2spilit = path2spilit = self.root_path.joinpath('splits').joinpath(mode).joinpath('clear_day'+'.txt')
            fake = fake + load_data(path2spilit, root=self.root_path, lidar=self.lidar, weather=self.weather, real=False)

        for filename in real:
            info = {
                'path2pointcloud': self.root_path.joinpath('cloud').joinpath(self.lidar).joinpath(filename+'.bin'),
                'path2label': self.root_path.joinpath('gt_labels/lidar3D').joinpath(filename+'.txt'),
                'sample_id': id,
                'is_real' : True
            }
            id += 1
            infos.append(info)

        for filename in fake:
            info = {
                'path2pointcloud': self.root_path.joinpath(self.fake).joinpath(self.weather).joinpath(self.lidar).joinpath(filename+'.bin'),
                'path2label': self.root_path.joinpath('gt_labels/lidar3D').joinpath(filename+'.txt'),
                'sample_id': id,
                'is_real' : False
            }
            id += 1
            infos.append(info)


        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            sample_idx = info['sample_id']
            points = read_pcd(info['path2pointcloud'])
            labels = read_label('path2label')

            names = [label['name'] for label in labels]
            gt_boxes = [label['gt_boxes'] for label in labels]

            num_obj = len(gt_boxes)
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                points_in_boxes = points[point_indices[i] > 0]

                points_in_boxes[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    points_in_boxes.tofile(f)

                if names[i] in self.used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': points_in_boxes.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        return all_db_infos



    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if 'gt_boxes' not in data_dict.keys():
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)
            
            
        data_dict = self.set_lidar_aug_matrix(data_dict)
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names).tolist()
            if len(selected) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            else:
                gt_boxes = [data_dict['gt_boxes'][i] for i in selected]
                gt_names = [data_dict['gt_names'][i] for i in selected]

                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.float32).reshape(-1, 1)
                gt_boxes = np.array(gt_boxes, dtype=np.float32)
                
                
            
                gt_boxes = np.concatenate((gt_boxes, gt_classes), axis=1)
                data_dict['gt_boxes'] = gt_boxes
                data_dict['gt_names'] = gt_names


        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)


        data_dict.pop('gt_names', None)

        return data_dict
    

def create_weathershift_infos(dataset_cfg, class_names, path2dataset, save_path, workers=4, spilit=[0.1,0.1]):
    dataset = WeatherShiftDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=None, training=False)

    print('---------------Start to generate data infos---------------')
    infos = []

    id = 0
    for mode in ['train', 'valid', 'test']:
        path2spilit = path2dataset.joinpath('splits').joinpath(mode).joinpath(weather+'.txt')
        real = real + load_data(path2spilit, root= path2dataset, lidar=lidar, weather=weather, real=True)

        path2spilit = path2dataset.joinpath('splits').joinpath(mode).joinpath('clear_day'+'.txt')
        fake = fake + load_data(path2spilit, root=path2dataset, lidar=lidar, weather=weather, real=False)

    for filename in real:
        info = {
            'path2pointcloud': path2dataset.joinpath('cloud').joinpath(lidar).joinpath(filename+'.bin'),
            'path2label': path2dataset.joinpath('gt_labels/lidar3D').joinpath(filename+'.txt'),
            'sample_id': id,
            'is_real' : True
        }
        id += 1
        infos.append(info)

    for filename in fake:
        info = {
            'path2pointcloud': path2dataset.joinpath(dataset_cfg['fake']).joinpath(weather).joinpath(lidar).joinpath(filename+'.bin'),
            'path2label': path2dataset.joinpath('gt_labels/lidar3D').joinpath(filename+'.txt'),
            'sample_id': id,
            'is_real' : False
        }
        id += 1
        infos.append(info)
    
    with open(path2dataset / ('all_infos.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    train_infos = []
    valid_infos = []
    test_infos = []

    all_len = len(infos)
    valid_len = all_len*spilit[0]
    test_len = all_len*spilit[1]
    train_len = all_len - valid_len - test_len

    assert train_len > 0, 'there must be something to train'
    
    while len(test_infos) < test_len:
        test_infos.append(infos[0])
        infos.pop(0)
    while len(valid_infos) < valid_len:
        valid_infos.append(infos[0])
        infos.pop(0)
    train_infos = infos
    
    with open(path2dataset / ('train_infos.pkl'), 'wb') as f:
        pickle.dump(train_infos, f)
    with open(path2dataset / ('valid_infos.pkl'), 'wb') as f:
        pickle.dump(valid_infos, f)
    with open(path2dataset / ('test_infos.pkl'), 'wb') as f:
        pickle.dump(test_infos, f)


    print('---------------Start create groundtruth database for data augmentation---------------')
    all_db_infos = dataset.create_groundtruth_database(save_path=save_path)
    for k, v in all_db_infos.items():
        print('Database %s: %d' % (k, len(v)))

    with open(path2dataset / ('db_infos.pkl'), 'wb') as f:
        pickle.dump(all_db_infos, f)

    print('---------------Data preparation Done---------------')

def main():
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='OpenPCDet/tools/cfgs/dataset_configs/weathershift_dataset.yaml', help='specify the config of dataset')
    args = parser.parse_args()

    try:
        yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
    except:
        yaml_config = yaml.safe_load(open(args.cfg_file))
    dataset_cfg = EasyDict(yaml_config)

    dataset = WeatherShiftDataset(
        dataset_cfg=dataset_cfg, 
        class_names=dataset_cfg['CLASS'],
        root_path=None,
        logger=common_utils.create_logger(),
        training=True,
        mode = 'train'
    )

    item = dataset[1]
    print(item)

if __name__ == '__main__':
    main()
    