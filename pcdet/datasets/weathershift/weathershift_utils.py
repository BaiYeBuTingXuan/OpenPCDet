import numpy as np
from pathlib import Path
import io as sysio
def read_label(path2label):
    # [name, cx, cy, cz, dx, dy, dz, heading]
    objects_list = []
    with open(path2label, 'r') as f:
        lines = f.readlines()
        for line in lines:
            obj = {}

            line = line.strip()
            line = line.split()
            
            obj['name'] = line[0]

            cx = float(line[1])
            cy = float(line[2])
            cz = float(line[3])

            dx = float(line[4])
            dy = float(line[5])
            dz = float(line[6])

            heading = float(line[7])

            obj['box'] = [cx,cy,cz,dx,dy,dz,heading]

            objects_list.append(obj)
            
    return objects_list

def load_data(path2spilit, root, lidar, weather, fake, real=True):
    file_list = []
    if real == True:
        root_pc = root.joinpath('cloud').joinpath(lidar)
    else:
        root_pc = root.joinpath(fake).joinpath(weather).joinpath(lidar)
    root_label = root.joinpath('gt_labels/lidar3D')

    with open(path2spilit, 'r') as f:
        lines = f.readlines()
        for l in lines:
            filename = l.strip().replace(',','_')
            path2pointcloud = root_pc.joinpath(filename+'.bin')
            path2label = root_label.joinpath(filename+'.txt')
            # if path2pointcloud.exists() and path2label.exists():
            file_list.append(filename)
    file_list.sort()

    return file_list

def get_template_prediction(num_samples):
    ret_dict = {
        'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
        'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
        'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
        'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
        'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
    }
    return ret_dict

def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()

# '/home/wanghejun/Desktop/wanghejun/WeatherShift/main/OpenPCDet/data'

def arrange_weathershift_data(path2dataset, save_path, weather, lidar, spilit = [0.1,0.1]):
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
            'path2pointcloud': path2dataset.joinpath('fake').joinpath(weather).joinpath(lidar).joinpath(filename+'.bin'),
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













    
