import numpy as np
import itertools
class_to_name = {
    0 : 'PassengerCar',
    1 : 'Pedestrian',
    2 : 'PassengerCar_is_group',
    3 : 'Obstacle',
    4 : 'DontCare',
    5 : 'Pedestrian_is_group',
    6 : 'Vehicle',
    7 : 'LargeVehicle',
    8 : 'RidableVehicle',
    9 : 'RidableVehicle_is_group',
    10 : 'Vehicle_is_group',
    11 : 'LargeVehicle_is_group'
}
   
name_to_class = {v: n for n, v in class_to_name.items()}
NUM_OF_CLASSES = 11

Challenges = {
    'Easy': 0.1,
    'Moderate': 0.3,
    'Hard':0.5
}

class Box():
    def __init__(self, box_params, box_name, box_score):
        self.x = box_params[0]
        self.y = box_params[1]
        self.z = box_params[2]
        self.dx = box_params[3]
        self.dy = box_params[4]
        self.dz = box_params[5]
        self.heading = box_params[6]

        self.name = box_name
        self.class_id = name_to_class[box_name]
        self.score = box_score
    
    @staticmethod
    def group_init(params, names, scores=None, gt = False):
        if params is None or names is None:
            return []
        if gt == False and scores is not None:
            boxes = [Box(params[i],names[i], scores[i]) for i in range(len(params))]
        else:
            boxes = [Box(params[i],names[i], 1.0) for i in range(len(params))]
        return boxes
    
    @property
    def center(self):
        return np.array([self.x,self.y,self.z])
    
    @property
    def size(self):
        return np.array([self.dx,self.dy,self.dz])
    
    @property
    def corners(self):
        return compute_corners(self.center, self.size, self.heading)

    def __mul__(self, other):
        return self.is_there_any_intersection(other)

    def is_there_any_intersection(self, other):
        corners_self = self.corners
        corners_other = other.corners

        for corner_self in corners_self:
            if self.is_point_inside_box(corner_self, other):
                return 1.0

        for corner_other in corners_other:
            if other.is_point_inside_box(corner_other, self):
                return 1.0

        return 0.0

    def is_point_inside_box(self, point, box):
        corners = box.corners
        min_x = min(corners[:, 0])
        max_x = max(corners[:, 0])
        min_y = min(corners[:, 1])
        max_y = max(corners[:, 1])
        min_z = min(corners[:, 2])
        max_z = max(corners[:, 2])

        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y and min_z <= point[2] <= max_z:
            return True
        else:
            return False

def compute_corners(center, size, heading):
    # 计算边界框的8个角点
    w, l, h = size
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)
    corners = np.array([
        [w/2, l/2, h/2],
        [-w/2, l/2, h/2],
        [-w/2, -l/2, h/2],
        [w/2, -l/2, h/2],
        [w/2, l/2, -h/2],
        [-w/2, l/2, -h/2],
        [-w/2, -l/2, -h/2],
        [w/2, -l/2, -h/2]
    ])
    rotation_matrix = np.array([
        [cos_heading, -sin_heading, 0],
        [sin_heading, cos_heading, 0],
        [0, 0, 1]
    ])
    rotated_corners = np.dot(corners, rotation_matrix.T)
    translated_corners = rotated_corners + center
    return translated_corners


def get_official_eval_result(annos, current_classes, PR_detail_dict=None):

    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    '''
    anno: 
        ['name']
        ['alpha']
        ['score']
        ['boxes_lidar']
        ['gt_boxes']
        ['gt_name']
    '''
    precision = {
        'Easy': np.array([0] * NUM_OF_CLASSES,dtype=float),
        'Moderate': np.array([0] * NUM_OF_CLASSES,dtype=float),
        'Hard': np.array([0] * NUM_OF_CLASSES,dtype=float)
    }
    recall = {
        'Easy': np.array([0] * NUM_OF_CLASSES,dtype=float),
        'Moderate': np.array([0] * NUM_OF_CLASSES,dtype=float),
        'Hard': np.array([0] * NUM_OF_CLASSES,dtype=float)
    }
    ap = {
        'Easy': np.array([0] * NUM_OF_CLASSES,dtype=float),
        'Moderate': np.array([0] * NUM_OF_CLASSES,dtype=float),
        'Hard': np.array([0] * NUM_OF_CLASSES,dtype=float)
    }
    f1 = {
        'Easy': np.array([0] * NUM_OF_CLASSES,dtype=float),
        'Moderate': np.array([0] * NUM_OF_CLASSES,dtype=float),
        'Hard': np.array([0] * NUM_OF_CLASSES,dtype=float)
    }
    nums = {
        'Easy': np.array([0] * NUM_OF_CLASSES,dtype=float),
        'Moderate': np.array([0] * NUM_OF_CLASSES,dtype=float),
        'Hard': np.array([0] * NUM_OF_CLASSES,dtype=float)
    }
    mAP = {
        'Easy': 0,
        'Moderate': 0,
        'Hard': 0
    }

    for anno in annos:
        if 'gt_boxes' in anno.keys() and 'gt_name' in anno.keys():
            pass
        else:
            anno['gt_boxes'] = None
            anno['gt_name'] = None

        pred = Box.group_init(params=anno['boxes_lidar'], names=anno['name'], scores=anno['score'])
        _gt_ = Box.group_init(params=anno['gt_boxes'], names=anno['gt_name'], gt=True)

        for c in Challenges.keys():
            precision_anno, recall_anno, ap_anno, f1_anno, nums_anno = calculate_precision_recall(pred, _gt_, confidence_threshold=Challenges[c])
            precision[c] = (precision[c]*nums[c] + precision_anno*nums_anno)/(nums[c] + nums_anno + 1e-6)
            recall[c] = (recall[c]*nums[c] + recall_anno*nums_anno)/(nums[c] + nums_anno + 1e-6)
            ap[c] = (ap[c]*nums[c] + ap_anno*nums_anno)/(nums[c] + nums_anno + 1e-6)
            f1[c] = (f1[c]*nums[c] + f1_anno*nums_anno)/(nums[c] + nums_anno + 1e-6)
            nums[c] = nums[c] + nums_anno

    result=''
    ret_dict = {}
    for c in Challenges.keys():
        mAP[c] = np.sum(ap[c]*nums[c] / (nums[c]+1e-6))
        result += f'===== {c} ===== \n'
        result += f'mAP={mAP[c]:.4f}\n'
        ret_dict[f'%s/mAP' % (c)] = mAP[c]
        for cls in range(NUM_OF_CLASSES):
            name = class_to_name[cls]
            p = precision[c][cls]
            r = recall[c][cls]
            n = int(nums[c][cls])
            result += f'Category:{name}({n:d}),precison={p:.4f},recall={r:.4f}\n'
            ret_dict[f'%s/%s_precision' % (c, name)] = p
            ret_dict[f'%s/%s_recall' % (c, name)] = r
    return result, ret_dict

def calculate_precision_recall(detected_boxes, true_boxes, num_classes=NUM_OF_CLASSES, confidence_threshold=0.5):
    # Initialize variables to store precision and recall values for each class
    precision = [0] * num_classes
    recall = [0] * num_classes
    ap = [0] * num_classes
    f1 = [0] * num_classes
    nums = [0] * NUM_OF_CLASSES

    # Iterate over each class
    for class_id in range(num_classes):
        # Filter detected and true boxes for the current class
        nums[class_id] = len([box for box in detected_boxes if box.class_id == class_id])

        detected_class_boxes = [box for box in detected_boxes if box.class_id == class_id and box.score >= confidence_threshold]
        true_class_boxes = [box for box in true_boxes if box.class_id == class_id]

        # Compute precision and recall for the current class
        true_positives, precision_curve, recall_curve = compute_TP(detected_class_boxes,true_class_boxes)

        if len(precision_curve) == 0:
            precision[class_id] = 0
        else:
            precision[class_id] = precision_curve[-1]

        if len(recall_curve) == 0:
            recall[class_id] = 0
        else:
            recall[class_id] = recall_curve[-1]

        ap[class_id] = compute_ap(recall_curve,precision_curve)

        # Compute F1 score for the current class
        if precision[class_id] + recall[class_id] == 0:
            f1[class_id] = 0
        else:
            f1[class_id] = 2 * (precision[class_id] * recall[class_id]) / (precision[class_id] + recall[class_id])

    precision = np.array(precision,dtype=float)
    recall = np.array(recall,dtype=float)
    ap = np.array(ap,dtype=float)
    f1 = np.array(f1,dtype=float)
    nums = np.array(nums,dtype=float)

    return precision, recall, ap, f1, nums

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (np.array).
        precision: The precision curve (np.array).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[:-1] != mrec[1:])[0]     #错位比较，前一个元素与其后一个元素比较,np.where()返回下标索引数组组成的元组

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_TP(detected, true):
    true_positives = 0
    sorted_boxes = sorted(detected, key=lambda x: x.score, reverse=True)
    precision_curve = []
    recall_curve = []
    num_true_boxes = len(true)

    for i, box in enumerate(sorted_boxes):
        for true_box in true:
            if true_box*box>=1.0:
                true.remove(true_box)
                true_positives += 1
                break
        
        precision_curve.append(true_positives / (i + 1))
        recall_curve.append(true_positives/(num_true_boxes+1e-6))

    return true_positives, precision_curve, recall_curve
        