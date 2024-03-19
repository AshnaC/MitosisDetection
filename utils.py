import json
import os
import  random
import numpy as np
import  argparse
import albumentations as Alb
from dpdl_defaults_cluster import sample_weight


def get_annotation_data(json_path):
    with open(json_path) as f:
        annotation_data = json.load(f)

    return annotation_data


def get_args(logs_path, check_point_path):
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-p', '--patchsize', type=int, default=512,
                        help='Patchsize - network will use pxp patches during training and inference')
    parser.add_argument('-b', '--batchsize', type=int, default=12, help='Batchsize')
    parser.add_argument('-nt', '--npatchtrain', type=int, default=10,
                        help='Number of patches per slide during training')
    parser.add_argument('-nv', '--npatchval', type=int, default=10,
                        help='Number of patches per slide during validation')
    parser.add_argument('-ne', '--nepochs', type=int, default=300, help='Total number of epochs for training')


    parser.add_argument('-m', '--mode', type=str, required=False,
                        help='Execution mode. Options: \'train\', \'val\', \'test\'', default='train', )
    parser.add_argument('-se', '--startepoch', type=int, default=0,
                        help='Starting epoch for training (remaining number of training epochs is nepochs-startepoch)')
    parser.add_argument('-lr', '--learningrate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--logdir', type=str, default=logs_path, help='Directory for lightning logs/checkpoints')
    # parser.add_argument('--resdir', type=str, default='./results', help='Directory for result files')
    parser.add_argument('-c', '--checkptfile', type=str, default=check_point_path,
                        help='Path to model file (necessary for reloading/retraining)')
    parser.add_argument('-s', '--seed', type=int, default='31415',
                        help='Seed for randomness, default=31415; set to -1 for random')

    args = parser.parse_args()
    return args

def get_patch(box, shape, level_dimensions, level):
    width, height = level_dimensions[level]
    x0 = box[0]
    y0 = box[1]
    x1 = box[2]
    y1 = box[3]
    w = x1 - x0
    h = y1 - y0
    x_start = x0 - (shape[0] / 2 - w / 2)
    y_start = y0 - (shape[1] / 2 - h / 2)
    if x_start < 0:
        x_start = 0
    if y_start < 0:
        y_start = 0
    if x_start + shape[0] > width:
        x_start = width - shape[0]
    if y_start + shape[1] > height:
        y_start = height - shape[1]
    return int(x_start), int(y_start)

MITOTIC =1

def skewed_sample_fn(targets, classes, shape, level_dimensions, level):
    is_rand_patch = random.choices([True, False], weights=sample_weight, k=1)[0]
    img_has_mitotic_figure = MITOTIC in classes
    width, height = level_dimensions[level]
    if is_rand_patch or not img_has_mitotic_figure:
        random_patch = np.random.randint(0, width - shape[0]), np.random.randint(0, height - shape[1])
        return random_patch
    else:
        boxes = targets[0]
        labels = [i for i, val in enumerate(targets[1]) if val == MITOTIC]
        label_index = random.choice(labels)
        patch = get_patch(boxes[label_index], shape, level_dimensions, level)
        return patch


def rand_sample_fn(targets, classes, shape, level_dimensions, level):
    width, height = level_dimensions[level]
    random_patch = np.random.randint(0, width - shape[0]), np.random.randint(0, height - shape[1])
    return random_patch