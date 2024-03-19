"""
        Non maxima suppression on WSI results
        Marc Aubreville, Pattern Recognition Lab, FAU Erlangen-Nürnberg, 2019
"""

import numpy as np
from sklearn.neighbors import KDTree

def non_max_suppression_by_distance(boxes, scores, radius: float = 25, det_thres=None):
    if det_thres is not None:  # perform thresholding
        to_keep = scores > det_thres
        boxes = boxes[to_keep]
        scores = scores[to_keep]
        center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
        center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2


    X = np.dstack((center_x, center_y))[0]
    # Uses KD Tree for querying neighbors
    tree = KDTree(X)

    sorted_ids = np.argsort(scores)[::-1]
    ids_to_keep = []
    ind = tree.query_radius(X, r=radius)

    while len(sorted_ids) > 0:
        ids = sorted_ids[0]
        ids_to_keep.append(ids)
        sorted_ids = np.delete(sorted_ids, np.in1d(sorted_ids, ind[ids]).nonzero()[0])

    return boxes[ids_to_keep]


def nms(result_boxes, scores, det_thres=None):
    arr = np.array(result_boxes)
    scores = np.array(scores)
    if arr is not None and isinstance(arr, np.ndarray) and (arr.shape[0] == 0):
        return result_boxes
    if arr.shape[0] > 0:
        try:
            arr = non_max_suppression_by_distance(arr, scores, 25, det_thres)
        except:
            pass

    result_boxes = arr

    return result_boxes

def calculate_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = 512,
    slice_width: int = 512,
    nms_threshold: float = 0.4,
) -> list[list[int]]:
    # Slice images into patches with overlap area defined by nms_threshold
    overlap_height_ratio: float = nms_threshold
    overlap_width_ratio: float = nms_threshold

    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                box = [xmin, ymin, xmax, ymax]
                slice_bboxes.append(box)
            else:
                box = [x_min, y_min, x_max, y_max]
                slice_bboxes.append(box)
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes