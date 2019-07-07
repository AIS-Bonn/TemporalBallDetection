import numpy as np
import pdb
from arguments import opt
import torch
import cv2
from scipy import ndimage

def predict_box(peaks_predicted, box_actual):
    '''
    get predicted box
    '''
    box_predicted = []
    for box, peak in zip(box_actual.tolist(), peaks_predicted):
        radius = min((box[2]-box[0])/2, (box[3]-box[1])/2)
        xmin, ymin = peak[1] - radius, peak[0] - radius
        xmax, ymax = peak[1] + radius, peak[0] + radius
        box_predicted.append(np.array([xmin, ymin, xmax, ymax]))

    return np.asarray(box_predicted)

def within_radius(x, y, x_p, y_p, radius):
    return (abs(y-y_p)<=radius and abs(x-x_p)<=radius)


def tp_fp_tn_fn(peaks_batch, peaks_value, centers, radius=5):
    '''
    '''
    TP, FP, FN, TN = 0, 0, 0, 0
    for peak, peak_value, center in zip(peaks_batch, peaks_value, centers):
        y, x = center[0], center[1]
        y_p, x_p = peak[0], peak[1]
        if (y==0 and x==0) and (y_p==0 and x_p==0):
            TN += 1
        elif (y==0 and x==0) and (y_p!=0 or x_p!=0) and within_radius(x, y, x_p, y_p, radius):
            FP += 1
        elif within_radius(x, y, x_p, y_p, radius):
            TP += 1
        else:
            FN += 1
    return TP, FP, FN, TN

def get_closest_peak(x, y, peaks):
    ball_peak = peaks[0]
    dist_old = 1000
    for peak in peaks:
        y_p, x_p = peak[0], peak[1]
        dist_new = abs(y-y_p) + abs(x-x_p)
        if dist_new<=dist_old:
            ball_peak = (y_p, x_p)
    return ball_peak

def performance_metric(TP, FP, FN, TN):

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    if FP==0 and TP==0:
        FDR = 1.0
    else:
        FDR = FP/(FP + TP)
    if FN==0 and TP==0:
        RC = 0.0
    else:
        RC = TP/(TP + FN)
    return FDR, RC, accuracy


def peak_detection(map_, threshold):
    '''
    peak detection on predicted score
    '''
    peak =  np.unravel_index(np.argmax(map_, axis=None), map_.shape)
    while map_[peak]>=threshold:
        map_[peak] = 0
        peak = np.unravel_index(np.argmax(map_, axis=None), map_.shape)
    return peak

def load_model(model_name, key='state_dict_model', min_radius='min_radius', threshold='threshold'):
    '''
    get checkpoint of trained model and threshold
    '''
    model_dir = opt.model_root + model_name
    checkpoint = torch.load(model_dir)
    checkpoint, min_radius, threshold = checkpoint[key], checkpoint[min_radius], checkpoint[threshold]

    return checkpoint, min_radius, threshold


def post_processing(maps, threshold):

    processed_maps, predicted_centers, maps_area = [], [], []
    for map_ in maps:
        binary_map = (map_>0.1).astype(np.uint8)
        contours = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        img = np.zeros(map_.shape, np.uint8)
        if len(contours)>0:
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            area, biggest_contour = max(contour_sizes, key=lambda x: x[0])
            cv2.drawContours(img, [biggest_contour], -1, (1), cv2.FILLED)
            apply_map = map_ * img
            processed_maps.append(apply_map)
            #cX, cY = peak_detection(apply_map, threshold)
            cX, cY = ndimage.measurements.center_of_mass(apply_map)
            #M = cv2.moments(apply_map)
            #cX = int(M['m10'] / M['m00'])
            #cY = int(M['m01'] / M['m00'])
            predicted_centers.append((cX, cY))
            maps_area.append(area)
        else:
            apply_map = map_ * img
            processed_maps.append(apply_map)
            maps_area.append(0)
            predicted_centers.append((-1, -1))
    return processed_maps, predicted_centers, maps_area

def tp_fp_tn_fn_alt(actual_centers, predicted_centers, maps_area, min_radius):
    min_radius=5

    minm_area = min_radius**2
    TP, FP, TN, FN = 0, 0, 0, 0
    for area, (a_x, a_y), (p_x, p_y) in zip(maps_area, actual_centers, predicted_centers):
        if a_x==-1 and a_y==-1 and (area<minm_area or (p_x==-1 and p_y==-1)):
            TN += 1
        elif (a_x>=0 and a_y>=0) and area<minm_area:
            FN += 1
        elif (a_x>=0 or a_y>=0) and area>=minm_area and within_radius(a_x, a_y, p_x, p_y, min_radius):
            TP += 1
        else:
            FP +=1
    return TP, FP, TN, FN