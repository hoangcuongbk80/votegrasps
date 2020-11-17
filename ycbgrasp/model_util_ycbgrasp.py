import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class ycbgraspDatasetConfig(object):
    def __init__(self):
        self.num_class = 10
        self.num_heading_bin = 12
        self.num_size_cluster = 10

        self.type2class={'007_tuna_fish_can':0, '008_pudding_box':1, '011_banana':2, '024_bowl':3, '025_mug':4,
                        '044_flat_screwdriver':5, '051_large_clamp':6, '055_baseball':7, '061_foam_brick':8, '065-h_cups':9}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.type2onehotclass={'007_tuna_fish_can':0, '008_pudding_box':1, '011_banana':2, '024_bowl':3, '025_mug':4,
                            '044_flat_screwdriver':5, '051_large_clamp':6, '055_baseball':7, '061_foam_brick':8, '065-h_cups':9}
        self.type_mean_size = {'007_tuna_fish_can': np.array([0.0427889, 0.0427731, 0.0170824]),
                            '008_pudding_box': np.array([0.0601333, 0.0489079, 0.0191283]),
                            '011_banana': np.array([0.0985726, 0.0371073, 0.019]),
                            '024_bowl': np.array([0.0806955, 0.0802303, 0.0274737]),
                            '025_mug': np.array([0.0599306, 0.0552335, 0.0483711]),
                            '044_flat_screwdriver': np.array([0.108077, 0.0172692, 0.0167839]),
                            '051_large_clamp': np.array([0.0833631, 0.0607794, 0.0181959]),
                            '055_baseball': np.array([0.036739, 0.0362683, 0.0361896]),
                            '061_foam_brick': np.array([0.0396758, 0.0324347, 0.0310957]),
                            '065-h_cups': np.array([0.0458534, 0.0455691, 0.0385724])}

        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i,:] = self.type_mean_size[self.class2type[i]]

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual
    
    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb

    def param2grasp(self, center, heading_class, heading_residual, size_class):
        heading_angle = self.class2angle(heading_class, heading_residual) * 180/np.pi
        object_name = self.class2type[int(size_class)]
        grasp = []
        grasp.append(object_name)
        grasp.append(center[0])
        grasp.append(center[1])
        grasp.append(center[2])
        grasp.append(180.0)
        grasp.append(0.0)
        grasp.append(heading_angle)
        grasp.append(0.922)
        grasp.append(0.13)
        return grasp
