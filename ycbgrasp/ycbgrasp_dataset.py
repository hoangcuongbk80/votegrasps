""" Dataset for grasping on YCB objects (with support of vote supervision).

A oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio # to load .mat files for depth points
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import ycbgrasp_utils
from model_util_ycbgrasp import ycbgraspDatasetConfig

DC = ycbgraspDatasetConfig() # dataset specific config
MAX_NUM_GRASP = 64 # maximum number of grasps allowed per scene
MEAN_COLOR_RGB = np.array([0.5,0.5,0.5]) # sunrgbd color is in 0~1

class ycbgraspVotesDataset(Dataset):
    def __init__(self, split_set='train', num_points=20000,
        use_color=False, use_height=False, augment=False, scan_idx_list=None):

        assert(num_points<=50000)
        self.data_path = os.path.join(ROOT_DIR, 'ycbgrasp/data/%s'%(split_set))

        self.scan_names = sorted(list(set([os.path.basename(x)[0:6] \
            for x in os.listdir(self.data_path)])))
        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_GRASP,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_GRASP,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_GRASP,)
            size_classe_label: (MAX_NUM_GRASP,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_GRASP,3)
            sem_cls_label: (MAX_NUM_GRASP,) semantic class index
            box_label_mask: (MAX_NUM_GRASP) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_grasps: unused
        """
        scan_name = self.scan_names[idx]
        point_cloud = np.load(os.path.join(self.data_path, scan_name)+'_pc.npz')['pc'] # Nx6
        grasps = np.load(os.path.join(self.data_path, scan_name)+'_grasp.npy') # K,8
        point_votes = np.load(os.path.join(self.data_path, scan_name)+'_votes.npz')['point_votes'] # Nx10

        if not self.use_color:
            point_cloud = point_cloud[:,0:3]
        else:
            point_cloud = point_cloud[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)

        # ------------------------------- LABELS ------------------------------
        grasp_centers = np.zeros((MAX_NUM_GRASP, 3))
        grasp_sizes = np.zeros((MAX_NUM_GRASP, 3))
        angle_classes = np.zeros((MAX_NUM_GRASP,))
        angle_residuals = np.zeros((MAX_NUM_GRASP,))
        size_classes = np.zeros((MAX_NUM_GRASP,))
        size_residuals = np.zeros((MAX_NUM_GRASP, 3))
        label_mask = np.zeros((MAX_NUM_GRASP))
        label_mask[0:grasps.shape[0]] = 1

        for i in range(grasps.shape[0]):
            grasp = grasps[i]
            semantic_class = grasp[7]
            grasp_center = grasp[0:3]
            angle_class, angle_residual = DC.angle2class(grasp[6]) 
            grasp_size = grasp[3:6]*2
            size_class, size_residual = DC.size2class(grasp_sizes, DC.class2type[semantic_class])
            grasp_centers[i,:] = grasp_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            #size_residuals[i] = size_residual
            grasp_sizes[i,:] = grasp_size

        target_grasps_mask = label_mask 
        target_grasps = np.zeros((MAX_NUM_GRASP, 6))
        for i in range(grasps.shape[0]):
            grasp = grasps[i]
            target_grasp = grasp[0:6]
            target_grasps[i,:] = target_grasp

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)

        end_points['width_label'] = 0.06
        end_points['quality_label'] = 0.06

        ret_dict['center_label'] = target_grasps.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_grasps_semcls = np.zeros((MAX_NUM_GRASP))
        target_grasps_semcls[0:grasps.shape[0]] = grasps[:,-1] # from 0 to 9
        ret_dict['sem_cls_label'] = target_grasps_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_grasps_mask.astype(np.float32)
        
        return ret_dict

def viz_votes(pc, point_votes, point_votes_mask):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]
    pc_obj_voted2 = pc_obj + point_votes[inds,3:6]
    pc_obj_voted3 = pc_obj + point_votes[inds,6:9]
    pc_util.write_ply(pc_obj, 'pc_obj.ply')
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1.ply')
    pc_util.write_ply(pc_obj_voted2, 'pc_obj_voted2.ply')
    pc_util.write_ply(pc_obj_voted3, 'pc_obj_voted3.ply')

def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_GRASP
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = DC.class2angle(angle_classes[i], angle_residuals[i])
        box_size = DC.class2size(size_classes[i], size_residuals[i])
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs.ply')
    pc_util.write_ply(label[mask==1,:], 'gt_centroids.ply')

def get_sem_cls_statistics():
    """ Compute number of objects for each semantic class """
    d = SunrgbdDetectionVotesDataset(use_height=True, use_color=True, use_v1=True, augment=True)
    sem_cls_cnt = {}
    for i in range(len(d)):
        if i%10==0: print(i)
        sample = d[i]
        pc = sample['point_clouds']
        sem_cls = sample['sem_cls_label']
        mask = sample['box_label_mask']
        for j in sem_cls:
            if mask[j] == 0: continue
            if sem_cls[j] not in sem_cls_cnt:
                sem_cls_cnt[sem_cls[j]] = 0
            sem_cls_cnt[sem_cls[j]] += 1
    print(sem_cls_cnt)

if __name__=='__main__':
    d = ycbgraspVotesDataset(use_height=True, use_color=True, augment=True)
    sample = d[200]
    print(sample['vote_label'].shape, sample['vote_label_mask'].shape)
    pc_util.write_ply(sample['point_clouds'], 'pc.ply')
    viz_votes(sample['point_clouds'], sample['vote_label'], sample['vote_label_mask'])
    viz_obb(sample['point_clouds'], sample['center_label'], sample['box_label_mask'],
        sample['heading_class_label'], sample['heading_residual_label'],
        sample['size_class_label'], sample['size_residual_label'])
