from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import read_pfm
import random

np.random.seed(123)
random.seed(123)


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        print("mvsdataset kwargs", self.kwargs)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

        self.generate_img_index = []
        self.list_begin = []
        self.spliter = []
        total_imgs = 0
        keys = sorted(list(self.metas.keys()))
        for name in keys:
            num_imgs = len(self.metas[name])
            total_imgs += num_imgs
            # print(name, num_imgs, seq_size)
            if (self.mode == 'train') and (self.kwargs['seq_size'] is not None):
                indices = np.arange(num_imgs)
                for ptr in range(0, num_imgs, self.kwargs['seq_size']):
                    self.spliter.append((name, indices[ptr:(ptr + self.kwargs['seq_size'])]))
            else:
                self.spliter.append((name, np.arange(num_imgs)))
        print("dataset", self.mode, "metas:", total_imgs)

        self.generate_indices()

    def build_list(self):
        metas = {}
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    if ref_view < (self.nviews - 1) // 2:
                        left = 0
                    else:
                        left = ref_view - (self.nviews - 1) // 2
                    if (left + self.nviews) > num_viewpoint:
                        left = num_viewpoint - self.nviews
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    src_views = src_views[:(self.nviews-1)]

                    # f.readline() # ignore the given source views
                    # src_views = [x for x in range(left, left+self.nviews) if x != ref_view]
                    # light conditions 0-6
                    # for light_idx in range(7):
                    #     metas.append((scan, light_idx, ref_view, src_views))
                    for light_idx in range(7):
                        key = '%s_%s' % (scan, light_idx)
                        if key not in metas:
                            metas[key] = [(ref_view, src_views)]
                        else:
                            metas[key].append((ref_view, src_views))
        # print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.generate_img_index)
        # return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def prepare_img(self, hr_img):
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128

        #downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        #crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

        # #downsample
        # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

        return hr_img_crop

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": np_img,
        }
        return np_img_ms

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_depth_hr(self, filename):
        # read pfm depth file
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_lr,
        }
        return depth_lr_ms

    def generate_indices(self):
        self.generate_img_index = []
        self.list_begin = []
        batch_size = self.kwargs['batch_size']

        if self.kwargs['shuffle']:
            random.shuffle(self.spliter)

        if self.mode == 'train':
            idx = batch_size - 1
            batch_ptrs = list(range(batch_size))
            id_ptrs = np.zeros(batch_size, dtype=np.uint8)
            while idx < len(self.spliter):
                for i in range(len(batch_ptrs)):
                    if id_ptrs[i] == 0:
                        self.list_begin.append(True)
                    else:
                        self.list_begin.append(False)
                    name, id_data = self.spliter[batch_ptrs[i]]
                    self.generate_img_index.append((name, id_data[id_ptrs[i]]))
                    id_ptrs[i] += 1
                    if id_ptrs[i] >= len(id_data):
                        idx += 1
                        batch_ptrs[i] = idx
                        id_ptrs[i] = 0
                    if idx >= len(self.spliter):
                        if i < len(batch_ptrs) - 1:
                            self.generate_img_index = self.generate_img_index[:-(i + 1)]
                            self.list_begin = self.list_begin[:-(i + 1)]
                        break
        else:
            for ptr in range(len(self.spliter)):
                name, indices = self.spliter[ptr]
                for i, idx in enumerate(indices):
                    self.generate_img_index.append((name, idx))
                    if i == 0:
                        self.list_begin.append(True)
                    else:
                        self.list_begin.append(False)

        # print("Number samples of %s dataset: " % self.mode, len(self.generate_img_index))

    def __getitem__(self, idx):
        # meta = self.metas[idx]
        # scan, light_idx, ref_view, src_views = meta
        key, real_idx = self.generate_img_index[idx]
        scan, light_idx = key.split('_')[0], int(key.split('_')[1])
        ref_view, src_views = self.metas[key][real_idx]
        # use only the reference view and first nviews-1 source views

        view_ids = [ref_view] + src_views

        imgs = []
        mask = None
        depth_values = None
        proj_matrices = []
        input_depths = {"stage1": [], "stage2": [], "stage3": []}
        input_confs = {"stage1": [], "stage2": [], "stage3": []}

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))

            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            img = self.read_img(img_filename)

            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                mask_read_ms = self.read_mask_hr(mask_filename_hr)
                depth_ms = self.read_depth_hr(depth_filename_hr)

                #get depth values
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                mask = mask_read_ms

            imgs.append(img)

            mask_vid = self.read_mask_hr(mask_filename_hr)

            stage = "stage3"
            in_depth_file = os.path.join(self.datapath, 'inputs/{}/{}/depth_est/{:0>3}_{}.png'.format(stage, scan, vid, light_idx))
            in_depth = np.array(Image.open(in_depth_file), dtype=np.float32) / 10 * (mask_vid[stage] > 0.5).astype(np.float32)
            in_conf_file = os.path.join(self.datapath, 'inputs/{}/{}/confidence/{:0>3}_{}.png'.format(stage, scan, vid, light_idx))
            in_conf = np.array(Image.open(in_conf_file), dtype=np.float32) / 255 * (mask_vid[stage] > 0.5).astype(np.float32)
            height, width = in_depth.shape
            input_depths["stage1"].append(cv2.resize(in_depth, (width//4, height//4), interpolation=cv2.INTER_NEAREST))
            input_depths["stage2"].append(cv2.resize(in_depth, (width//2, height//2), interpolation=cv2.INTER_NEAREST))
            input_depths["stage3"].append(in_depth)
            input_confs["stage1"].append(cv2.resize(in_conf, (width//4, height//4), interpolation=cv2.INTER_NEAREST))
            input_confs["stage2"].append(cv2.resize(in_conf, (width//2, height//2), interpolation=cv2.INTER_NEAREST))
            input_confs["stage3"].append(in_conf)

        #all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        #ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }
        for stage in input_depths.keys():
            input_depths[stage] = np.expand_dims(np.stack(input_depths[stage]), axis=1)
            input_confs[stage] = np.expand_dims(np.stack(input_confs[stage]), axis=1)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "mask": mask,
                "is_begin": self.list_begin[idx],
                "prior_depths": input_depths,
                "prior_confs": input_confs}
