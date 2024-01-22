from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import read_pfm
import random
from .utils import ImgAug

np.random.seed(123)
random.seed(123)


# the DTU dataset preprocessed by Yao Yao (only for training)
class DTUMVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(DTUMVSDataset, self).__init__()
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
        self.img_aug = ImgAug()

    def build_list(self):
        metas = []
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
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # src_views = src_views[:(self.nviews-1)]

                    # light conditions 0-6
                    if self.mode == 'train':
                        lights = np.random.choice(np.arange(7), 4, replace=False)
                    else:
                        lights = np.arange(7)
                    for light_idx in lights:
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        # return len(self.generate_img_index)
        return len(self.metas)

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

    def read_img(self, filename, augmentation=False):
        # img = Image.open(filename)
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if augmentation:
            img = self.img_aug(img)
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

    def read_mask_hr(self, filename, new_h, new_w):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)
        np_img = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        h, w = np_img.shape
        if self.kwargs["num_stages"] == 4:
            np_img_ms = {
                "stage1": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
                "stage2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
                "stage3": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
                "stage4": np_img,
            }
        else:
            np_img_ms = {
                "stage1": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
                "stage2": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
                "stage3": np_img,
            }
        return np_img_ms

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_depth_hr(self, filename, new_h, new_w):
        # read pfm depth file
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.prepare_img(depth_hr)
        depth_lr = cv2.resize(depth_lr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        h, w = depth_lr.shape
        if self.kwargs["num_stages"] == 4:
            depth_lr_ms = {
                "stage1": cv2.resize(depth_lr, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
                "stage2": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
                "stage3": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
                "stage4": depth_lr,
            }
        else:
            depth_lr_ms = {
                "stage1": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
                "stage2": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
                "stage3": depth_lr,
            }
        return depth_lr_ms

    def scale_img_cam(self, img, intrinsics, max_h, base=64):
        h, w = img.shape[:2]
        if h > max_h:
            max_w = w * max_h / h
            new_w, new_h = max_w // base * base, max_h // base * base
        else:
            new_w, new_h = w // base * base, h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h

        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics, int(new_h), int(new_w)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        if self.mode == 'train':
            np.random.shuffle(src_views)
        view_ids = [ref_view] + src_views[:(self.nviews-1)]
        # view_ids = [ref_view] + src_views

        imgs = []
        mask = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))

            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            img = self.read_img(img_filename, augmentation=True if self.mode == "train" else False)

            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            img, intrinsics, scaled_h, scaled_w = self.scale_img_cam(img, intrinsics, max_h=512)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                mask_read_ms = self.read_mask_hr(mask_filename_hr, scaled_h, scaled_w)
                depth_ms = self.read_depth_hr(depth_filename_hr, scaled_h, scaled_w)

                #get depth values
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                mask = mask_read_ms

            imgs.append(img)

        #all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        #ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
        if self.kwargs["num_stages"] == 4:
            proj_matrices_ms = {
                "stage1": stage0_pjmats,
                "stage2": proj_matrices,
                "stage3": stage2_pjmats,
                "stage4": stage3_pjmats
            }
        else:
            proj_matrices_ms = {
                "stage1": proj_matrices,
                "stage2": stage2_pjmats,
                "stage3": stage3_pjmats
            }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "mask": mask,
                "dataset_name": "dtu",
                }
