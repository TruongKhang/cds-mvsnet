from torch.utils.data import Dataset
import numpy as np
import os, cv2, time
from PIL import Image
import random
from datasets.data_io import *

s_h, s_w = 0, 0
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.max_h, self.max_w = kwargs["max_h"], kwargs["max_w"]
        self.fix_res = kwargs.get("fix_res", False)  #whether to fix the resolution of input image.
        self.fix_wh = False
        self.kwargs = kwargs

        assert self.mode == "test"
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
        self.trans_norm = {}

        scale_file = os.path.join(self.datapath, 'scale_dtu.txt')
        if not os.path.exists(scale_file):
            self.trans_norm = None
        else:
            with open(scale_file) as fp:
                for line in fp:
                    line = line.strip().split(':')
                    scan = line[0]
                    self.trans_norm[scan] = np.array(line[1].split(','), dtype=np.float32)

        self.generate_indices()

    def build_list(self):
        metas = {}
        scans = self.listfile

        interval_scale_dict = {}
        # scans
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                    # uncommet this to evaluate on tanks tand temples
                    """if ref_view < (self.nviews - 1):
                        left = 0
                    else:
                        left = ref_view - (self.nviews - 1)
                    if (left + self.nviews) > num_viewpoint:
                        left = num_viewpoint - self.nviews
                    f.readline() # ignore the given source views
                    src_views = [x for x in range(left, left+self.nviews) if x != ref_view]
                    src_views = src_views[::-1]"""

                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            print("{}< num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        src_views = src_views[:(self.nviews-1)]
                        if scan not in metas:
                            metas[scan] = [(ref_view, src_views)]
                        else:
                            metas[scan].append((ref_view, src_views))
                        # metas.append((scan, ref_view, src_views, scan))

        self.interval_scale = interval_scale_dict
        # print("dataset", self.mode, "metas:", len(metas), "interval_scale:{}".format(self.interval_scale))
        return metas

    def __len__(self):
        return len(self.generate_img_index)

    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)


    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = cv2.resize(np_img, (1152, 864), interpolation=cv2.INTER_NEAREST)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": np_img,
        }
        return np_img_ms

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

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
            # batch_crop_coords = [(np.random.choice(range_h), np.random.choice(range_w)) for i in range(self.batch_size)]
            while idx < len(self.spliter):
                for i in range(len(batch_ptrs)):
                    if id_ptrs[i] == 0:
                        self.list_begin.append(True)
                    else:
                        self.list_begin.append(False)
                    name, id_data = self.spliter[batch_ptrs[i]]
                    self.generate_img_index.append((name, id_data[id_ptrs[i]]))
                    # self.list_crop_coords.append(batch_crop_coords[i])
                    id_ptrs[i] += 1
                    if id_ptrs[i] >= len(id_data):
                        idx += 1
                        batch_ptrs[i] = idx
                        id_ptrs[i] = 0
                        # batch_crop_coords[i] = (np.random.choice(range_h), np.random.choice(range_w))
                    if idx >= len(self.spliter):
                        if i < len(batch_ptrs) - 1:
                            self.generate_img_index = self.generate_img_index[:-(i + 1)]
                            self.list_begin = self.list_begin[:-(i + 1)]
                            # self.list_crop_coords = self.list_crop_coords[:-(i + 1)]
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

    def __getitem__(self, idx):
        global s_h, s_w
        key, real_idx = self.generate_img_index[idx]
        meta = self.metas[key][real_idx] #self.metas[idx]
        ref_view, src_views = meta #scan, ref_view, src_views, scene_name = meta
        scan = scene_name = key
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views #[:self.nviews - 1]

        imgs = []
        depth_values = None
        proj_matrices = []
        input_depths = {"stage1": [], "stage2": [], "stage3": []}
        input_confs = {"stage1": [], "stage2": [], "stage3": []}
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images_post/{:0>8}.jpg'.format(scan, vid))
            if not os.path.exists(img_filename):
                img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            # mask_filename_hr = '/mnt/sdb/khang/tanksandtemples_short_range/intermediate/inputs/stage3/{}/mask/{:0>8}_final.png'.format(scan, vid) 
            mask_filename_hr = '/mnt/sdb/khang/dtu_dataset/train/Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid)

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=
                                                                                   self.interval_scale[scene_name])
            # scale input
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)

            if self.fix_res:
                # using the same standard height or width in entire scene.
                s_h, s_w = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True

            if i == 0:
                if not self.fix_wh:
                    # using the same standard height or width in each nviews.
                    s_h, s_w = img.shape[:2]

            # resize to standard height or width
            c_h, c_w = img.shape[:2]
            if (c_h != s_h) or (c_w != s_w):
                scale_h = 1.0 * s_h / c_h
                scale_w = 1.0 * s_w / c_w
                img = cv2.resize(img, (s_w, s_h))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h

            imgs.append(img)
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)

            mask_vid = self.read_mask_hr(mask_filename_hr)

            stage = "stage3"
            #in_depth_file = os.path.join(self.datapath, 'inputs/{}/{}/depth_est/{:0>8}.pfm'.format(stage, scan, vid))
            #in_depth = np.array(read_pfm(in_depth_file)[0], dtype=np.float32) * (mask_vid[stage] > 0.5).astype(np.float32)
            in_depth_file = os.path.join(self.datapath, 'inputs/{}/{}/depth_est/{:0>8}.png'.format(stage, scan, vid))
            in_depth = np.array(Image.open(in_depth_file), dtype=np.float32) / 10 * (mask_vid[stage] > 0.5).astype(np.float32)
            #in_conf_file = os.path.join(self.datapath, 'inputs/{}/{}/confidence/{:0>8}.pfm'.format(stage, scan, vid))
            #in_conf = np.array(read_pfm(in_conf_file)[0], dtype=np.float32) * (mask_vid[stage] > 0.5).astype(np.float32)
            in_conf_file = os.path.join(self.datapath, 'inputs/{}/{}/confidence/{:0>8}.png'.format(stage, scan, vid))
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
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                "is_begin": self.list_begin[idx],
                "prior_depths": input_depths,
                "prior_confs": input_confs}
