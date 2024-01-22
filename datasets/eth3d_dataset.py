from torch.utils.data import Dataset
import os, cv2, time
from PIL import Image, ImageOps
from datasets.data_io import *
from .utils import ImgAug


class ETH3DDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(ETH3DDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        self.img_aug = ImgAug()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]
        # scans = self.listfile

        interval_scale_dict = {}
        # scans
        for scan in scans:
            pair_file = "{}/cams/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    src_views = [v for v in src_views if v != ref_view]

                    # filter by no src view and fill to nviews
                    if len(src_views) > 2:
                        if len(src_views) < self.nviews:
                            print("{}< num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        # src_views = src_views[:(self.nviews-1)]
                        metas.append((scan, ref_view, src_views, scan))

        # self.interval_scale = interval_scale_dict
        print("dataset ", self.mode, "metas: ", len(metas), "interval_scale: {}".format(self.interval_scale))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[0, 2] -= 24.0
        # intrinsics[1, 2] -= 32.0
        intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            if len(lines[11].split()) >= 4:
                depth_max = float(lines[11].split()[3])
            else:
                depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= self.interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def prepare_img(self, img):
        h, w = img.shape[:2]
        target_h, target_w = 672, 960
        start_h, start_w = (h - target_h)//2, (w - target_w)//2
        img_crop = img[start_h: start_h + target_h, start_w: start_w + target_w]
        return img_crop

    def read_img(self, filename, augmentation=False):
        # img = Image.open(filename)
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if augmentation:
            img = self.img_aug(img)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        np_img = self.prepare_img(np_img)

        return np_img

    def read_depth(self, filename, new_h, new_w):
        # read pfm depth file
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth = self.prepare_img(depth)
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        h, w = depth.shape
        
        if self.kwargs["num_stages"] == 4:
            depth_ms = {
                "stage1": cv2.resize(depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
                "stage2": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
                "stage3": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
                "stage4": depth,
            }
        else:
            depth_ms = {
                "stage1": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
                "stage2": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
                "stage3": depth,
            }

        return depth_ms

    def read_mask(self, filename, new_h, new_w):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        np_img = (depth > 0).astype(np.float32)
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
        # global s_h, s_w
        # key, real_idx = self.generate_img_index[idx]
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        # use only the reference view and first nviews-1 source views
        if self.mode == 'train':
            src_views = src_views[:7]
            np.random.shuffle(src_views)
        view_ids = [ref_view] + src_views[:(self.nviews-1)]
        # view_ids = [ref_view] + src_views #[:self.nviews - 1]

        imgs = []
        depth_values = None
        proj_matrices = []
        mask, depth_ms = None, None
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))
            mask_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))

            img = self.read_img(img_filename, augmentation=True if (i==0) and (self.mode == "train") else False)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            img, intrinsics, scaled_h, scaled_w = self.scale_img_cam(img, intrinsics, max_h=448)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                mask_read_ms = self.read_mask(mask_filename, scaled_h, scaled_w)
                depth_ms = self.read_depth(depth_filename, scaled_h, scaled_w)

                # get depth values
                depth_max = depth_interval * (self.ndepths - 0.5) + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                mask = mask_read_ms

            imgs.append(img)

        #all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
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
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
