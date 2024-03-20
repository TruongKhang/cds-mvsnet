import os, cv2, time
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from kornia.geometry.transform import warp_perspective
import kornia.augmentation as KA

from datasets.data_io import *
from .utils import ImgAug


class GeometricSequential:
    def __init__(self, *transforms, align_corners=True) -> None:
        self.transforms = transforms
        self.align_corners = align_corners

    def __call__(self, x, mode="bilinear"):
        b, c, h, w = x.shape
        M = torch.eye(3, device=x.device)[None].expand(b, 3, 3)
        for t in self.transforms:
            if np.random.rand() < t.p:
                M = M.matmul(
                    t.compute_transformation(x, t.generate_parameters((b, c, h, w)), flags=None)
                )
        return (
            warp_perspective(
                x, M, dsize=(h, w), mode=mode, align_corners=self.align_corners
            ),
            M,
        )

    def apply_transform(self, x, M, mode="bilinear"):
        b, c, h, w = x.shape
        return warp_perspective(
            x, M, dsize=(h, w), align_corners=self.align_corners, mode=mode
        )


class BlendedMVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(BlendedMVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        self.img_aug = ImgAug()
        self.geo_aug = None # GeometricSequential(KA.RandomAffine(degrees=90, p=0.3)) if mode == "train" else None
        if kwargs["random_image_scale"] is True:
            self.image_scale = np.random.choice([0.5417, 0.5, 0.4115])
            print("random image scale: ", self.image_scale)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

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
                    line = f.readline().rstrip()
                    src_views = [int(x) for x in line.split()[1::2]]
                    scores = [float(x) for x in line.split()[2::2]]
                    src_views = [src_views[i] for i, s in enumerate(scores) if (s > 0.1) and (src_views[i] != ref_view)]

                    # filter by no src view and fill to nviews
                    if len(src_views) > 2:
                        if len(src_views) < self.nviews:
                            print("{}< num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        src_views = src_views[:(self.nviews-1)]
                        metas.append((scan, ref_view, src_views, scan))

        # self.interval_scale = interval_scale_dict
        print("dataset ", self.mode, "metas: ", len(metas), "interval_scale: {}".format(self.interval_scale))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, H_mat=None):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        
        # intrinsics[0, 2] -= 32.0
        if not self.kwargs["high_res"]:
            intrinsics[1, 2] -= 32.0 
        # intrinsics[:2, :] /= 4.0
        if H_mat is not None:
            K = torch.from_numpy(intrinsics).float()
            intrinsics = H_mat[0] @ K
            intrinsics = intrinsics.numpy()

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= self.interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def prepare_img(self, img):
        h, w = img.shape[:2]
        target_h, target_w = 512, 768
        if self.kwargs["high_res"]:
            if self.mode == "train":
                target_h, target_w = 1536, 2048
            else:
                target_h, target_w = 576, 768
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

        H_mat = None
        if self.geo_aug is not None:
            tensor_img = torch.from_numpy(np_img).float()
            tensor_img, H_mat = self.geo_aug(tensor_img.permute(2, 0, 1).unsqueeze(0))
            np_img = tensor_img[0].permute(1, 2, 0).numpy()

        return np_img, H_mat

    def read_depth(self, filename, new_h, new_w, H_mat=None):
        # read pfm depth file
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth = self.prepare_img(depth)
        if H_mat is not None:
            depth = torch.from_numpy(depth)
            depth = self.geo_aug.apply_transform(depth[None, None], H_mat, mode='nearest')
            depth = depth.squeeze(0).squeeze(0).numpy()

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

        mask_ms = {}
        for stage, dm in depth_ms.items():
            mask_ms[stage] = (dm > 0).astype(np.float32)

        return depth_ms, mask_ms

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
            # src_views = src_views[:7]
            np.random.shuffle(src_views)
        view_ids = [ref_view] + src_views[:(self.nviews-1)]
        # view_ids = [ref_view] + src_views #[:self.nviews - 1]

        imgs = []
        depth_values = None
        proj_matrices = []
        mask, depth_ms = None, None
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))
            # mask_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))

            img, H_mat = self.read_img(img_filename, augmentation=True if self.mode == "train" else False)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, H_mat)

            if self.kwargs["high_res"]:
                max_h = int(1536 * self.image_scale) if self.mode == "train" else 576
                img, intrinsics, scaled_h, scaled_w = self.scale_img_cam(img, intrinsics, max_h=max_h)
            else:    
                img, intrinsics, scaled_h, scaled_w = self.scale_img_cam(img, intrinsics, max_h=512)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                # mask_read_ms = self.read_mask(mask_filename, scaled_h, scaled_w, H_mat)
                depth_ms, mask = self.read_depth(depth_filename, scaled_h, scaled_w, H_mat)

                # get depth values
                depth_max = depth_interval * (self.ndepths - 0.5) + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                # mask = mask_read_ms

            imgs.append(img)

        #all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4

        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 8

        if self.kwargs["num_stages"] == 4:
            proj_matrices_ms = {
                "stage1": stage0_pjmats,
                "stage2": stage1_pjmats,
                "stage3": stage2_pjmats,
                "stage4": proj_matrices
            }
        else:
            proj_matrices_ms = {
                "stage1": stage1_pjmats,
                "stage2": stage2_pjmats,
                "stage3": proj_matrices
            }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "mask": mask,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                "dataset_name": "blendedmvs",
                }
