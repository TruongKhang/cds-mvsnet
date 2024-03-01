import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from plyfile import PlyData, PlyElement
import torch

from utils import tocuda
from datasets.data_io import read_pfm
from fusion import utils, gipuma

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


class MVSRGBD(Dataset):
    def __init__(self, pair_folder, scan_folder, n_views=10):
        super(MVSRGBD, self).__init__()
        pair_file = os.path.join(pair_folder, "pair.txt")
        self.scan_folder = scan_folder
        self.pair_data = read_pair_file(pair_file)
        self.n_views = n_views

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, idx):
        id_ref, id_srcs = self.pair_data[idx]
        id_srcs = id_srcs[:self.n_views]

        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(self.scan_folder, 'cams/{:0>8}_cam.txt'.format(id_ref)))
        ref_cam = np.zeros((2, 4, 4), dtype=np.float32)
        ref_cam[0] = ref_extrinsics
        ref_cam[1, :3, :3] = ref_intrinsics
        ref_cam[1, 3, 3] = 1.0
        # load the reference image
        ref_img = read_img(os.path.join(self.scan_folder, 'images/{:0>8}.jpg'.format(id_ref)))
        ref_img = ref_img.transpose([2, 0, 1])
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(self.scan_folder, 'depth_est/{:0>8}.pfm'.format(id_ref)))[0]
        ref_depth_est = np.array(ref_depth_est, dtype=np.float32)
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(self.scan_folder, 'confidence/{:0>8}.pfm'.format(id_ref)))[0]
        confidence = np.array(confidence, dtype=np.float32).transpose([2, 0, 1])

        src_depths, src_confs, src_cams = [], [], []
        for ids in id_srcs:
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(self.scan_folder, 'cams/{:0>8}_cam.txt'.format(ids)))
            src_proj = np.zeros((2, 4, 4), dtype=np.float32)
            src_proj[0] = src_extrinsics
            src_proj[1, :3, :3] = src_intrinsics
            src_proj[1, 3, 3] = 1.0
            src_cams.append(src_proj)
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(self.scan_folder, 'depth_est/{:0>8}.pfm'.format(ids)))[0]
            src_depths.append(np.array(src_depth_est, dtype=np.float32))
            src_conf = read_pfm(os.path.join(self.scan_folder, 'confidence/{:0>8}.pfm'.format(ids)))[0]
            src_confs.append(np.array(src_conf, dtype=np.float32).transpose([2, 0, 1]))
        src_depths = np.expand_dims(np.stack(src_depths, axis=0), axis=1)
        src_confs = np.stack(src_confs, axis=0)
        src_cams = np.stack(src_cams, axis=0)
        return {"ref_depth": np.expand_dims(ref_depth_est, axis=0),
                "ref_cam": ref_cam,
                "ref_conf": confidence, #np.expand_dims(confidence, axis=0),
                "src_depths": src_depths,
                "src_cams": src_cams,
                "src_confs": src_confs,
                "ref_img": ref_img,
                "ref_id": id_ref}


def main(mvs_rgbd_dir, img_pair_dir, plyfilename, config, device=torch.device("cpu")):
    if config.filter_method in ["pcd", "dpcd"]:
        mvsrgbd_dataset = MVSRGBD(img_pair_dir, mvs_rgbd_dir, n_views=config.n_views)
        sampler = SequentialSampler(mvsrgbd_dataset)
        dataloader = DataLoader(mvsrgbd_dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=2,
                               pin_memory=True, drop_last=False)
        views = {}
        # prob_threshold = config.conf_thr
        prob_threshold = [float(p) for p in config.conf_thr.split(',')]
        for batch_idx, sample_np in enumerate(dataloader):
            sample = tocuda(sample_np)
            # for ids in range(sample["src_depths"].size(1)):
            #     src_prob_mask = utils.prob_filter(sample['src_confs'][:, ids, ...], prob_threshold)
            #     sample["src_depths"][:, ids, ...] *= src_prob_mask.float()

            prob_mask = utils.prob_filter(sample['ref_conf'], prob_threshold)

            if config.filter_method == "pcd":
                reproj_xyd, in_range = utils.get_reproj(
                    *[sample[attr] for attr in ['ref_depth', 'src_depths', 'ref_cam', 'src_cams']])
                vis_masks, vis_mask = utils.vis_filter(sample['ref_depth'], reproj_xyd, in_range, config.disp_thr, 0.003, config.nview_thr)

                ref_depth_ave = utils.ave_fusion(sample['ref_depth'], reproj_xyd, vis_masks)

                mask = utils.bin_op_reduce([prob_mask, vis_mask], torch.min)
                idx_img = utils.get_pixel_grids(*ref_depth_ave.size()[-2:]).unsqueeze(0)
                idx_cam = utils.idx_img2cam(idx_img, ref_depth_ave, sample['ref_cam'])
                points = utils.idx_cam2world(idx_cam, sample['ref_cam'])[..., :3, 0].permute(0, 3, 1, 2)
            else:
                ref_depth = sample['ref_depth']  # [n 1 h w ]
                reproj_xyd = utils.get_reproj_dynamic(*[sample[attr] for attr in ['ref_depth', 'src_depths', 'ref_cam', 'src_cams']])
                # reproj_xyd   nv 3 h w

                # 4 1300
                vis_masks, vis_mask = utils.vis_filter_dynamic(sample['ref_depth'], reproj_xyd, dist_base=4, rel_diff_base=1300)

                # mask reproj_depth
                reproj_depth = reproj_xyd[:, :, -1]  # [1 v h w]
                reproj_depth[~vis_mask.squeeze(2)] = 0  # [n v h w ]
                geo_mask_sums = vis_masks.sum(dim=1)  # 0~v
                geo_mask_sum = vis_mask.sum(dim=1)
                depth_est_averaged = (torch.sum(reproj_depth, dim=1, keepdim=True) + ref_depth) / (geo_mask_sum + 1)  # [1,1,h,w]
                num_src_views = sample['src_depths'].shape[1]
                dy_range = num_src_views + 1
                geo_mask = geo_mask_sum >= dy_range  # all zero
                for i in range(2, dy_range):
                    geo_mask = torch.logical_or(geo_mask, geo_mask_sums[:, i - 2] >= i)
            
                mask = utils.bin_op_reduce([prob_mask, geo_mask], torch.min)
                idx_img = utils.get_pixel_grids(*depth_est_averaged.size()[-2:]).unsqueeze(0)
                idx_cam = utils.idx_img2cam(idx_img, depth_est_averaged, sample['ref_cam'])
                points = utils.idx_cam2world(idx_cam, sample['ref_cam'])[..., :3, 0].permute(0, 3, 1, 2)

            points_np = points.cpu().data.numpy()
            mask_np = mask.cpu().data.numpy().astype(bool)
            #dir_vecs = dir_vecs.cpu().data.numpy()
            ref_img = sample_np['ref_img'].data.numpy()
            for i in range(points_np.shape[0]):
                print(np.sum(np.isnan(points_np[i])))
                p_f_list = [points_np[i, k][mask_np[i, 0]] for k in range(3)]
                p_f = np.stack(p_f_list, -1)
                c_f_list = [ref_img[i, k][mask_np[i, 0]] for k in range(3)]
                c_f = np.stack(c_f_list, -1) * 255
                #d_f_list = [dir_vecs[i, k][mask_np[i, 0]] for k in range(3)]
                #d_f = np.stack(d_f_list, -1)
                ref_id = str(sample_np['ref_id'][i].item())
                views[ref_id] = (p_f, c_f.astype(np.uint8))
                print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(mvs_rgbd_dir, int(ref_id), prob_mask[i].float().mean().item(), vis_mask[i].float().mean().item(), mask[i].float().mean().item()))

        print('Write combined PCD')
        p_all, c_all = [np.concatenate([v[k] for key, v in views.items()], axis=0) for k in range(2)]

        vertexs = np.array([tuple(v) for v in p_all], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_colors = np.array([tuple(v) for v in c_all], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
        for prop in vertexs.dtype.names:
            vertex_all[prop] = vertexs[prop]
        for prop in vertex_colors.dtype.names:
            vertex_all[prop] = vertex_colors[prop]

        el = PlyElement.describe(vertex_all, 'vertex')
        PlyData([el]).write(plyfilename)
        print("saving the final model to", plyfilename)
    elif config.filter_method == "gipuma":
        prob_threshold = config.conf_thr
        # prob_threshold = [float(p) for p in prob_threshold.split(',')]
        work_dir, scene = os.path.split(mvs_rgbd_dir)
        gipuma.gipuma_filter([scene], work_dir, prob_threshold, config.disp_thr, config.nview_thr, config.fusibile_exe_path)
