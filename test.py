import argparse, os, time, sys, gc, cv2
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from parse_config import ConfigParser
import datasets.data_loaders as module_data
import models.model as module_arch
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from gipuma import gipuma_filter
from utils import tocuda, print_args, generate_pointcloud, tensor2numpy
import fusion

from multiprocessing import Pool
from functools import partial
import signal

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
parser.add_argument('--config', default=None, type=str, help='config file path (default: None)')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', help='testing scene list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')

parser.add_argument('--resume', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--interval_scale', type=float, required=True, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=3, help='num of view')
parser.add_argument('--max_h', type=int, default=864, help='testing max h')
parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')
parser.add_argument('--depth_scale', type=float, default=1.0, help='depth scale')

parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')


parser.add_argument('--filter_method', type=str, default='normal', choices=["gipuma", "normal"], help="filter method")

# filter
parser.add_argument('--conf', type=str, default='0.9', help='prob confidence')
parser.add_argument('--thres_view', type=int, default=3, help='threshold of num view')
parser.add_argument('--thres_disp', type=float, default=1.0, help='threshold of disparity')

# filter by gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='./fusibile/fusibile')
parser.add_argument('--prob_threshold', type=str, default='0.8')
parser.add_argument('--disp_threshold', type=float, default='0.2')
parser.add_argument('--num_consistent', type=float, default='3')


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)
if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])

Interval_Scale = args.interval_scale
print("***********Interval_Scale**********\n", Interval_Scale)


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


def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


# def save_depth(testlist, config):
#     for scene in testlist:
#         save_scene_depth([scene], config)


# run CasMVS model to save depth maps and confidence maps
def save_depth(testlist, config):
    # dataset, dataloader

    init_kwags = {
        "data_path": args.testpath,
        "data_list": testlist,
        "mode": "test",
        "num_srcs": args.num_view,
        "num_depths": args.numdepth,
        "interval_scale": Interval_Scale,
        "shuffle": False,
        "seq_size": 49,
        "batch_size": 1,
        "fix_res": args.fix_res,
        "max_h": args.max_h,
        "max_w": args.max_w
    }
    test_data_loader = getattr(module_data, config['data_loader'][0]['type'])(**init_kwags)
    # model
    # build models architecture
    model = config.init_obj('arch', module_arch)

    print('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(str(config.resume))
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, val in state_dict.items():
        new_state_dict[key.replace('module.', '')] = val
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(new_state_dict)

    # prepare models for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_data_loader):
            start_time = time.time()
            sample_cuda = tocuda(sample)
            is_begin = sample['is_begin'].type(torch.uint8)
            num_stage = len(config["arch"]["args"]["ndepths"])

            imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]
            outputs = model(imgs, cam_params, sample_cuda["depth_values"])
            torch.cuda.synchronize()
            outputs["ps_map"] = model.feature.extract_ps_map()

            end_time = time.time()
            outputs = tensor2numpy(outputs)
            del sample_cuda
            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            imgs = sample["imgs"].numpy()
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(test_data_loader), end_time - start_time,
                                                      imgs[0].shape))

            # save depth maps and confidence maps
            for filename, cam, img, depth_est, conf_stage1, conf_stage2, conf_stage3, ps_map in zip(filenames, cams, imgs, outputs["depth"], outputs["stage1"]["photometric_confidence"], outputs["stage2"]["photometric_confidence"],
                                                                             outputs["photometric_confidence"], outputs["ps_map"]):
                img = img[0]  # ref view
                cam = cam[0]  # ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                ps_filename = os.path.join(args.outdir, filename.format('ps_maps', '.png'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(ps_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                # depth_est = cv2.resize(depth_est, (args.max_w, args.max_h))
                # depth_est = Image.fromarray((depth_est*100).astype(np.uint16))
                # depth_est.save(depth_filename)
                # np.save(depth_filename, depth_est)
                # save confidence maps
                h, w = conf_stage3.shape[0], conf_stage3.shape[1]
                conf_stage1 = cv2.resize(conf_stage1, (w, h), interpolation=cv2.INTER_NEAREST)
                conf_stage2 = cv2.resize(conf_stage2, (w, h), interpolation=cv2.INTER_NEAREST)
                photometric_confidence = np.stack([conf_stage1, conf_stage2, conf_stage3]).transpose([1,2,0])
                save_pfm(confidence_filename, photometric_confidence)
                # save cams, img
                write_cam(cam_filename, cam)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

                ps_map = Image.fromarray((ps_map * 100).astype(np.uint16))
                ps_map.save(ps_filename)
                # vis
                # print(photometric_confidence.mean(), photometric_confidence.min(), photometric_confidence.max())
                # import matplotlib.pyplot as plt
                # plt.subplot(1, 3, 1)
                # plt.imshow(img)
                # plt.subplot(1, 3, 2)
                # plt.imshow((depth_est - depth_est.min())/(depth_est.max() - depth_est.min()))
                # plt.subplot(1, 3, 3)
                # plt.imshow(photometric_confidence)
                # plt.show()

                """if num_stage == 1:
                    downsample_img = cv2.resize(img, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)))
                elif num_stage == 2:
                    downsample_img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
                elif num_stage == 3:
                    downsample_img = img

                if batch_idx % args.save_freq == 0:
                    generate_pointcloud(downsample_img, depth_est, ply_filename, cam[1, :3, :3])"""

    torch.cuda.empty_cache()
    gc.collect()


class TTDataset(Dataset):
    def __init__(self, pair_folder, scan_folder, n_src_views=10):
        super(TTDataset, self).__init__()
        pair_file = os.path.join(pair_folder, "pair.txt")
        self.scan_folder = scan_folder
        self.pair_data = read_pair_file(pair_file)
        self.n_src_views = n_src_views

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, idx):
        id_ref, id_srcs = self.pair_data[idx]
        id_srcs = id_srcs[:self.n_src_views]

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

        src_depths, src_cams = [], []
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
        src_depths = np.expand_dims(np.stack(src_depths, axis=0), axis=1)
        src_cams = np.stack(src_cams, axis=0)
        return {"ref_depth": np.expand_dims(ref_depth_est, axis=0),
                "ref_cam": ref_cam,
                "ref_conf": confidence, #np.expand_dims(confidence, axis=0),
                "src_depths": src_depths,
                "src_cams": src_cams,
                "ref_img": ref_img,
                "ref_id": id_ref}


def filter_depth(pair_folder, scan_folder, out_folder, plyfilename):
    tt_dataset = TTDataset(pair_folder, scan_folder, n_src_views=10)
    sampler = SequentialSampler(tt_dataset)
    tt_dataloader = DataLoader(tt_dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=2,
                               pin_memory=True, drop_last=False)
    views = {}
    prob_threshold = args.conf
    prob_threshold = [float(p) for p in prob_threshold.split(',')]
    for batch_idx, sample_np in enumerate(tt_dataloader):
        sample = tocuda(sample_np)
        #if sample_np.get('skip') is not None and np.any(sample_np['skip']): continue
        #sample = {attr: torch.from_numpy(sample_np[attr]).float().cuda() for attr in sample_np if
        #          attr not in ['skip', 'id']}

        prob_mask = fusion.prob_filter(sample['ref_conf'], prob_threshold)

        reproj_xyd, in_range = fusion.get_reproj(
            *[sample[attr] for attr in ['ref_depth', 'src_depths', 'ref_cam', 'src_cams']])
        vis_masks, vis_mask = fusion.vis_filter(sample['ref_depth'], reproj_xyd, in_range, args.thres_disp, 0.01, args.thres_view)

        ref_depth_ave = fusion.ave_fusion(sample['ref_depth'], reproj_xyd, vis_masks)

        mask = fusion.bin_op_reduce([prob_mask, vis_mask], torch.min)

        idx_img = fusion.get_pixel_grids(*ref_depth_ave.size()[-2:]).unsqueeze(0)
        idx_cam = fusion.idx_img2cam(idx_img, ref_depth_ave, sample['ref_cam'])
        points = fusion.idx_cam2world(idx_cam, sample['ref_cam'])[..., :3, 0].permute(0, 3, 1, 2)
        points_np = points.cpu().data.numpy()
        mask_np = mask.cpu().data.numpy().astype(np.bool)
        ref_img = sample_np['ref_img'].data.numpy()
        for i in range(points_np.shape[0]):
            print(np.sum(np.isnan(points_np[i])))
            p_f_list = [points_np[i, k][mask_np[i, 0]] for k in range(3)]
            p_f = np.stack(p_f_list, -1)
            c_f_list = [ref_img[i, k][mask_np[i, 0]] for k in range(3)]
            c_f = np.stack(c_f_list, -1) * 255
            ref_id = str(sample_np['ref_id'][i].item())
            views[ref_id] = (p_f, c_f.astype(np.uint8))
            print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, int(ref_id), prob_mask[i].float().mean().item(), vis_mask[i].float().mean().item(), mask[i].float().mean().item()))

    print('Write combined PCD')
    p_all, c_all = [np.concatenate([v[k] for key, v in views.items()], axis=0) for k in range(2)]

    vertexs, vertex_colors = [], []
    for i in range(len(p_all)):
        if np.sum(np.isnan(p_all[i])) == 0:
            vertexs.append(tuple(p_all[i]))
            vertex_colors.append(tuple(c_all[i]))
    vertexs = np.array(vertexs, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array(vertex_colors, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    #vertexs = np.array([tuple(v) for v in p_all], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    #vertex_colors = np.array([tuple(v) for v in c_all], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pcd_filter_worker(scan):
    #if args.testlist != "all":
    #    scan_id = int(scan[4:])
    #    save_name = 'mvsnet{:0>3}_l3.ply'.format(scan_id)
    #else:
    save_name = '{}.ply'.format(scan)
    pair_folder = os.path.join(args.testpath, scan)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)
    filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))


def pcd_filter(testlist, number_worker):
    for scan in testlist:
        pcd_filter_worker(scan)

    """partial_func = partial(pcd_filter_worker)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()"""


if __name__ == '__main__':
    config = ConfigParser.from_args(parser)

    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        #for tanks & temples or eth3d or colmap
        testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
            if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    # step1. save all the depth maps and the masks in outputs directory
    save_depth(testlist, config)

    # step2. filter saved depth maps with photometric confidence maps and geometric constraints

    # if args.filter_method != "gipuma":
    # #     #support multi-processing, the default number of worker is 4
    #     pcd_filter(testlist, args.num_worker)
    # else:
    #     prob_threshold = args.prob_threshold
    #     prob_threshold = [float(p) for p in prob_threshold.split(',')]
    #     gipuma_filter(testlist, args.outdir, prob_threshold, args.disp_threshold, args.num_consistent,
    #                   args.fusibile_exe_path)
