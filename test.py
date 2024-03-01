import argparse, os, time, sys, gc, cv2
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader

from datasets.general_eval import MVSDataset
from models import CDSMVSNet
from datasets.data_io import save_pfm
from utils import tocuda, print_args, tensor2numpy, read_json
from fusion import depth_fusion

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
parser.add_argument('--config_path', default="configs/config_blended.json", type=str, help='config file path (default: None)')

parser.add_argument('--dataset', default='dtu', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testlist', help='testing scene list')
parser.add_argument('--interval_scale', type=float, default=1.0, required=True, help='the depth interval scale')
parser.add_argument('--n_views', type=int, default=3, help='num of view')
parser.add_argument('--max_h', type=int, default=-1, help='testing max h')
parser.add_argument('--max_w', type=int, default=-1, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')
parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
# parser.add_argument('--n_depth_planes', type=int, default=192, help='the number of depth plane candidates')

parser.add_argument('--ckpt_path', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--depth_hypotheses', type=str, default=None, help='ndepths')
parser.add_argument('--depth_inter_ratio', type=str, default=None, help='depth_intervals_ratio')
parser.add_argument('--num_stages', type=int, default=4, help='number of cascade stages')
parser.add_argument('--num_workers', type=int, default=4, help='depth_filer worker')
# parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')

parser.add_argument('--filter_method', type=str, default='pcd', choices=["gipuma", "pcd", "dpcd"], help="filter method")

# filter
parser.add_argument('--conf_thr', type=str, default='0.5,0.5,0.5,0.5', help='prob confidence')
parser.add_argument('--nview_thr', type=int, default=3, help='threshold of num view')
parser.add_argument('--disp_thr', type=float, default=1.0, help='threshold of disparity')
parser.add_argument('--downsample', type=float, default=None, help='downsampling point cloud')

# filter by gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='./fusibile/fusibile')
parser.add_argument('--prob_threshold', type=str, default='0.0,0.0,0.0')
parser.add_argument('--disp_threshold', type=float, default='0.2')
parser.add_argument('--num_consistent', type=float, default='3')


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


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


# run model to save depth maps and confidence maps
def build_3d_model(args, config, scene_list):
    # model
    # build models architecture
    model_kwargs = config["arch"]["args"]
    if args.depth_hypotheses is not None:
        model_kwargs["depth_hypotheses"] = [int(nd) for nd in str(args.depth_hypotheses).split(",")]
    if args.depth_inter_ratio is not None:
        model_kwargs["depth_intervals_ratio"] = [float(r) for r in str(args.depth_inter_ratio).split(",")]
    
    print("model params: ", model_kwargs)
    model = CDSMVSNet(**model_kwargs)

    print('Loading checkpoint: {} ...'.format(args.ckpt_path))
    checkpoint = torch.load(str(args.ckpt_path))
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, val in state_dict.items():
        new_state_dict[key.replace('model.', '')] = val
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(new_state_dict)

    # prepare models for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # dataset, dataloader
    dataset_kwags = {
        "datapath": args.testpath,
        "scene_list": scene_list,
        "mode": "test",
        "nviews": args.n_views,
        "ndepths": model_kwargs["depth_hypotheses"][0] * model_kwargs["depth_intervals_ratio"][0],
        "interval_scale": args.interval_scale,
        # "shuffle": False,
        # "batch_size": 1,
        "fix_res": args.fix_res,
        "max_h": args.max_h,
        "max_w": args.max_w,
        # "dataset_eval": args.dataset,
        "num_stages": args.num_stages
    }

    test_dataset = MVSDataset(**dataset_kwags)
    test_data_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    times = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_data_loader):
            # torch.cuda.synchronize()
            start_time = time.time()
            sample_cuda = tocuda(sample)
            num_stages = args.num_stages
            imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]
            outputs = model(imgs, cam_params, sample_cuda["depth_values"])
            # torch.cuda.synchronize()
            # outputs["ps_map"] = model.feature.extract_ps_map()

            end_time = time.time()
            times.append(end_time - start_time)
            outputs = tensor2numpy(outputs)
            del sample_cuda

            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(num_stages)].numpy()
            imgs = sample["imgs"].numpy()
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(test_data_loader), end_time - start_time,
                                                      outputs["out_depth"][0].shape))

            # save depth maps and confidence maps
            for filename, cam, img, depth_est, conf_stage1, conf_stage2, conf_stage3, conf_stage4 in zip(filenames, cams, imgs, outputs["out_depth"],
                                                                                            outputs["stage1"]["photometric_confidence"], 
                                                                                            outputs["stage2"]["photometric_confidence"],
                                                                                            outputs["stage3"]["photometric_confidence"], outputs["stage4"]["photometric_confidence"]): #, outputs["ps_map"]):
                img = img[0]  # ref view
                cam = cam[0]  # ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                #ps_filename = os.path.join(args.outdir, filename.format('ps_maps', '.png'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                #os.makedirs(ps_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                # save confidence maps
                h, w = depth_est.shape[0], depth_est.shape[1]
                conf_stage1 = cv2.resize(conf_stage1, (w, h), interpolation=cv2.INTER_NEAREST)
                conf_stage2 = cv2.resize(conf_stage2, (w, h), interpolation=cv2.INTER_NEAREST)
                conf_stage3 = cv2.resize(conf_stage3, (w, h), interpolation=cv2.INTER_NEAREST)
                conf_stage4 = cv2.resize(conf_stage4, (w, h), interpolation=cv2.INTER_NEAREST)
                photometric_confidence = np.stack([conf_stage1, conf_stage2, conf_stage3, conf_stage4]).transpose([1,2,0])
                save_pfm(confidence_filename, photometric_confidence)
                # save cams, img
                img = np.transpose(img, (1, 2, 0))
                # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

                write_cam(cam_filename, cam)

    print("average time: ", sum(times) / len(times))
    torch.cuda.empty_cache()
    gc.collect()

    for scene_name in scene_list:
        save_file = f'{args.outdir}/{scene_name}.ply'
        mvs_rgbd_dir = f'{args.outdir}/{scene_name}'
        pair_folder = f'{args.testpath}/{scene_name}'
        
        depth_fusion.main(mvs_rgbd_dir, pair_folder, save_file, args, device)


if __name__ == '__main__':
    config = read_json(args.config_path)

    if os.path.isfile(args.testlist):
        with open(args.testlist) as f:
            content = f.readlines()
            scene_list = [line.rstrip() for line in content]
    elif isinstance(args.testlist, str):
        scene_list = args.testlist.split(",")
    else:
        raise f"unknown {args.testlist}"

    for scene_name in scene_list:
        build_3d_model(args, config, [scene_name])
