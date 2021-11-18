import os, cv2
import numpy as np
from PIL import Image
from datasets.data_io import read_pfm
from utils import DictAverageMeter


def thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = np.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return 1.0 - float(np.mean(err_mask.astype(np.float32)))


class Evaluation(object):
    def __init__(self, gt_depth_folder, input_folder, list_scenes, method='casmvsnet', depth_folder='depth_est'):
        self.gt_depths, self.masks, self.est_depths = [], [], []
        self.method = method
        for scene in list_scenes:
            scene_gt_folder = os.path.join(gt_depth_folder, scene)
            if (method == 'casmvsnet') or (method == 'pvamvsnet') or (method == 'cvpmvsnet'):
                scene_est_depth = os.path.join(input_folder, scene, depth_folder) #"depth_est")
                # scene_est_conf = os.path.join(input_folder, scene, "confidence")
                indices = [int(f.split('.')[0]) for f in os.listdir(scene_est_depth) if os.path.isfile(os.path.join(scene_est_depth, f))] # and '_3.pfm' in f]
                indices.sort()
                for idx in indices:
                    mask_filename_hr = os.path.join(scene_gt_folder, 'depth_visual_{:0>4}.png'.format(idx))
                    depth_filename_hr = os.path.join(scene_gt_folder, 'depth_map_{:0>4}.pfm'.format(idx))
                    est_depth_filename = os.path.join(scene_est_depth, '{:0>8}.pfm'.format(idx))
                    # est_conf_filename = os.path.join(scene_est_conf, '{:0>8}.pfm'.format(idx))

                    self.gt_depths.append(depth_filename_hr)
                    self.masks.append(mask_filename_hr)
                    self.est_depths.append(est_depth_filename)
                    # self.est_confs.append(est_conf_filename)
            elif (method == 'mvsnet') or (method == 'rmvsnet'):
                scene_est = os.path.join(input_folder, scene, "depths_mvsnet") if method == 'mvsnet' else os.path.join(input_folder, scene, "depths_rmvsnet")
                indices = [int(f.split('_')[0]) for f in os.listdir(scene_est) if os.path.isfile(os.path.join(scene_est, f)) and ('prob' in f)]
                indices.sort()
                for idx in indices:
                    mask_filename_hr = os.path.join(scene_gt_folder, 'depth_visual_{:0>4}.png'.format(idx))
                    depth_filename_hr = os.path.join(scene_gt_folder, 'depth_map_{:0>4}.pfm'.format(idx))
                    est_depth_filename = os.path.join(scene_est, '{:0>8}_init.pfm'.format(idx))
                    est_conf_filename = os.path.join(scene_est, '{:0>8}_prob.pfm'.format(idx))

                    self.gt_depths.append(depth_filename_hr)
                    self.masks.append(mask_filename_hr)
                    self.est_depths.append(est_depth_filename)
                    # self.est_confs.append(est_conf_filename)

        self.eval_depth = DictAverageMeter()

    def scale_input(self, hr_img, max_w, max_h, base=32):
        hr_img_ds = cv2.resize(hr_img, (int(max_w), int(max_h)), interpolation=cv2.INTER_NEAREST)

        return hr_img_ds

    def read_mask_hr(self, filename, max_w, max_h):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.scale_input(np_img, max_w=max_w, max_h=max_h)

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_depth_hr(self, filename, max_w, max_h):
        # read pfm depth file
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.scale_input(depth_hr, max_w=max_w, max_h=max_h)

        return depth_lr

    def eval(self, max_w=1152, max_h=864):
        # batch_gt_depths, batch_est_depths, batch_masks = [], [], []
        print("Resolution test: ({},{})".format(max_h, max_w))
        for idx in range(len(self.gt_depths)):
            gt_depth = self.read_depth_hr(self.gt_depths[idx], max_w=max_w, max_h=max_h)
            mask = self.read_mask_hr(self.masks[idx], max_w=max_w, max_h=max_h)
            est_depth = np.array(read_pfm(self.est_depths[idx])[0], dtype=np.float32)

            eval_metrics = {"MAE": float(np.mean(np.abs(est_depth - gt_depth)[(mask > 0.5)])),
                            "RMSE": float(np.sqrt(np.mean(((est_depth - gt_depth)**2)[(mask > 0.5)]))),
                            "thresh1mm_error": thres_metrics(est_depth, gt_depth, (mask > 0.5), 1),
                            "thresh2mm_error": thres_metrics(est_depth, gt_depth, (mask > 0.5), 2),
                            "thresh4mm_error": thres_metrics(est_depth, gt_depth, (mask > 0.5), 4)}
            self.eval_depth.update(eval_metrics)
        return self.eval_depth.mean()


if __name__ == '__main__':
    root_dir = "/home/khangtg/Documents/lab/cds-mvsnet"
    mvsnet_input_folder = '/home/khangtg/Documents/lab/mvs/dataset/mvs/dtu_dataset/test' #/mnt/sdb/khang/dtu_dataset/test'
    rmvsnet_input_folder = '/home/khangtg/Documents/lab/mvs/dataset/mvs/dtu_dataset/test'
    casmvsnet_input_folder = 'casmvsnet_outputs' #'/mnt/sdb/khang/cascade-stereo/CasMVSNet/dtu_outputs'
    ours_input_folder = 'outputs_final' #'/mnt/sdb/khang/seq-prob-mvs/dtu_outputs'
    pvamvsnet_input_folder = 'pvamvsnet_outputs'
    cvpmvsnet_input_folder = 'cvpmvsnet_outputs'
    gt_depths_folder = '/home/khangtg/Documents/lab/mvs/dataset/mvs/dtu_dataset/test/Depths_raw' #'/mnt/sdb/khang/dtu_dataset/train/Depths_raw'

    with open('/home/khangtg/Documents/lab/cds-mvsnet/lists/dtu/test.txt') as f:
        list_scenes = [scene.strip() for scene in f]
    print(list_scenes)

    for i, depth_folder in enumerate(["depth_est"]): #['384_512', '576_832', '832_1152', '960_1280', '1088_1408']: #, '1152_1536']:
        input_folder = "{}/outputs".format(root_dir)
        # casmvsnet_input_folder = "/home/khangtg/Documents/lab/seq-prob-mvs//outputs"
        casmvsnet_eval = Evaluation(gt_depths_folder, input_folder, list_scenes, method='casmvsnet', depth_folder=depth_folder)
        max_h, max_w = 1152, 1536 #1152 // (2**(3 - i)), 1536 // (2**(3 - i)) #in_folder.split('_')
        metrics = casmvsnet_eval.eval(max_h=int(max_h), max_w=int(max_w))
        # print('AUSE of CasMVSNet: %f' % ause)
        print(depth_folder, metrics)