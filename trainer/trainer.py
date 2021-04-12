import numpy as np
import os
import torch
import torch.nn.functional as F
import time
from PIL import Image
import matplotlib.pyplot as plt

from base import BaseTrainer
from utils import AbsDepthError_metrics, Thres_metrics, tocuda, DictAverageMeter, inf_loop, tensor2float, tensor2numpy, save_images


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None):
        super().__init__(model, criterion, optimizer, config, writer=writer)
        self.config = config
        self.data_loader = data_loader
        self.data_loader.set_device(self.device)
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer']['logging_every'] # int(np.sqrt(data_loader.batch_size))
        self.depth_scale = config["trainer"]["depth_scale"]
        self.train_metrics = DictAverageMeter()
        self.valid_metrics = DictAverageMeter()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        print('Epoch {}:'.format(epoch))

        self.data_loader.dataset.generate_indices()
        # training
        for batch_idx, sample in enumerate(self.data_loader):
            start_time = time.time()

            # modified from the original by Khang
            sample_cuda = tocuda(sample)
            is_begin = sample_cuda['is_begin'].type(torch.uint8)
            depth_gt_ms = sample_cuda["depth"]
            mask_ms = sample_cuda["mask"]
            num_stage = len(self.config["arch"]["args"]["ndepths"])
            depth_gt = depth_gt_ms["stage{}".format(num_stage)]
            mask = mask_ms["stage{}".format(num_stage)]

            imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]

            self.optimizer.zero_grad()

            outputs = self.model(imgs, cam_params, sample_cuda["depth_values"])

            loss, depth_loss = self.criterion(outputs, depth_gt_ms, mask_ms, dlossw=self.config["trainer"]["dlossw"])
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # scalar_outputs = {"loss": loss,
            #                   "depth_loss": depth_loss,
            #                   "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
            #                   "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
            #                   "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
            #                   "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)}

            # image_outputs = {"depth_est": depth_est * mask,
            #                  "depth_est_nomask": depth_est,
            #                  "depth_gt": sample_cuda["depth"]["stage1"].cpu(),
            #                  "ref_img": sample_cuda["imgs"][:, 0].cpu(),
            #                  "mask": sample_cuda["mask"]["stage1"].cpu(),
            #                  "errormap": (depth_est - depth_gt).abs() * mask,
            #                  }

            if batch_idx % self.log_step == 0:
                # save_scalars(self.writer, 'train', scalar_outputs, global_step)
                # save_images(self.writer, 'train', image_outputs, global_step)
                print(
                    "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                        epoch, self.epochs, batch_idx, len(self.data_loader),
                        self.optimizer.param_groups[0]["lr"], loss, depth_loss, time.time() - start_time))
            # del scalar_outputs, image_outputs
            self.train_metrics.update({"loss": loss.item(), "depth_loss": depth_loss.item()}, n=depth_gt.size(0))

        if (epoch % self.config["trainer"]["eval_freq"] == 0) or (epoch == self.epochs - 1):
            self._valid_epoch(epoch)

        return self.train_metrics.mean()

    def _valid_epoch(self, epoch, save_folder='saved/samples'):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        print("Validation at epoch %d, size of validation set: %d, batch_size: %d" % (epoch, len(self.valid_data_loader),
                                                                                     self.valid_data_loader.batch_size))

        self.model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                start_time = time.time()

                # modified from the original by Khang
                sample_cuda = tocuda(sample)
                is_begin = sample['is_begin'].type(torch.uint8)
                depth_gt_ms = sample_cuda["depth"]
                mask_ms = sample_cuda["mask"]
                num_stage = len(self.config["arch"]["args"]["ndepths"])
                depth_gt = depth_gt_ms["stage{}".format(num_stage)]
                mask = mask_ms["stage{}".format(num_stage)]

                imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]

                outputs = self.model(imgs, cam_params, sample_cuda["depth_values"])

                loss, depth_loss = self.criterion(outputs, depth_gt_ms, mask_ms,
                                                  dlossw=self.config["trainer"]["dlossw"])

                depth_est = outputs["depth"].detach()

                scalar_outputs = {"loss": loss,
                                  "depth_loss": depth_loss,
                                  "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                                  "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                                  "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                                  "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                                  "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 14),
                                  "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 20),

                                  "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, 2.0]),
                                  "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                             [2.0, 4.0]),
                                  "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                             [4.0, 8.0]),
                                  "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                              [8.0, 14.0]),
                                  "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                              [14.0, 20.0]),
                                  "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                               [20.0, 1e5]),
                                  }

                """depth_est[depth_est > 1500] = 1500
                depth_est[depth_est < 400] = 400
                depth_est[0, 0] = 400
                prior_depth_est[prior_depth_est > 1500] = 1500
                prior_depth_est[prior_depth_est < 400] = 400
                prior_depth_est[0, 0] = 400
                mvs_depth_est[mvs_depth_est > 1500] = 1500
                mvs_depth_est[mvs_depth_est < 400] = 400
                mvs_depth_est[0, 0] = 400"""

                """error_map = (depth_est - depth_gt).abs()
                error_map[error_map > 20] = 20
                error_map[0, 0, 0] = 0
                error_mvs_depth = (mvs_depth_est - depth_gt).abs()
                error_mvs_depth[error_mvs_depth > 20] = 20
                error_mvs_depth[0, 0, 0] = 0
                error_prior_depth = (prior_depth_est - depth_gt).abs()
                error_prior_depth[error_prior_depth > 20] = 20
                error_prior_depth[0, 0, 0] = 0"""

                """prior_conf = outputs["prior_conf"].detach().squeeze(1)
                prior_conf = (prior_conf - torch.min(prior_conf)) / torch.max(prior_conf) * 255

                mvs_conf = outputs["mvs_conf"].detach() * 255
                mvs_conf[0, 0, 0] = 0
                final_conf = outputs["photometric_confidence"].detach() * 255
                final_conf[0, 0, 0] = 0

                image_outputs = {"final_conf": final_conf,
                                 "mvs_conf": mvs_conf,
                                 "prior_conf": prior_conf,
                                 "ref_img": sample_cuda["imgs"][:, 0].permute(0, 2, 3, 1).cpu() * 255,
                                 "mask": (sample_cuda["mask"]["stage3"].cpu() > 0.5).float() * 255,
                                 "final_depth": depth_est,
                                 "prior_depth": prior_depth_est,
                                 "mvs_depth": mvs_depth_est}

                image_outputs = tensor2numpy(image_outputs)
                for k, v in image_outputs.items():
                    v = np.squeeze(v, axis=0)
                    img = Image.fromarray(v.astype(np.uint8))
                    if 'depth' in k:
                        img = Image.fromarray(v.astype(np.uint16))
                        dir = '%s/depth' %save_folder
                    elif 'conf' in k:
                        dir = '%s/conf' %save_folder
                    elif 'img' in k:
                        dir = '%s/ref_img' %save_folder
                    else:
                        dir = '%s/mask' %save_folder
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    if 'depth' in k:
                        plt.imsave('%s/%s_%d.png' % (dir, k, batch_idx), v, vmin=400, vmax=1500)
                        img.save('%s/%s_%d.png' % (dir, k, batch_idx))
                        gt_dir = '%s/groundtruth_depth' % save_folder
                        if not os.path.exists(gt_dir):
                            os.makedirs(gt_dir)
                        gt_depth = (depth_gt*10).squeeze(0).cpu().numpy()
                        gt_depth = Image.fromarray(gt_depth.astype(np.uint16))
                        gt_depth.save('%s/%s_%d.png' % (gt_dir, k, batch_idx))"""

                """image_outputs = {"final_depth_masked": depth_est * mask,
                                 "final_depth": depth_est,
                                 "gt_depth": sample_cuda["depth"]["stage1"].cpu(),
                                 "ref_img": sample_cuda["imgs"][:, 0].cpu(),
                                 "mask": sample_cuda["mask"]["stage1"].cpu(),
                                 "errormap": error_map,
                                 "prior_depth": prior_depth_est,
                                 "error_prior_depth": error_prior_depth,
                                 "mvs_depth": mvs_depth_est,
                                 "error_mvs_depth": error_mvs_depth}
                save_images(self.writer, 'val', tensor2numpy(image_outputs), batch_idx)"""

                if batch_idx % self.log_step == 0:
                    # save_scalars(logger, 'test', scalar_outputs, global_step)
                    # save_images(logger, 'test', image_outputs, global_step)
                    print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, time = {:3f}".format(
                        epoch, self.epochs, batch_idx, len(self.valid_data_loader), loss, scalar_outputs["depth_loss"],
                        time.time() - start_time))
                self.valid_metrics.update(tensor2float(scalar_outputs))
                del scalar_outputs  # , image_outputs

        # save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", self.valid_metrics.mean())

        return self.valid_metrics.mean()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
