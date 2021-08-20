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
        # self.data_loader.set_device(self.device)
        # if len_epoch is None:
            # epoch-based training
        # self.len_epoch = len(self.data_loader)
        # else:
        #     # iteration-based training
        #     self.data_loader = inf_loop(data_loader)
        #     self.len_epoch = len_epoch
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
        if epoch <= 6:
            p = (epoch - 1) / 3.0
            temperature = np.power(10.0, -p)
        else:
            temperature = 0.01
        print('Epoch {} temperature {}'.format(epoch, temperature))

        # self.data_loader.dataset.generate_indices()
        outputs = None
        # training
        for dl in self.data_loader:
            dataset_name = dl.mvs_dataset.datapath
            dlossw = self.config["trainer"]["dlossw"]
            if 'blended' in dataset_name:
                dlossw = [w * 1.0 for w in dlossw]
            for batch_idx, sample in enumerate(dl): #self.data_loader):
                start_time = time.time()

                # modified from the original by Khang
                sample_cuda = tocuda(sample)
                # is_begin = sample_cuda['is_begin'].type(torch.uint8)
                depth_gt_ms = sample_cuda["depth"]
                mask_ms = sample_cuda["mask"]
                #num_stage = len(self.config["arch"]["args"]["ndepths"])
                #depth_gt = depth_gt_ms["stage{}".format(num_stage)]
                #mask = mask_ms["stage{}".format(num_stage)]

                imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]

                self.optimizer.zero_grad()

                depth_values = sample_cuda["depth_values"]
                depth_interval = depth_values[:, 1] - depth_values[:, 0]
                outputs = self.model(imgs, cam_params, depth_values, gt_depths=depth_gt_ms, temperature=temperature)

                loss, depth_loss = self.criterion(outputs, depth_gt_ms, mask_ms, dlossw=dlossw, depth_interval=depth_interval)
                loss.backward()
                self.optimizer.step()
                # self.lr_scheduler.step()

                if batch_idx % self.log_step == 0:
                    # save_scalars(self.writer, 'train', scalar_outputs, global_step)
                    # save_images(self.writer, 'train', image_outputs, global_step)
                    print(
                        "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                            epoch, self.epochs, batch_idx, len(dl),
                            self.optimizer.param_groups[0]["lr"], loss, depth_loss, time.time() - start_time))
                # del scalar_outputs, image_outputs
                self.train_metrics.update({"loss": loss.item(), "depth_loss": depth_loss.item()}, n=imgs.size(0))
        self.lr_scheduler.step()

        if (epoch % self.config["trainer"]["eval_freq"] == 0) or (epoch == self.epochs - 1):
            del outputs
            self._valid_epoch(epoch, 0.01)

        return self.train_metrics.mean()

    def _valid_epoch(self, epoch, temperature, save_folder='saved/samples'):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        #print("Validation at epoch %d, size of validation set: %d, batch_size: %d" % (epoch, len(self.valid_data_loader),
        #                                                                             self.valid_data_loader.batch_size))

        self.model.eval()
        with torch.no_grad():
            for dl in self.valid_data_loader:
                dataset_name = dl.mvs_dataset.datapath
                self.valid_metrics.reset()
                dlossw = self.config["trainer"]["dlossw"]
                if 'blended' in dataset_name:
                    dlossw = [w * 1.0 for w in dlossw]
                for batch_idx, sample in enumerate(dl): #self.valid_data_loader):
                    start_time = time.time()

                    # modified from the original by Khang
                    sample_cuda = tocuda(sample)
                    # is_begin = sample['is_begin'].type(torch.uint8)
                    depth_gt_ms = sample_cuda["depth"]
                    mask_ms = sample_cuda["mask"]
                    num_stage = 4 #len(self.config["arch"]["args"]["ndepths"])
                    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
                    mask = mask_ms["stage{}".format(num_stage)]

                    imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]

                    depth_values = sample_cuda["depth_values"]
                    depth_interval = depth_values[:, 1] - depth_values[:, 0]
                    outputs = self.model(imgs, cam_params, depth_values, temperature=temperature) #, gt_depths=depth_gt_ms)

                    loss, depth_loss = self.criterion(outputs, depth_gt_ms, mask_ms, dlossw=dlossw, depth_interval=depth_interval)

                    depth_est = outputs["refined_depth"].detach()
                    di = depth_interval[0].item() / 2.65
                    scalar_outputs = {"loss": loss,
                                      "depth_loss": depth_loss,
                                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*2),
                                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*4),
                                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*8),
                                      "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*14),
                                      "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*20),

                                      "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                                 [0, di*2.0]),
                                      "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                                 [di*2.0, di*4.0]),
                                      "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                                 [di*4.0, di*8.0]),
                                      "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                                  [di*8.0, di*14.0]),
                                      "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                                  [di*14.0, di*20.0]),
                                      "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                                                   [di*20.0, 1e5]),
                                      }

                    if batch_idx % self.log_step == 0:
                        # save_scalars(logger, 'test', scalar_outputs, global_step)
                        # save_images(logger, 'test', image_outputs, global_step)
                        print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, time = {:3f}".format(
                            epoch, self.epochs, batch_idx, len(dl), loss,
                            scalar_outputs["depth_loss"],
                            time.time() - start_time))
                    self.valid_metrics.update(tensor2float(scalar_outputs))
                    del scalar_outputs  # , image_outputs
                print(dataset_name, "avg_test_scalars:", self.valid_metrics.mean())

        # save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        # print("avg_test_scalars:", self.valid_metrics.mean())

        return self.valid_metrics.mean()
