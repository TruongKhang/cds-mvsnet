import numpy as np
from pprint import pprint
import torch
import pytorch_lightning as pl
from pytorch_lightning.profilers import PassThroughProfiler

from models.model import CDSMVSNet
from models.losses import final_loss
from utils import AbsDepthError_metrics, Thres_metrics, DictAverageMeter, tensor2float


class PL_Trainer(pl.LightningModule):
    def __init__(self, config, profiler=None, ckpt_path=None):
        super(PL_Trainer, self).__init__()

        self.config = config

        self.profiler = profiler or PassThroughProfiler()

        # initialize model
        self.pretrained_ckpt_path = ckpt_path
        self.model = CDSMVSNet(**config["arch"]["args"])
        # self.model.load_pretrained_model(ckpt_path)
        
        self.loss_func = final_loss

        self.validation_step_outputs = {}

    def setup(self, stage):
        self.model.load_pretrained_model(self.pretrained_ckpt_path)

    def configure_optimizers(self):
        optim_name = self.config["optimizer"]["type"]
        optim_args = self.config["optimizer"]["args"]
        lr, weight_decay = optim_args["lr"], optim_args["weight_decay"]
        if optim_name.lower() == "adam":
            optim = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        elif optim_name.lower() == "adamw":
            optim = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        else:
            optim = torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        
        # lr scheduler
        lr_milestones = self.config["lr_scheduler"]["args"]["step_size"]
        lr_gamma = self.config["lr_scheduler"]["args"]["gamma"]

        LRScheduler = getattr(torch.optim.lr_scheduler, self.config["lr_scheduler"]["type"])
        scheduler = LRScheduler(optim, lr_milestones, gamma=lr_gamma)
        return {
            "optimizer": optim,
            "lr_scheduler": {"interval": "epoch", "scheduler": scheduler}
        }
    
    def training_step(self, batch_data, batch_idx):
        if self.current_epoch < 4:
            p = self.current_epoch / 2.0
            temperature = np.power(10.0, -p)
        else:
            temperature = 0.01
        
        # print('Epoch {} temperature {}'.format(self.current_epoch, temperature))
        if isinstance(batch_data, torch.Tensor):
            batch_data = [batch_data]
        combined_loss, combined_depth_loss = 0, 0
        for batch in batch_data:

        # with self.profiler.profile("mutile-stage depth prediction"):
            depth_gt_ms = batch["depth"]
            mask_ms = batch["mask"]
            imgs, cam_params = batch["imgs"], batch["proj_matrices"]
            depth_values = batch["depth_values"]
            depth_interval = depth_values[:, 1] - depth_values[:, 0]
            # depth_range = depth_values[:, -1] - depth_values[:, 0]
            outputs = self.model(imgs, cam_params, depth_values, gt_depths=depth_gt_ms, temperature=temperature)

        # with self.profiler.profile("loss computation"):
            dlossw = self.config["trainer"]["dlossw"]
            loss, depth_loss = self.loss_func(outputs, depth_gt_ms, mask_ms, dlossw=dlossw, depth_interval=depth_interval)

            combined_loss = combined_loss + loss
            combined_depth_loss = combined_depth_loss + depth_loss
        
        self.log("depth_loss", combined_depth_loss, on_step=True, on_epoch=True, prog_bar=True, 
                 logger=True, sync_dist=True, batch_size=len(batch_data[0]))
        return {"loss": combined_loss / len(batch_data)}
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with self.profiler.profile("mutile-stage depth prediction"):
            depth_gt_ms = batch["depth"]
            mask_ms = batch["mask"]
            num_stages = self.model.num_stages
            depth_gt = depth_gt_ms["stage{}".format(num_stages)]
            mask = mask_ms["stage{}".format(num_stages)]

            imgs, cam_params = batch["imgs"], batch["proj_matrices"]
            depth_values = batch["depth_values"]
            depth_interval = depth_values[:, 1] - depth_values[:, 0]
            
            outputs = self.model(imgs, cam_params, depth_values, temperature=0.01)

            depth_est = outputs["out_depth"].detach()
            di = depth_interval[0].item() / 2.65
            error_stats = {
                "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                "prec@2mm": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*2),
                "prec@4mm": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*4),
                "prec@8mm": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*8),
                # "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*14),
                # "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di*20),

                # "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di*2]),
                # "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [di*2, di*4]),
                # "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [di*4, di*8]),
                # "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [di*8, di*14]),
                # "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [di*14, di*20]),
                # "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [di*20, 1e5]),
            }
        if dataloader_idx not in self.validation_step_outputs:
            self.validation_step_outputs[dataloader_idx] = [error_stats]
        else:
            self.validation_step_outputs[dataloader_idx].append(error_stats)

        return error_stats
    
    def on_validation_epoch_end(self):
        prec_metrics = {}
        for idx, all_error_stats in self.validation_step_outputs.items():
            calculator = DictAverageMeter()
            for error_stats in all_error_stats:
                calculator.update(tensor2float(error_stats))

            avg_metrics = calculator.mean()
            for k, v in avg_metrics.items():
                prec_metrics[f"id{idx}_{k}"] = v
        
        for k, v in prec_metrics.items():
            self.log(f'{k}', v, sync_dist=True)
        
        self.validation_step_outputs.clear()

