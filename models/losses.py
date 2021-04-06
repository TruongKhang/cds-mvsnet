import torch
import torch.nn as nn
import torch.nn.functional as F


def seq_prob_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    use_prior = kwargs.get("use_prior_loss", False)
    depth_scale = kwargs.get("depth_scale", 1.0)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    depth_loss = 0.0

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in ["stage1", "stage2", "stage3"]]: # inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        if stage_key == "stage3":
            conf = stage_inputs["photometric_confidence"]
            prob_loss, depth_loss = masked_prob_loss(depth_est, conf, (depth_gt, mask))
        else:
            depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
            prob_loss = depth_loss

        if "mvs_depth" in stage_inputs:
            mvs_depth = stage_inputs["mvs_depth"]
            mvs_depth_loss = F.smooth_l1_loss(mvs_depth[mask], depth_gt[mask], reduction='mean')
        else:
            mvs_depth_loss = 0.0

        if use_prior and ("prior_depth" in stage_inputs):
            scaled_depth_gt = depth_gt.unsqueeze(1) / depth_scale
            target = (scaled_depth_gt, mask.unsqueeze(1))
            prior_loss, _ = masked_prob_loss(stage_inputs["prior_depth"], stage_inputs["prior_conf"], target)
        else:
            prior_loss = 0.0

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss = total_loss + depth_loss_weights[stage_idx] * (prob_loss + prior_loss + mvs_depth_loss)
        else:
            total_loss += 1.0 * (prob_loss + prior_loss + mvs_depth_loss)

    return total_loss, depth_loss


def masked_prob_loss(depth, var, target):
    gt_depths, valid_mask = target
    # valid_mask = valid_mask.float()
    # cnt = gt_depths.size(0) * gt_depths.size(2) * gt_depths.size(3)
    gt = gt_depths[valid_mask]

    regl = torch.log(var + 1e-16)
    mean = depth[valid_mask]
    res = var[valid_mask]
    regl = regl[valid_mask]
    depth_error = torch.abs(gt - mean)
    final_loss = torch.mean(res * depth_error - regl) #torch.pow(gt - mean, 2) - regl)
    return final_loss, depth_error.mean()
