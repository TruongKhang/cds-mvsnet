import torch
import torch.nn as nn
import torch.nn.functional as F


def final_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    depth_interval = kwargs.get("depth_interval", 1.0)
    depth_interval = depth_interval.unsqueeze(-1).unsqueeze(-1)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    depth_loss = 0.0

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in ["stage1", "stage2", "stage3"]]: # inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"] / depth_interval
        depth_gt = depth_gt_ms[stage_key] / depth_interval
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        norm_curv_reg = torch.mean(stage_inputs["norm_curv"].squeeze(1)[mask])

        feat_loss = 0.0
        if "feat_distance" in stage_inputs:
            feat_dis = stage_inputs["feat_distance"]
            target = stage_inputs["feat_target"]
            ndepths = target.size(1)
            mask = mask.unsqueeze(1).repeat(1, ndepths, 1, 1)
            pos_pixels = target[mask].sum()
            neg_pixels = torch.numel(target[mask]) - pos_pixels
            balanced_weight = neg_pixels / pos_pixels
            feat_loss = F.binary_cross_entropy_with_logits(feat_dis[mask], target[mask], reduction='mean',
                                                           pos_weight=balanced_weight)
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss = total_loss + depth_loss_weights[stage_idx] * (depth_loss + 5 * feat_loss + 0.1*norm_curv_reg)
        else:
            total_loss += 1.0 * (depth_loss+ 5 * feat_loss + 0.1*norm_curv_reg)

    if "refined_depth" in inputs:
        depth_gt = depth_gt_ms["stage4"] / depth_interval
        depth_est = inputs["refined_depth"] / depth_interval
        mask = mask_ms["stage4"] > 0.5
        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
        total_loss = total_loss + 2*depth_loss

    return total_loss, depth_loss
