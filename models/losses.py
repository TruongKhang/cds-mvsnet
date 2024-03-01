import torch
import torch.nn as nn
import torch.nn.functional as F


def final_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    depth_interval = kwargs.get("depth_interval", 1.0)
    depth_interval = depth_interval.unsqueeze(-1).unsqueeze(-1)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    depth_loss = 0.0

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"] / depth_interval
        depth_gt = depth_gt_ms[stage_key] / depth_interval
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        init_depth_loss = 0.0
        if "init_depth" in stage_inputs:
            init_depth = stage_inputs["init_depth"] / depth_interval
            init_depth_loss = F.l1_loss(init_depth[mask], depth_gt[mask], reduction='mean')

        feat_loss = 0.0
        if "feat_distance" in stage_inputs:
            feat_dis = stage_inputs["feat_distance"]
            target = stage_inputs["feat_target"]
            ndepths = target.size(1)
            mask = mask.unsqueeze(1).repeat(1, ndepths, 1, 1)
            pos_pixels = target[mask].sum()
            neg_pixels = torch.numel(target[mask]) - pos_pixels
            balanced_weight = neg_pixels / pos_pixels
            feat_loss = F.binary_cross_entropy(feat_dis[mask], target[mask], reduction='mean',
                                                           pos_weight=balanced_weight)
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss = total_loss + depth_loss_weights[stage_idx] * (depth_loss + 5 * feat_loss + 0.1 * init_depth_loss)
        else:
            total_loss += 1.0 * (depth_loss+ 5 * feat_loss + 0.1 * init_depth_loss)

    return total_loss, depth_loss
