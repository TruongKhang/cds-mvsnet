import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module import depth_regression, conf_regression, FeatureNet, CostRegNet, RefineNet, get_depth_range_samples
from models.utils.warping import homo_warping_3D
from models.dynamic_conv import compute_Fmatrix, compute_epipole

Align_Corners_Range = False


class StageNet(nn.Module):
    def __init__(self, base_channels):
        super(StageNet, self).__init__()
        self.feature_net = FeatureNet(base_channels=base_channels)

    def forward(self, imgs, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_img, src_imgs = imgs[0], imgs[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        volume_sum = 0.0
        # volume_sq_sum = ref_volume ** 2
        # del ref_volume
        for src_img, src_proj in zip(src_imgs, src_projs):
            # compute epipoles
            fundamental_matrix = compute_Fmatrix(ref_proj, src_proj)
            ref_epipole = compute_epipole(fundamental_matrix)
            src_epipole = compute_epipole(torch.transpose(fundamental_matrix, 1, 2))

            # extract features
            ref_fea = self.feature_net(ref_img, ref_epipole)
            src_fea = self.feature_net(src_img, src_epipole)
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping_3D(src_fea, src_proj_new, ref_proj_new, depth_values)

            ref_volume = ref_fea.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
            volume_sum = volume_sum + (ref_volume - warped_volume)**2
            # if self.training:
            #     volume_sum = volume_sum + warped_volume
            #     volume_sq_sum = volume_sq_sum + warped_volume ** 2
            # else:
            #     # TODO: this is only a temporal solution to save memory, better way?
            #     volume_sum += warped_volume
            #     volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            # del warped_volume
        # aggregate multiple feature volumes by variance
        # volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        volume_mean = volume_sum / (num_views - 1)

        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_mean)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)
        photometric_confidence = conf_regression(prob_volume)

        return {"depth": depth,  "photometric_confidence": photometric_confidence}


class TAMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=(48, 32, 8), depth_interals_ratio=(4, 2, 1), share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=(8, 8, 8)):
        super(TAMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            "stage1":{
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }

        # self.feature = FeatureNet(base_channels=16, arch_mode=self.arch_mode)
        self.stage_net = StageNet(base_channels=16)
        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.stage_net.feature_net.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.stage_net.feature_net.out_channels,
                                                                 base_channels=self.cr_base_chs[i])
                                                      for i in range(self.num_stage)])
        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        # features = []
        # for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
        #     img = imgs[:, nview_idx]
        #     features.append(self.feature(img))
        batch_size, nviews, height, width = imgs.shape[0], imgs.shape[1], imgs.shape[3], imgs.shape[4]

        outputs = {}
        depth, cur_depth = None, None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            # features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
            imgs_stage = [F.interpolate(imgs[:, view_idx], [height // int(stage_scale), width // int(stage_scale)],
                                        mode='bilinear', align_corners=Align_Corners_Range) for view_idx in range(nviews)]

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1), [height, width], mode='bilinear',
                                          align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth, ndepth=self.ndepths[stage_idx],
                                                          depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                          dtype=imgs[0].dtype, device=imgs[0].device,
                                                          shape=[batch_size, height, width],
                                                          max_depth=depth_max, min_depth=depth_min)

            outputs_stage = self.stage_net(imgs_stage, proj_matrices_stage,
                                           depth_values=F.interpolate(depth_range_samples.unsqueeze(1),
                                                                      [self.ndepths[stage_idx], height//int(stage_scale), width//int(stage_scale)], mode='trilinear',
                                                                      align_corners=Align_Corners_Range).squeeze(1),
                                           num_depth=self.ndepths[stage_idx],
                                           cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx])

            depth = outputs_stage['depth']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        # depth map refinement
        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth

        return outputs


if __name__ == '__main__':
    model = CascadeMVSNet()
    model = model.to(torch.device('cuda'))
    result = model(torch.rand(1, 3, 3, 512, 640).cuda(), {"stage1": torch.rand(1, 3, 2, 4, 4).cuda(),
                                                          "stage2": torch.rand(1, 3, 2, 4, 4).cuda(),
                                                          "stage3": torch.rand(1, 3, 2, 4, 4).cuda()}, torch.arange(3, 100, 10, dtype=torch.float32).unsqueeze(0).repeat(1, 1).cuda())
