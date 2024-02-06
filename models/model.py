import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module import depth_regression, conf_regression, FeatureNet, CostRegNet, Refinement, get_depth_range_samples, ConvBnReLU, Conv2d
from models.utils.warping import homo_warping_3D
from models.dynamic_conv import compute_Fmatrix, compute_epipole

Align_Corners_Range = False


class StageNet(nn.Module):
    def __init__(self, num_mvs_stages=3):
        super(StageNet, self).__init__()
        # self.vis = nn.ModuleList([nn.Sequential(ConvBnReLU(2, 16), ConvBnReLU(16, 16), ConvBnReLU(16, 16), nn.Conv2d(16, 1, 1), nn.Sigmoid()) for _ in range(num_mvs_stages)])
        self.vis = nn.ModuleList([nn.Sequential(Conv2d(1, 32, 5, padding=2), 
                                  Conv2d(32, 32, 5, padding=2), Conv2d(32, 32, 5, padding=2), 
                                  nn.Conv2d(32, 1, 1), nn.Sigmoid()) for _ in range(num_mvs_stages)])

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None, stage_idx=0,
                gt_depth=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices)-1, "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(proj_matrices)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        # ref_fea, src_feats = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        # ref_volume = ref_fea.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = 0.0 # ref_volume
        feat_distance_vol, gt_feat_distance = 0.0, 0.0
        vis_sum = 0.0
        nc_sum = 0.0
        for feat, src_proj in zip(features, src_projs):
            # # extract features
            ref_fea, ref_nc_sum, ref_nc = feat["ref"]
            src_fea, src_nc_sum, _ = feat["src"]
            #ref_fea, src_fea = feat["ref"], feat["src"]
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping_3D(src_fea, src_proj_new, ref_proj_new, depth_values)

            ref_volume = ref_fea.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
            in_prod_vol = ref_volume * warped_volume / ref_volume.shape[1]
            sim_vol = in_prod_vol.sum(dim=1)
            sim_vol_norm = F.softmax(sim_vol.detach(), dim=1)
            entropy = (- sim_vol_norm * torch.log(sim_vol_norm)).sum(dim=1, keepdim=True)
            vis_weight = self.vis[stage_idx](entropy)
            if self.training:
                volume_sum = volume_sum + in_prod_vol * vis_weight.unsqueeze(1)
                vis_sum = vis_sum + vis_weight
                nc_sum = nc_sum + (ref_nc_sum + src_nc_sum) / 2
                feat_distance_vol = feat_distance_vol + sim_vol * vis_weight
            else:
                volume_sum += in_prod_vol * vis_weight.unsqueeze(1)
                vis_sum += vis_weight
                nc_sum += (ref_nc_sum + src_nc_sum) / 2
                # feat_distance_vol += sim_vol * vis_weight

            if gt_depth is not None:
                gt_warped_vol = homo_warping_3D(src_fea, src_proj_new, ref_proj_new, gt_depth)
                sim_vol = torch.sum(ref_fea.unsqueeze(2) * gt_warped_vol, dim=1)
                #sim_vol_norm = F.softmax(sim_vol.detach(), dim=1)
                #entropy = (- sim_vol_norm * torch.log(sim_vol_norm)).sum(dim=1, keepdim=True)
                #vis_weight = self.vis[stage_idx](torch.cat((entropy, torch.sqrt(ref_nc.detach())), dim=1))
                gt_feat_distance = gt_feat_distance + sim_vol * vis_weight
                #feat_vis_sum = feat_vis_sum + vis_weight
            # del warped_volume
        # aggregate multiple feature volumes by variance
        # volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        volume_mean = volume_sum / (vis_sum.unsqueeze(1) + 1e-6) #volume_sum / (num_views - 1)
        feat_distance_vol = feat_distance_vol / (vis_sum + 1e-6)
        if gt_depth is not None:
            gt_feat_distance = gt_feat_distance / (vis_sum + 1e-6) #feat_distance_vol / (num_views - 1)
            feat_distance_vol = torch.cat((feat_distance_vol, gt_feat_distance), dim=1)
        nc_mean = nc_sum / (num_views - 1)

        # step 3. cost volume regularization
        # cost_reg = cost_regularization(volume_variance)
        cost_reg = cost_regularization(volume_mean)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        prob_volume_pre = cost_reg.squeeze(1) / 0.1

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)
        photometric_confidence = conf_regression(prob_volume)

        return {"depth": depth,  "photometric_confidence": photometric_confidence, "feat_distance": feat_distance_vol, "norm_curv": nc_mean} if self.training else {"depth": depth,  "photometric_confidence": photometric_confidence, "norm_curv": nc_mean}


class CDSMVSNet(nn.Module):
    def __init__(self, num_stages=3, ndepths=(48, 32, 8), depth_interals_ratio=(4, 2, 1), share_cr=False,
                 grad_method="detach", cr_base_chs=(8, 8, 8)):
        super(CDSMVSNet, self).__init__()
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.cr_base_chs = cr_base_chs
        self.num_stages = num_stages

        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        if self.num_stages == 3:
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
            cost_reg_levels = [3, 3, 3]
        else:
            self.stage_infos = {
                "stage1":{
                    "scale": 8.0,
                },
                "stage2": {
                    "scale": 4.0,
                },
                "stage3": {
                    "scale": 2.0,
                },
                "stage4": {
                    "scale": 1.0,
                }
            }
            cost_reg_levels = [3, 3, 3, 2]

        self.feature = FeatureNet(base_channels=8)
        # for p in self.feature.parameters():
        #     p.requires_grad = False
        self.stage_net = StageNet(num_mvs_stages=len(ndepths))
        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i], n_levels=cost_reg_levels[i])
                                                      for i in range(self.num_stages)])
        #self.depth_params = list(self.cost_regularization.parameters()) + list(self.stage_net.parameters())
    def load_pretrained_model(self, ckpt_path=None):
        if ckpt_path is not None:
            print('Loading checkpoint: {} ...'.format(ckpt_path))
            checkpoint = torch.load(str(ckpt_path))
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for key, val in state_dict.items():
                new_key = key.replace('model.', '')
                new_state_dict[new_key] = val
                if "refine_network" in new_key:
                    del new_state_dict[new_key]
                if "stage_net.vis" in new_key:
                    del new_state_dict[new_key]
            self.load_state_dict(new_state_dict, strict=False)
        # model.load_state_dict(state_dict)


    def forward(self, imgs, proj_matrices, depth_values, gt_depths=None, temperature=0.001):
        depth_min = depth_values[:, [0]].unsqueeze(-1).unsqueeze(-1) #float(depth_values[0, 0].cpu().numpy())
        depth_max = depth_values[:, [-1]].unsqueeze(-1).unsqueeze(-1) #float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_values[:, 1] - depth_values[:, 0]).unsqueeze(-1).unsqueeze(-1) #(depth_max - depth_min) / depth_values.size(1)

        batch_size, height, width = imgs.shape[0], imgs.shape[3], imgs.shape[4]

        # step 1. feature extraction
        features = []
        list_imgs = torch.unbind(imgs, dim=1)
        ref_img, src_imgs = list_imgs[0], list_imgs[1:]
        cam_params = torch.unbind(proj_matrices[f"stage{self.num_stages}"], dim=1)
        ref_proj, src_projs = cam_params[0], cam_params[1:]
        for src_img, src_proj in zip(src_imgs, src_projs):  #imgs shape (B, N, C, H, W)
            # compute epipoles
            fundamental_matrix = compute_Fmatrix(ref_proj, src_proj)
            ref_epipole = compute_epipole(fundamental_matrix)
            src_epipole = compute_epipole(torch.transpose(fundamental_matrix, 1, 2))
            ref_feat = self.feature(F.interpolate(ref_img, (height, width)), epipole=ref_epipole, temperature=temperature)
            src_feat = self.feature(F.interpolate(src_img, (height, width)), epipole=src_epipole, temperature=temperature)
            features.append({"ref": ref_feat, "src": src_feat})

        outputs = {}
        depth, cur_depth = None, None
        for stage_idx in range(self.num_stages):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            stage_name = "stage{}".format(stage_idx + 1)
            features_stage = [{"ref": feat["ref"][stage_name], "src": feat["src"][stage_name]} for feat in features]
            # features_stage = [feat[stage_name] for feat in features]
            proj_matrices_stage = proj_matrices[stage_name]
            stage_scale = self.stage_infos[stage_name]["scale"]
            gt_depth_stage = gt_depths[stage_name].unsqueeze(1) if gt_depths is not None else None
            di_stage = depth_interval.unsqueeze(1) * self.depth_interals_ratio[stage_idx]

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1), [height//int(stage_scale), width//int(stage_scale)], mode='bilinear',
                                          align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values
            depth_samples = get_depth_range_samples(cur_depth=cur_depth, ndepth=self.ndepths[stage_idx],
                                                          depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                          dtype=imgs[0].dtype, device=imgs[0].device,
                                                          shape=[batch_size, height//int(stage_scale), width//int(stage_scale)],
                                                          max_depth=depth_max, min_depth=depth_min)

            # depth_samples = F.interpolate(depth_range_samples.unsqueeze(1),
            #                               [self.ndepths[stage_idx], height//int(stage_scale), width//int(stage_scale)], mode='trilinear',
            #                               align_corners=Align_Corners_Range).squeeze(1)
            outputs_stage = self.stage_net(features_stage, proj_matrices_stage,
                                           depth_values=depth_samples,
                                           num_depth=self.ndepths[stage_idx],
                                           cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx],
                                           gt_depth=gt_depth_stage, stage_idx=stage_idx)

            depth = outputs_stage['depth']

            if gt_depths is not None:
                target = (depth_samples - gt_depth_stage).abs() / di_stage
                # target = (feat_depth_samples - gt_depth_stage).abs() / di_stage
                target = (target < 0.5 / self.depth_interals_ratio[stage_idx]).float()
                target = torch.cat((target, torch.ones_like(gt_depth_stage)), dim=1)
                outputs_stage.update({"feat_target": target})

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        outputs["out_depth"] = depth

        return outputs


if __name__ == '__main__':
    model = CDSMVSNet()
    model = model.to(torch.device('cuda'))
    result = model(torch.rand(1, 3, 3, 512, 640).cuda(), {"stage1": torch.rand(1, 3, 2, 4, 4).cuda(),
                                                          "stage2": torch.rand(1, 3, 2, 4, 4).cuda(),
                                                          "stage3": torch.rand(1, 3, 2, 4, 4).cuda()}, torch.arange(3, 100, 10, dtype=torch.float32).unsqueeze(0).repeat(1, 1).cuda())
