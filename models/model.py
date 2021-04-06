import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import depth_regression, conf_regression, CostRegNet, FeatureNet, RefineNet, get_depth_range_samples
from .utils.warping import homo_warping_3D
from .prior_net import PriorNet


Align_Corners_Range = False


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None,
                log=False):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping_3D(src_fea, src_proj_new, ref_proj_new, depth_values)
            # warped_volume = homo_warping(src_fea, src_proj[:, 2], ref_proj[:, 2], depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_variance)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        # log_prob_volume = F.log_softmax(prob_volume_pre, dim=1)
        prob_volume = F.softmax(prob_volume_pre, dim=1) if not log else F.log_softmax(prob_volume_pre, dim=1)

        return prob_volume


class SeqProbMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=(48, 32, 8), depth_interals_ratio=(4, 2, 1), share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=(8, 8, 8), pretrained_prior=None,
                 pretrained_mvs=None, use_prior=True):
        super(SeqProbMVSNet, self).__init__()
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

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)
        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i])
                                                      for i in range(self.num_stage)])
        if self.refine:
            print("Perform depth refinement network")
            self.refine_network = RefineNet(self.feature.out_channels[-1] + 4)

        self.dnet = DepthNet()
        if use_prior:
            self.pr_net = PriorNet()
            if pretrained_prior is not None:
                print("Load pretrained prior net")
                ckpt = torch.load(pretrained_prior)
                self.pr_net.load_state_dict(ckpt['state_dict'])
                for p in self.pr_net.parameters():
                    p.requires_grad = False

        if pretrained_mvs is not None:
            print("Load pretrained CasMVSNet")
            ckpt = torch.load(pretrained_mvs)
            feat_dict, cost_reg_dict = {}, {}
            for k, v in ckpt['model'].items():
                if "feature" in k:
                    feat_dict[k.replace("feature.", "")] = v
                if "cost_regularization" in k:
                    cost_reg_dict[k.replace("cost_regularization.", "")] = v
            self.feature.load_state_dict(feat_dict)
            self.cost_regularization.load_state_dict(cost_reg_dict)

        self.mvsnet_parameters = list(self.feature.parameters()) + list(self.cost_regularization.parameters())
        #if self.refine:
        #    self.mvsnet_parameters += list(self.refine_network.parameters())

    def get_log_prior(self, depth, conf, depth_values):
        min_depth, max_depth = depth_values[:, [0], ...], depth_values[:, [-1], ...]
        mask = (depth >= min_depth) & (depth <= max_depth)
        mask = mask.repeat(1, depth_values.size(1), 1, 1)
        log_dist_masked = torch.zeros_like(depth_values)
        log_dist = - conf * torch.abs(depth - depth_values)
        log_dist_masked[mask] = log_dist[mask]
        return log_dist_masked

    def forward(self, imgs, proj_matrices, depth_values, prior=None, depth_scale=1.0):

        # prior_depths, prior_confs, is_begin = prior if prior is not None else

        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        batch_size, height, width = imgs.size(0), imgs.size(3), imgs.size(4)

        outputs = {}
        depth, cur_depth = None, None
        feat_img = features[0]["stage3"].detach()
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            stage_name = "stage{}".format(stage_idx + 1)
            features_stage = [feat[stage_name] for feat in features]
            proj_matrices_stage = proj_matrices[stage_name]
            stage_scale = self.stage_infos[stage_name]["scale"]

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1), [height, width], mode='bilinear',
                                          align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                        ndepth=self.ndepths[stage_idx],
                                                        depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                        dtype=imgs[0].dtype,
                                                        device=imgs[0].device,
                                                        shape=[batch_size, height, width],
                                                        max_depth=depth_max,
                                                        min_depth=depth_min)

            # added by Khang
            depth_values_stage = F.interpolate(depth_range_samples.unsqueeze(1),
                                               [self.ndepths[stage_idx], height//int(stage_scale), width//int(stage_scale)],
                                               mode='trilinear', align_corners=Align_Corners_Range).squeeze(1)

            log = False if self.refine else True

            log_likelihood = self.dnet(features_stage, proj_matrices_stage, depth_values=depth_values_stage,
                                       num_depth=self.ndepths[stage_idx],
                                       cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx], log=log)

            outputs_stage = {}
            if prior is not None:
                prior_depths, prior_confs = prior[stage_name]
                est_prior_depth, est_prior_conf = self.pr_net(prior_depths, prior_confs)
                outputs_stage.update({"prior_depth": est_prior_depth, "prior_conf": est_prior_conf})
            else:
                est_prior_depth, est_prior_conf = None, None

            if self.refine:
                mvs_depth = depth_regression(log_likelihood, depth_values_stage)
                outputs_stage["mvs_depth"] = mvs_depth
                mvs_depth = mvs_depth.unsqueeze(1)
                feat_img_stage = F.interpolate(feat_img, [height//int(stage_scale), width//int(stage_scale)], mode='nearest')
                if prior is not None:
                    input_p = (est_prior_depth, est_prior_conf)
                    depth, final_conf = self.refine_network(mvs_depth.detach(), stage_idx, input_p,
                                                            feat_img=feat_img_stage)
                else:
                    if stage_idx == 0:
                        depth, final_conf = mvs_depth, None
                    else:
                        depth, final_conf = self.refine_network(mvs_depth.detach(), stage_idx,
                                                                (mvs_depth.detach(), conf_regression(log_likelihood, n=2)),
                                                                feat_img=feat_img_stage)
            else:
                log_prior = self.get_log_prior(est_prior_depth, est_prior_conf, depth_values_stage / depth_scale) if prior is not None else 0.0
                log_posterior = log_likelihood + log_prior
                posterior_vol = F.softmax(log_posterior, dim=1)
                depth = depth_regression(posterior_vol, depth_values_stage)
                final_conf = conf_regression(posterior_vol)
                # likelihood = F.softmax(log_likelihood, dim=1)
                # mvs_depth = depth_regression(likelihood, depth_values_stage)
                # mvs_conf = conf_regression(likelihood)

            outputs_stage["depth"] = depth
            if final_conf is not None:
                outputs_stage["photometric_confidence"] = final_conf
            outputs[stage_name] = outputs_stage
            outputs.update(outputs_stage)
            # depth_range_values["stage{}".format(stage_idx + 1)] = depth_values_stage.detach()

        # depth map refinement
        # if self.refine:
        #     refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
        #     outputs["refined_depth"] = refined_depth
        # outputs["depth_candidates"] = depth_range_values
        # outputs["total_loss_vol"] = total_loss

        return outputs
