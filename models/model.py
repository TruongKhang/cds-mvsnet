import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from models.module import depth_regression, conf_regression, FeatureNet, CostRegNet, get_depth_range_samples, Conv2d, ConvBlock
from models.utils.warping import homo_warping_3D
from models.attention import TopicFormer

Align_Corners_Range = False


class StageNet(nn.Module):
    def __init__(self, dim, nhead, pool_layers=["seed"]*5, n_merge_layers=1, n_topics=100, n_samples=8, topic_dim=None, attn_type="vanilla", vis_net=True):
        super(StageNet, self).__init__()

        self.topicfm = TopicFormer(dim, nhead, pool_layers, n_merge_layers, n_topics, n_samples, topic_dim, attn_type=attn_type) # if n_merge_layers > 0 else None
        self.vis = nn.Sequential(ConvBlock(2, 32), ConvBlock(32, 32),
                                 ConvBlock(32, 32), nn.Conv2d(32, 1, 1), nn.Sigmoid()) if vis_net else None
        self.cost_scale = nn.Parameter(10 * torch.ones(1, 1, 1, 1))

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, stage_idx=0, gt_depth=None, topic_init=None, vis_init=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices)-1, "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        n_src_views = len(proj_matrices) - 1

        # step 1. feature extraction
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        volume_sum = 0.0 # ref_volume
        init_volume_sum = 0.0
        vis_sum = 0.0
        topic_init = [None] * n_src_views if topic_init is None else topic_init
        learned_topics = []
        all_vis_maps = []
        for src_idx, (feat, src_proj) in enumerate(zip(features, src_projs)):
            # # extract features
            ref_fea, src_fea = feat["ref"], feat["src"]
            # if self.topicfm is not None:
            ref_h, src_h = ref_fea.shape[2], src_fea.shape[2]
            ref_fea = rearrange(ref_fea, 'n c h w -> n (h w) c')
            src_fea = rearrange(src_fea, 'n c h w -> n (h w) c')
            ref_fea, src_fea, topic = self.topicfm(ref_fea, src_fea, topic_init[src_idx])
            ref_fea = rearrange(ref_fea, "n (h w) c -> n c h w", h=ref_h)
            src_fea = rearrange(src_fea, "n (h w) c -> n c h w", h=src_h)
            if stage_idx == 0:
                learned_topics.append(topic)

            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping_3D(src_fea, src_proj_new, ref_proj_new, depth_values)

            ref_volume = ref_fea.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
            in_prod_vol = (ref_volume * warped_volume)
            init_vol = in_prod_vol.mean(dim=1) * (in_prod_vol.shape[1]**.5)
            init_vol = F.softmax(init_vol, dim=1)

            if self.vis is not None:
                entropy = (- init_vol * torch.log(init_vol + 1e-6)).sum(dim=1, keepdim=True)
                max_cost = torch.max(init_vol, dim=1, keepdim=True)[0]
                vis_weight = self.vis(torch.cat((max_cost, entropy), dim=1).detach())
            else:
                if vis_init is not None:
                    vis_weight = F.interpolate(vis_init[src_idx], [ref_fea.shape[2], ref_fea.shape[3]], mode='bilinear', align_corners=Align_Corners_Range)
                else:
                    raise "unkown visibility map"

            all_vis_maps.append(vis_weight.detach())
            # if self.training:
            volume_sum = volume_sum + in_prod_vol * vis_weight.unsqueeze(1)
            vis_sum = vis_sum + vis_weight
            init_volume_sum = init_volume_sum + init_vol * vis_weight
                # feat_distance_vol = feat_distance_vol + topic_sim_vol * vis_weight
            # else:
            #     volume_sum += in_prod_vol * vis_weight.unsqueeze(1)
            #     vis_sum += vis_weight

            # if gt_depth is not None:
            #     gt_warped_vol = homo_warping_3D(src_theta, src_proj_new, ref_proj_new, gt_depth)
            #     sim_vol = torch.sum(ref_theta.unsqueeze(2) * gt_warped_vol, dim=1)
            #     gt_feat_distance = gt_feat_distance + sim_vol * vis_weight

        # aggregate multiple feature volumes by variance
        volume_mean = volume_sum / (vis_sum.unsqueeze(1) + 1e-6)
        init_volume_mean = init_volume_sum / (vis_sum + 1e-6)
        init_depth = depth_regression(init_volume_mean, depth_values=depth_values)
        # feat_distance_vol = feat_distance_vol / (vis_sum + 1e-6)
        # if gt_depth is not None:
        #     gt_feat_distance = gt_feat_distance / (vis_sum + 1e-6) 
        #     feat_distance_vol = torch.cat((feat_distance_vol, gt_feat_distance), dim=1)

        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_mean)
        prob_volume_pre = cost_reg.squeeze(1) * self.cost_scale

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)
        photometric_confidence = conf_regression(prob_volume, n=(4 - stage_idx))

        # return {"depth": depth,  "photometric_confidence": photometric_confidence, "feat_distance": feat_distance_vol, "learned_topics": learned_topics} if self.training else {"depth": depth,  "photometric_confidence": photometric_confidence, "learned_topics": learned_topics}
        return {"depth": depth,  "photometric_confidence": photometric_confidence, "learned_topics": learned_topics, "init_depth": init_depth, "vis_maps": all_vis_maps}


class CDSMVSNet(nn.Module):
    def __init__(self, num_stages=3, depth_hypotheses=(48, 32, 8), depth_intervals_ratio=(4, 2, 1), share_cr=False,
                 cr_base_chs=(8, 8, 8), pretrained_fpn=None,
                 topicfm_cfg={"n_topics": 128, "n_samples": 8, "n_merge_layers": 2, "attn_type": "linear"}):
        super(CDSMVSNet, self).__init__()
        self.share_cr = share_cr
        self.depth_hypotheses = depth_hypotheses
        self.depth_intervals_ratio = depth_intervals_ratio
        self.cr_base_chs = cr_base_chs
        self.num_stages = num_stages

        print("**********netphs:{}, depth_intervals_ratio:{},  pretrained FPN:{}, chs:{}************".format(depth_hypotheses,
              depth_intervals_ratio, pretrained_fpn, self.cr_base_chs))

        assert len(depth_hypotheses) == len(depth_intervals_ratio)

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
            cost_reg_levels = [3, 3, 2, 1]

        self.feature = FeatureNet(base_channels=12)
        if pretrained_fpn:
            print(f"Loading feature extractor from {pretrained_fpn}")
            state_dict = torch.load(pretrained_fpn)["state_dict"]
            self.feature.load_state_dict(state_dict)
            for p in self.feature.parameters():
                p.requires_grad = False
        
        stage_nets = []
        nheads = [4,2,1,1]
        vis_nets = [True, True, False, False]
        for i in range(self.num_stages):
            # topicfm_cfg["n_merge_layers"] = 0 if i > 2 else 1
            if i == 0:
                topicfm_cfg = {"n_topics": 256, "n_samples": 0, "n_merge_layers": 2, "attn_type": "linear"}
                snet = StageNet(self.feature.out_channels[i], nheads[i], vis_net=vis_nets[i], topic_dim=None, **topicfm_cfg)
            # elif i == (self.num_stages-1):
            #     topicfm_cfg = {"n_samples": 0, "n_merge_layers": 0, "attn_type": "linear"}
            #     snet = StageNet(self.feature.out_channels[i], nheads[i], topic_dim=self.feature.out_channels[0], **topicfm_cfg)
            else:
                topicfm_cfg = {"n_samples": 0, "n_merge_layers": 1, "attn_type": "linear"}
                snet= StageNet(self.feature.out_channels[i], nheads[i], vis_net=vis_nets[i], topic_dim=self.feature.out_channels[0], **topicfm_cfg)
            stage_nets.append(snet)

        self.stage_nets = nn.ModuleList(stage_nets)
        # for p in self.stage_nets.parameters():
        #     p.requires_grad = False
        
        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i], n_levels=cost_reg_levels[i])
                                                      for i in range(self.num_stages)])
        # for p in self.cost_regularization.parameters():
        #     p.requires_grad = False
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
            self.load_state_dict(new_state_dict)


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
            ref_feat = self.feature(ref_img)
            src_feat = self.feature(src_img)
            features.append({"ref": ref_feat, "src": src_feat})

        outputs = {}
        depth, cur_depth = None, None
        topic_init = None
        vis_init = None
        for stage_idx in range(self.num_stages):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            stage_name = "stage{}".format(stage_idx + 1)
            features_stage = [{"ref": feat["ref"][stage_name], "src": feat["src"][stage_name]} for feat in features]
            # features_stage = [feat[stage_name] for feat in features]
            proj_matrices_stage = proj_matrices[stage_name]
            stage_scale = self.stage_infos[stage_name]["scale"]
            gt_depth_stage = gt_depths[stage_name].unsqueeze(1) if gt_depths is not None else None
            di_stage = depth_interval.unsqueeze(1) * self.depth_intervals_ratio[stage_idx]

            if depth is not None:
                cur_depth = depth.detach()
                cur_depth = F.interpolate(cur_depth.unsqueeze(1), [height//int(stage_scale), width//int(stage_scale)], mode='bilinear',
                                          align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values

            depth_samples = get_depth_range_samples(cur_depth=cur_depth, ndepth=self.depth_hypotheses[stage_idx],
                                                          depth_inteval_pixel=self.depth_intervals_ratio[stage_idx] * depth_interval,
                                                          dtype=imgs[0].dtype, device=imgs[0].device,
                                                          shape=[batch_size, height//int(stage_scale), width//int(stage_scale)],
                                                          max_depth=depth_max, min_depth=depth_min)

            outputs_stage = self.stage_nets[stage_idx](features_stage, proj_matrices_stage, depth_values=depth_samples, 
                                                       num_depth=self.depth_hypotheses[stage_idx],
                                                       cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx],
                                                       gt_depth=gt_depth_stage, stage_idx=stage_idx, topic_init=topic_init, vis_init=vis_init)
            vis_init = outputs_stage["vis_maps"]
            if stage_idx == 0:
                topic_init = outputs_stage["learned_topics"]

            depth = outputs_stage['depth']

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
