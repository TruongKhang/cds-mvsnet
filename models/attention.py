import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class MLP(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim):
        super(MLP, self).__init__()

        self.out = nn.Sequential(nn.Linear(in_dim, inter_dim),
                                 nn.GELU(),
                                 nn.Linear(inter_dim, out_dim),
                                #  nn.LayerNorm(out_dim)
                                 )
        
    def forward(self, x):
        return self.out(x)


class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                 proj_drop=0., drop_path=0., layer_scale=None):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_layer = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_layer = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = MLP(dim, dim*2, dim)

    def forward(self, q, kv):
        B, N, C = q.shape
    
        q_vec = self.q_layer(self.norm1(q)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_vec = self.kv_layer(self.norm1(kv)).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_vec, v_vec = kv_vec[0], kv_vec[1]
        attn = (q_vec @ k_vec.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v_vec).transpose(1, 2).reshape(B, -1, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x = q + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))

        return x
    
class LinearAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                 proj_drop=0., drop_path=0., layer_scale=None):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_layer = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_layer = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = MLP(dim, dim*2, dim)

    def forward(self, q, kv, q_mask=None, kv_mask=None):
        B, N, C = q.shape
        
        q_vec = self.q_layer(self.norm1(q)).reshape(B, N, self.num_heads, C // self.num_heads)
        kv_vec = self.kv_layer(self.norm1(kv)).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        k_vec, v_vec = kv_vec[0].contiguous(), kv_vec[1].contiguous()
        k_vec = F.elu(k_vec) + 1.0
        q_vec = F.elu(q_vec) + 1.0

        if q_mask is not None:
            q_vec = q_vec * q_mask[:, :, None, None]
        if kv_mask is not None:
            k_vec = k_vec * kv_mask[:, :, None, None]
            v_vec = v_vec * kv_mask[:, :, None, None]
        
        with torch.cuda.amp.autocast(enabled=False):
            v_length = v_vec.size(1)
            v_vec = v_vec / v_length
            kv = torch.einsum("nlhd,nlhm->nhmd", k_vec, v_vec)  # [B,nhead,C2,C2]
            # Compute the normalizer
            z = 1 / (torch.einsum("nlhd,nhd->nlh", q_vec, k_vec.sum(dim=1)) + 1e-6)
            # Finally compute and return the new values
            x = torch.einsum("nlhd,nhmd,nlh->nlhm", q_vec, kv, z) * v_length  # [B,N,nhead,C2]

        x = x.reshape(B, -1, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        x = q + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))

        return x
    

class TopicFormer(nn.Module):

    def __init__(self, dim, nhead, pool_layers=["seed"]*5, n_merge_layers=1, n_topics=100, n_samples=8, topic_dim=None, attn_type="vanilla"):
        super(TopicFormer, self).__init__()

        # self.config = config
        self.d_model = dim
        self.nhead = nhead
        self.topic_dim = topic_dim
        self.n_topics = n_topics
        self.n_samples = n_samples

        if attn_type == "vanilla":
            encoder_layer = AttentionLayer(self.d_model, self.nhead, attn_drop=0.0)
        elif attn_type == "linear":
            encoder_layer = LinearAttentionLayer(self.d_model, self.nhead, attn_drop=0.0)
        else:
            raise f"unknown {attn_type} attention"
        
        if n_merge_layers > 0:
            self.feat_aug = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(2*n_merge_layers)])
        self.n_iter_topic_transformer = n_merge_layers

        if topic_dim is None:
            self.layer_names = pool_layers
            self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
            self.seed_tokens = nn.Parameter(torch.randn(n_topics, dim))
            self.register_parameter('seed_tokens', self.seed_tokens)
            self.topic_drop = nn.Dropout1d(p=0.1)
        else:
            # self.seed_tokens = topic_init
            self.emb_layer = nn.Linear(topic_dim, self.d_model, bias=False) if n_merge_layers > 0 else nn.Identity()
        
        self.norm = nn.LayerNorm(self.d_model) # nn.Tanh()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sample_topic(self, prob_topics, topics, L):
        prob_topics0, prob_topics1 = prob_topics[:, :L], prob_topics[:, L:]
        topics0, topics1  = topics[:, :L], topics[:, L:]

        theta0 = F.normalize(prob_topics0.sum(dim=1), p=1, dim=-1) # [N, K]
        theta1 = F.normalize(prob_topics1.sum(dim=1), p=1, dim=-1)
        theta = F.normalize(theta0 * theta1, p=1, dim=-1)
        if self.n_samples == 0:
            return None
        if self.training:
            sampled_inds = torch.multinomial(theta, self.n_samples)
            sampled_values = torch.gather(theta, dim=-1, index=sampled_inds)
        else:
            sampled_values, sampled_inds = torch.topk(theta, self.n_samples, dim=-1)
        sampled_topics0 = torch.gather(topics0, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics0.shape[1], 1))
        sampled_topics1 = torch.gather(topics1, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics1.shape[1], 1))
        return sampled_topics0, sampled_topics1

    def reduce_feat(self, feat, topick, N, C):
        len_topic = topick.sum(dim=-1).int()
        max_len = len_topic.max().item()
        selected_ids = topick.bool()
        resized_feat = torch.zeros((N, max_len, C), dtype=torch.float32, device=feat.device)
        new_mask = torch.zeros_like(resized_feat[..., 0]).bool()
        for i in range(N):
            new_mask[i, :len_topic[i]] = True
        resized_feat[new_mask, :] = feat[selected_ids, :]
        return resized_feat, new_mask, selected_ids

    def forward(self, feat0, feat1, topic_init=None):

        assert self.d_model == feat0.shape[2], "the feature number of src and transformer must be equal"
        N, L, S, C, K = feat0.shape[0], feat0.shape[1], feat1.shape[1], feat0.shape[2], self.n_topics

        feat = torch.cat((feat0, feat1), dim=1)

        if topic_init is None:
            seeds = self.seed_tokens.unsqueeze(0).repeat(N, 1, 1)
            seeds = self.topic_drop(seeds)

            for layer, name in zip(self.layers, self.layer_names):
                seeds = layer(seeds, feat)
        else:
            seeds = self.emb_layer(topic_init)

        # dmatrix = torch.einsum("nmd,nkd->nmk", self.norm(feat), self.norm(seeds) / C**.5)
        # prob_topics = F.softmax(dmatrix, dim=-1)

        # feat_topics = torch.zeros_like(dmatrix).scatter_(-1, torch.argmax(dmatrix, dim=-1, keepdim=True), 1.0)

        # sampled_topics = self.sample_topic(prob_topics.detach(), feat_topics, L)
        # if sampled_topics is not None:
        #     updated_feat0, updated_feat1 = torch.zeros_like(feat0), torch.zeros_like(feat1)
        #     s_topics0, s_topics1 = sampled_topics
        #     for k in range(s_topics0.shape[-1]):
        #         topick0, topick1 = s_topics0[..., k], s_topics1[..., k] # [N, L+S]
        #         if (topick0.sum() > 0) and (topick1.sum() > 0):
        #             new_feat0, new_mask0, selected_ids0 = self.reduce_feat(feat0, topick0, N, C)
        #             new_feat1, new_mask1, selected_ids1 = self.reduce_feat(feat1, topick1, N, C)
        #             for idt in range(self.n_iter_topic_transformer):
        #                 new_feat0 = self.feat_aug[idt*2](new_feat0, new_feat0, new_mask0, new_mask0)
        #                 new_feat1 = self.feat_aug[idt*2](new_feat1, new_feat1, new_mask1, new_mask1)
        #                 new_feat0 = self.feat_aug[idt*2+1](new_feat0, new_feat1, new_mask0, new_mask1)
        #                 new_feat1 = self.feat_aug[idt*2+1](new_feat1, new_feat0, new_mask1, new_mask0)
        #             updated_feat0[selected_ids0, :] = new_feat0[new_mask0, :]
        #             updated_feat1[selected_ids1, :] = new_feat1[new_mask1, :]

        #     feat0 = (1 - s_topics0.sum(dim=-1, keepdim=True)) * feat0 + updated_feat0
        #     feat1 = (1 - s_topics1.sum(dim=-1, keepdim=True)) * feat1 + updated_feat1
        # else:
        for idt in range(self.n_iter_topic_transformer * 2):
            feat0 = self.feat_aug[idt](feat0, seeds)
            feat1 = self.feat_aug[idt](feat1, seeds)

        # if self.training and topic_init is None:
        #     topic_matrix = torch.einsum("nlk,nsk->nls", prob_topics[:, :L], prob_topics[:, L:])
        # else:
        #     topic_matrix = {"img0": feat_topics[:, :L], "img1": feat_topics[:, L:]}

        return self.norm(feat0), self.norm(feat1), seeds #, (prob_topics[:, :L], prob_topics[:, L:])