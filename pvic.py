"""
Two-stage HOI detector with enhanced visual context

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""
#CLIPViC
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist

import clip

from torch import nn, Tensor
from collections import OrderedDict
from typing import Optional, Tuple, List
from torchvision.ops import FeaturePyramidNetwork

from transformers import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,TransformerDecoderLayer2,
    SwinTransformer,
    ClipTransformerDecoderLayer,ClipTransformerDecoder
)

from ops import (
    binary_focal_loss_with_logits,
    compute_spatial_encodings,
    prepare_region_proposals,
    associate_with_ground_truth,
    compute_prior_scores,
    compute_sinusoidal_pe
)

from detr.models import build_model as build_base_detr
from h_detr.models import build_model as build_advanced_detr
from detr.models.position_encoding import PositionEmbeddingSine,ClipPositionEmbeddingSine
from detr.util.misc import NestedTensor, nested_tensor_from_tensor_list


class MultiModalFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size):
        super().__init__()
        self.fc1 = nn.Linear(fst_mod_size, repr_size)
        self.fc2 = nn.Linear(scd_mod_size, repr_size)
        self.ln1 = nn.LayerNorm(repr_size)
        self.ln2 = nn.LayerNorm(repr_size)

        mlp = []
        repr_size = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size[:-1], repr_size[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.ln1(self.fc1(x))
        y = self.ln2(self.fc2(y))
        z = F.relu(torch.cat([x, y], dim=-1))
        z = self.mlp(z)
        return z

class HumanObjectMatcher(nn.Module):
    # def __init__(self, repr_size, num_verbs, obj_to_verb, zs_obj_to_verb, dropout=.1, human_idx=0):
    # def __init__(self, repr_size, num_verbs, obj_to_verb, object_to_interaction, dropout=.1, human_idx=0):
    def __init__(self, repr_size, num_verbs, obj_to_verb, dropout=.1, human_idx=0):
        super().__init__()
        self.repr_size = repr_size
        self.num_verbs = num_verbs
        self.human_idx = human_idx
        self.obj_to_verb = obj_to_verb

        # self.zs_obj_to_verb = zs_obj_to_verb
        # self.object_to_interaction = object_to_interaction

        self.ref_anchor_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, repr_size), nn.ReLU(),
        )
        self.encoder = TransformerEncoder(num_layers=2, dropout=dropout)
        self.mmf = MultiModalFusion(512, repr_size, repr_size)

    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        n_h = torch.sum(is_human)
        if not torch.all(labels[:n_h]==self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return n_h

    def compute_box_pe(self, boxes, embeds, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]

        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)

        # Modulate the positional embeddings with box widths and heights by
        # applying different temperatures to x and y
        ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()    # n_query, 2
        # Note that the positional embeddings are stacked as [pe(y), pe(x)]
        c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
        c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)

        return box_pe, c_pe

    def forward(self, region_props, image_sizes, device=None):
        if device is None:
            device = region_props[0]["hidden_states"].device

        # if self.training:
        #     obj_2_verb = self.zs_obj_to_verb
        # else:
        #     obj_2_verb = self.obj_to_verb

        ho_queries = []
        paired_indices = []
        prior_scores = []
        object_types = []
        positional_embeds = []
        for i, rp in enumerate(region_props):
            boxes, scores, labels, embeds = rp.values()
            nh = self.check_human_instances(labels)
            n = len(boxes)
            # Enumerate instance pairs
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < nh)).unbind(1)
            # Skip image when there are no valid human-object pairs
            if len(x_keep) == 0:
                ho_queries.append(torch.zeros(0, self.repr_size, device=device))
                paired_indices.append(torch.zeros(0, 2, device=device, dtype=torch.int64))
                prior_scores.append(torch.zeros(0, 2, self.num_verbs, device=device))
                # prior_scores.append(torch.zeros(0, 2, 600, device=device))
                object_types.append(torch.zeros(0, device=device, dtype=torch.int64))
                positional_embeds.append({})
                continue
            x = x.flatten(); y = y.flatten()
            # Compute spatial features
            pairwise_spatial = compute_spatial_encodings(
                [boxes[x],], [boxes[y],], [image_sizes[i],]
            )
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)

            box_pe, c_pe = self.compute_box_pe(boxes, embeds, image_sizes[i])
            embeds, _ = self.encoder(embeds.unsqueeze(1), box_pe.unsqueeze(1))
            embeds = embeds.squeeze(1)
            # Compute human-object queries
            ho_q = self.mmf(
                torch.cat([embeds[x_keep], embeds[y_keep]], dim=1),
                pairwise_spatial_reshaped[x_keep, y_keep]
            )
            # Append matched human-object pairs
            ho_queries.append(ho_q)
            paired_indices.append(torch.stack([x_keep, y_keep], dim=1))
            prior_scores.append(compute_prior_scores(
                x_keep, y_keep, scores, labels, self.num_verbs, self.training,
                self.obj_to_verb#物体对应的动作
                # self.object_to_interaction
                # obj_2_verb
            ))
            object_types.append(labels[y_keep])
            positional_embeds.append({
                "centre": torch.cat([c_pe[x_keep], c_pe[y_keep]], dim=-1).unsqueeze(1),#[15,1,512]
                "box": torch.cat([box_pe[x_keep], box_pe[y_keep]], dim=-1).unsqueeze(1)#[15,1,1024]
            })

        return ho_queries, paired_indices, prior_scores, object_types, positional_embeds

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)

class FeatureHead(nn.Module):
    def __init__(self, dim, dim_backbone, return_layer, num_layers):
        super().__init__()
        self.dim = dim
        self.dim_backbone = dim_backbone
        self.return_layer = return_layer

        in_channel_list = [
            int(dim_backbone * 2 ** i)
            for i in range(return_layer + 1, 1)
        ]
        self.fpn = FeaturePyramidNetwork(in_channel_list, dim)

        # self.sge = SpatialGroupEnhance(groups=8)

        self.layers = nn.Sequential(
            Permute([0, 2, 3, 1]),
            SwinTransformer(dim, num_layers)
        )
        # pass

    def forward(self, x):
        pyramid = OrderedDict(
            (f"{i}", x[i].tensors)
            for i in range(self.return_layer, 0)
        )
        mask = x[self.return_layer].mask
        x = self.fpn(pyramid)[f"{self.return_layer}"]

        # x = self.sge(x)

        x = self.layers(x)
        return x, mask

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class PViC(nn.Module):
    """Two-stage HOI detector with enhanced visual context"""

    def __init__(self,
        detector: Tuple[nn.Module, str], postprocessor: nn.Module,
        feature_head: nn.Module, ho_matcher: nn.Module,
        triplet_decoder: nn.Module, clip_decoder: nn.Module, num_verbs: int,
        repr_size: int = 384, human_idx: int = 0,
        # Focal loss hyper-parameters
        alpha: float = 0.5, gamma: float = .1,
        # Sampling hyper-parameters
        box_score_thresh: float = .05,
        min_instances: int = 3,
        max_instances: int = 15,
        raw_lambda: float = 2.8,
    ) -> None:
        super().__init__()

        self.detector = detector[0]
        self.od_forward = {
            "base": self.base_forward,
            "advanced": self.advanced_forward,
        }[detector[1]]
        self.postprocessor = postprocessor

        self.ho_matcher = ho_matcher
        self.feature_head = feature_head
        self.kv_pe = PositionEmbeddingSine(128, 20, normalize=True)

        self.pes = ClipPositionEmbeddingSine(128, 20, normalize=True)
        self.clip_decoder = clip_decoder
        self.fcc = nn.Linear(384, 117)

        # self.fcc = nn.Linear(384, 600)
        # self.binary_classifier = nn.Linear(repr_size, 600)

        # self.fcc = nn.Linear(384, 24)
        # self.clip, _ = clip.load("ViT-B/32")
        self.clip, _ = clip.load("ViT-B/16")
        # self.clip, _ = clip.load("ViT-L/14")
        # self.clip, _ = clip.load("ViT-L/14@336px")
        # self.binary_classifier = nn.Linear(repr_size*2, num_verbs)

        # self.clip_swin = SwinTransformer(256, 1)  # x:Input (B, H, W, C).Output (B, H, W, C).
        # self.fc768_256 = nn.Linear(768,256)

        # self.clip_act = torch.load('./hico117.pth').float().to("cuda")#117,512
        # self.clip_act /= self.clip_act.norm(dim=-1,keepdim=True)

        # self.clip_act = torch.load('./hoi600.pth').float().to("cuda")#600,512
        # self.clip_act /= self.clip_act.norm(dim=-1,keepdim=True)
        # self.fc384_512 = nn.Linear(384,512)
        # self.fc384_5122 = nn.Linear(384,512)

        # self.emb = nn.Embedding(117, 384)
        # self.emb /= self.emb.weight.norm(dim=-1, keepdim=True)
        # self.emb2 = nn.Embedding(117, 384)
        # self.emb2 /= self.emb2.weight.norm(dim=-1, keepdim=True)

        self.decoder = triplet_decoder
        self.binary_classifier = nn.Linear(repr_size, num_verbs)

        self.repr_size = repr_size
        self.human_idx = human_idx
        self.num_verbs = num_verbs
        self.alpha = alpha
        self.gamma = gamma
        self.box_score_thresh = box_score_thresh
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.raw_lambda = raw_lambda

    def freeze_detector(self):
        for p in self.detector.parameters():
            p.requires_grad = False
        for p in self.clip.parameters():
            p.requires_grad = False

    def compute_classification_loss(self, logits, prior, labels):
        prior = torch.cat(prior, dim=0).prod(1)
        x, y = torch.nonzero(prior).unbind(1)

        logits = logits[:, x, y]
        prior = prior[x, y]
        labels = labels[None, x, y].repeat(len(logits), 1)

        n_p = labels.sum()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )

        return loss / n_p

    def postprocessing(self,
            boxes, paired_inds, object_types,
            logits, prior, image_sizes
        ):
        n = [len(p_inds) for p_inds in paired_inds]
        logits = logits.split(n)

        # logits = logits[-1].split(n)
        # logits0 = logits[0].split(n)

        detections = []
        for bx, p_inds, objs, lg, pr, size, in zip(
            boxes, paired_inds, object_types,
            logits, prior, image_sizes#, logits0
        ):
            pr = pr.prod(1)
            x, y = torch.nonzero(pr).unbind(1)
            scores = lg[x, y].sigmoid() * pr[x, y].pow(self.raw_lambda)
            # scores = (lg[x, y].sigmoid() + lg0[x, y].sigmoid()) * pr[x, y].pow(self.raw_lambda)
            detections.append(dict(
                boxes=bx, pairing=p_inds[x], scores=scores,
                labels=y, objects=objs[x], size=size, x=x
            ))

        return detections

    @staticmethod
    def base_forward(ctx, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = ctx.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = ctx.transformer(ctx.input_proj(src), mask, ctx.query_embed.weight, pos[-1])[0]
        # hs,encode = ctx.transformer(ctx.input_proj(src), mask, ctx.query_embed.weight, pos[-1])#encode:[bs, c, h, w]

        outputs_class = ctx.class_embed(hs)
        outputs_coord = ctx.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out, hs, features#, encode

    @staticmethod
    def advanced_forward(ctx, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = ctx.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(ctx.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if ctx.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, ctx.num_feature_levels):
                if l == _len_srcs:
                    src = ctx.input_proj[l](features[-1].tensors)
                else:
                    src = ctx.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = ctx.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not ctx.two_stage or ctx.mixed_selection:
            query_embeds = ctx.query_embed.weight[0 : ctx.num_queries, :]

        self_attn_mask = (
            torch.zeros([ctx.num_queries, ctx.num_queries,]).bool().to(src.device)
        )
        self_attn_mask[ctx.num_queries_one2one :, 0 : ctx.num_queries_one2one,] = True
        self_attn_mask[0 : ctx.num_queries_one2one, ctx.num_queries_one2one :,] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = ctx.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = ctx.class_embed[lvl](hs[lvl])
            tmp = ctx.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(outputs_class[:, 0 : ctx.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, ctx.num_queries_one2one :])
            outputs_coords_one2one.append(outputs_coord[:, 0 : ctx.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, ctx.num_queries_one2one :])
        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }

        if ctx.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out, hs, features

    def forward(self,
        images: List[Tensor],
        clipimgs: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (M, 2) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `size`: torch.Tensor
                (2,) Image height and width
            `x`: torch.Tensor
                (M,) Index tensor corresponding to the duplications of human-objet pairs. Each
                pair was duplicated once for each valid action.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        image_sizes = torch.as_tensor([im.size()[-2:] for im in images], device=images[0].device)

        with torch.no_grad():
            results, hs, features = self.od_forward(self.detector, images)
            # results, hs, features, encode = self.od_forward(self.detector, images)
            results = self.postprocessor(results, image_sizes)

            clipimg = torch.stack(clipimgs)#[b,3,224,224]

            contextual = self.clip.encode_image(clipimg)[1].float()

            # cb = clipimg.shape[0]
            # cont_shape = contextual[:, 1:, :].reshape(cb, 14, 14, 768)
        # cont = self.fc768_256(cont_shape)
        # cont = self.clip_swin(cont).reshape(cb, 196, 256)

        region_props = prepare_region_proposals(
            results, hs[-1], image_sizes,
            box_score_thresh=self.box_score_thresh,
            human_idx=self.human_idx,
            min_instances=self.min_instances,
            max_instances=self.max_instances
        )
        boxes = [r['boxes'] for r in region_props]
        # Produce human-object pairs.
        (
            ho_queries,
            paired_inds, prior_scores,
            object_types, positional_embeds
        ) = self.ho_matcher(region_props, image_sizes)
        # Compute keys/values for triplet decoder.
        memory, mask = self.feature_head(features)
        b, h, w, c = memory.shape

        # encode = encode.permute(0, 2, 3, 1).reshape(b, h * w, c)

        memory = memory.reshape(b, h * w, c)

        # memory = encode+memory

        kv_p_m = mask.reshape(-1, 1, h * w)
        k_pos = self.kv_pe(NestedTensor(memory, mask)).permute(0, 2, 3, 1).reshape(b, h * w, 1, c)
        # clip_pos = self.pes(torch.ones((1, 7, 7), dtype=torch.bool, device="cuda")).permute(0, 2, 3, 1).reshape(1, 49, 1, 256)
        clip_pos = self.pes(torch.ones((1, 14, 14), dtype=torch.bool, device="cuda")).permute(0, 2, 3, 1).reshape(1,196,1,256)
        # clip_pos = self.pes(torch.ones((1, 16, 16), dtype=torch.bool, device="cuda")).permute(0, 2, 3, 1).reshape(1,256,1,256)
        # clip_pos = self.pes(torch.ones((1, 24, 24), dtype=torch.bool, device="cuda")).permute(0, 2, 3, 1).reshape(1,576, 1, 256)
        # Enhance visual context with triplet decoder.
        query_embeds = []
        for i, (ho_q, mem, con) in enumerate(zip(ho_queries, memory, contextual)):
        # for i, (ho_q, mem, con) in enumerate(zip(ho_queries, memory, cont)):
        # for i, (ho_q, mem) in enumerate(zip(ho_queries, memory)):
        # for i, (ho_q, con) in enumerate(zip(ho_queries, contextual)):
        #     query_embeds.append(self.decoder(
        #         ho_q.unsqueeze(1),              # (n, 1, q_dim)
        #         mem.unsqueeze(1),               # (hw, 1, kv_dim)
        #         kv_padding_mask=kv_p_m[i],      # (1, hw)
        #         q_pos=positional_embeds[i],     # centre: (n, 1, 2*kv_dim), box: (n, 1, 4*kv_dim)
        #         k_pos=k_pos[i]                  # (hw, 1, kv_dim)
        #     ).squeeze(dim=2))

            cc = self.clip_decoder(
                ho_q.unsqueeze(1),  # (n, 1, q_dim)
                con[1:].unsqueeze(1),  # (hw, 1, kv_dim) 50,1,768
                # con.unsqueeze(1),
                q_pos=positional_embeds[i],  # centre: (n, 1, 2*kv_dim), box: (n, 1, 4*kv_dim)
                k_pos=clip_pos[0]  # (hw, 1, kv_dim)
            ).squeeze(dim=2)  # .repeat(2, 1, 1)#.squeeze(dim=2)  # [2,435/15,1,384]

            toke = self.decoder(
                ho_q.unsqueeze(1),  # (n, 1, q_dim)
                mem.unsqueeze(1), #con[1:].unsqueeze(1),  # (hw, 1, kv_dim)
                # mem.unsqueeze(1), con.unsqueeze(1),
                kv_padding_mask=kv_p_m[i],  # (1, hw)
                q_pos=positional_embeds[i],  # centre: (n, 1, 2*kv_dim), box: (n, 1, 4*kv_dim)
                k_pos=k_pos[i],  # (hw, 1, kv_dim)
                #clip_k_pos=clip_pos[0]
            ).squeeze(dim=2)

            # query_embeds.append(torch.cat([cc, toke[-1:]], dim=-1))
            query_embeds.append(torch.cat([cc, toke], dim=0))
            # query_embeds.append(cc)
            # query_embeds.append(toke)
        # Concatenate queries from all images in the same batch.
        query_embeds = torch.cat(query_embeds, dim=1)   # (ndec, \sigma{n}, q_dim)

        sim = self.fcc(query_embeds[:1])
        logits = self.binary_classifier(query_embeds[1:])  # [2,450,117]
        logits = torch.cat([sim, logits], dim=0)


        # sim = torch.matmul(query_embeds[:1], self.emb.weight.T.unsqueeze(0))
        # logits = torch.matmul(query_embeds[1:], self.emb2.weight.T.unsqueeze(0))
        # logits = torch.cat([sim, logits], dim=0)
        # sim = self.fcc(query_embeds[:2])
        # logits = self.binary_classifier(query_embeds[2:])  # [2,450,117]
        # logits = torch.cat([sim, logits], dim=0)

        # sim2 = self.fc384_512(query_embeds[:1])
        # sims2 = torch.matmul(sim2, self.clip_act.T.unsqueeze(0))
        # # if self.clip_act.requires_grad:
        # #     print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # logits2 = self.fc384_5122(query_embeds[1:])
        # logits2 = torch.matmul(logits2, self.clip_act.T.unsqueeze(0))
        # logits = torch.cat([sims2, logits2], dim=0)

        # logits = self.binary_classifier(query_embeds)

        if self.training:
            labels = associate_with_ground_truth(
                boxes, paired_inds, targets, self.num_verbs
            )
            cls_loss = self.compute_classification_loss(logits, prior_scores, labels)
            loss_dict = dict(cls_loss=cls_loss)
            return loss_dict

        detections = self.postprocessing(
            boxes, paired_inds, object_types,
            # logits[-1], prior_scores, image_sizes
            # logits[0], prior_scores, image_sizes
            (logits[-1] + logits[0]) / 2, prior_scores, image_sizes
            # (logits[-1]+ logits[3] + logits[2] + logits[0]) / 4, prior_scores, image_sizes
            # (logits[-1] + logits[1]) / 2, prior_scores, image_sizes
            # logits, prior_scores, image_sizes
        )

        # torch.save(detections, 'dets.pth')


        return detections

# def build_detector(args, obj_to_verb, zs_obj_to_verb):
# def build_detector(args, obj_to_verb, object_to_interaction):
def build_detector(args, obj_to_verb):
    if args.detector == "base":
        detr, _, postprocessors = build_base_detr(args)
    elif args.detector == "advanced":
        detr, _, postprocessors = build_advanced_detr(args)

    if os.path.exists(args.pretrained):
        if dist.is_initialized():
            print(f"Rank {dist.get_rank()}: Load weights for the object detector from {args.pretrained}")
        else:
            print(f"Load weights for the object detector from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])

    ho_matcher = HumanObjectMatcher(
        repr_size=args.repr_dim,
        num_verbs=args.num_verbs,
        obj_to_verb=obj_to_verb, #object_to_interaction = object_to_interaction,#zs_obj_to_verb =zs_obj_to_verb,
        dropout=args.dropout
    )
    decoder_layer = TransformerDecoderLayer(
        q_dim=args.repr_dim, kv_dim=args.hidden_dim,
        ffn_interm_dim=args.repr_dim * 4,
        num_heads=args.nheads, dropout=args.dropout
    )
    decoder_layer2 = TransformerDecoderLayer2(
        q_dim=args.repr_dim, kv_dim=args.hidden_dim,
        ffn_interm_dim=args.repr_dim * 4,
        num_heads=args.nheads, dropout=args.dropout
    )
    clip_decoder_layer = ClipTransformerDecoderLayer(
        q_dim=args.repr_dim, kv_dim=args.hidden_dim,
        ffn_interm_dim=args.repr_dim * 4,
        num_heads=args.nheads, dropout=args.dropout
    )
    triplet_decoder = TransformerDecoder(
        decoder_layer=decoder_layer,decoder_layer2=decoder_layer2,
        num_layers=args.triplet_dec_layers
    )
    clip_decoder = ClipTransformerDecoder(
        decoder_layer=clip_decoder_layer,
        # num_layers=2
        num_layers = 1
    )
    return_layer = {"C5": -1, "C4": -2, "C3": -3}[args.kv_src]
    if isinstance(detr.backbone.num_channels, list):
        num_channels = detr.backbone.num_channels[-1]
    else:
        num_channels = detr.backbone.num_channels
    feature_head = FeatureHead(
        args.hidden_dim, num_channels,
        return_layer, args.triplet_enc_layers
    )
    model = PViC(
        (detr, args.detector), postprocessors['bbox'],
        feature_head=feature_head,
        ho_matcher=ho_matcher,
        triplet_decoder=triplet_decoder, clip_decoder=clip_decoder,
        num_verbs=args.num_verbs,
        repr_size=args.repr_dim,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        raw_lambda=args.raw_lambda,
    )
    return model
