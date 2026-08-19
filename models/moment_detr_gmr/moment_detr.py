"""
DETR model and criterion classes.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.moment_detr_gmr.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx
from models.moment_detr_gmr.position_encoding import build_position_encoding
from models.moment_detr_gmr.matcher import build_matcher
from models.moment_detr_gmr.misc import accuracy
from models.moment_detr_gmr.moment_transformer import build_transformer
from models.moment_detr_gmr.gmr_adapter import GMRAdapter, compute_existence_loss
from experiments.temporal_cgp import TemporalCGP
from experiments.vmr_cgp import DETRQueryCGP, VMRCGP

class MomentDETR(nn.Module):
    """ This is the Moment-DETR module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False, max_v_l=75, span_loss_type="l1",
                 use_txt_pos=False, n_input_proj=2, aud_dim=0, use_exist_head=False, exist_pool="max",
                 use_tcgp=False, tcgp_num_basis=16, tcgp_prompt_length=1,
                 tcgp_router_hidden_dim=256, tcgp_temperature=1.0, tcgp_alpha_init=0.1,
                 use_vmr_cgp=False, vmr_cgp_num_basis=16, vmr_cgp_prompt_length=4,
                 vmr_cgp_router_hidden_dim=256, vmr_cgp_temperature=1.0,
                 vmr_cgp_alpha_init=0.01, vmr_cgp_alpha_trainable=True,
                 vmr_cgp_gate_floor=0.0,
                 use_query_cgp=False, query_cgp_num_basis=16,
                 query_cgp_prompt_length=6, query_cgp_router_hidden_dim=256,
                 query_cgp_frf_hidden_dim=512, query_cgp_temperature=1.0,
                 query_cgp_beta=0.05):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Moment-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.aux_loss = aux_loss

        # Optional GMR Adapter for query-video existence prediction.
        self.use_exist_head = bool(use_exist_head)
        self.exist_pool = str(exist_pool)
        if self.use_exist_head:
            self.exist_head = GMRAdapter(hidden_dim, hidden_dim, pool=self.exist_pool)
        else:
            self.exist_head = None

        self.use_tcgp = bool(use_tcgp)
        if self.use_tcgp:
            self.tcgp = TemporalCGP(
                hidden_dim=hidden_dim,
                num_basis=int(tcgp_num_basis),
                prompt_length=int(tcgp_prompt_length),
                router_hidden_dim=int(tcgp_router_hidden_dim),
                temperature=float(tcgp_temperature),
                alpha_init=float(tcgp_alpha_init),
            )
        else:
            self.tcgp = None

        self.use_vmr_cgp = bool(use_vmr_cgp)
        if self.use_tcgp and self.use_vmr_cgp:
            raise ValueError("use_tcgp and use_vmr_cgp are mutually exclusive")
        if self.use_vmr_cgp:
            self.vmr_cgp = VMRCGP(
                hidden_dim=hidden_dim,
                num_basis=int(vmr_cgp_num_basis),
                prompt_length=int(vmr_cgp_prompt_length),
                router_hidden_dim=int(vmr_cgp_router_hidden_dim),
                temperature=float(vmr_cgp_temperature),
                alpha_init=float(vmr_cgp_alpha_init),
                alpha_trainable=bool(vmr_cgp_alpha_trainable),
                gate_floor=float(vmr_cgp_gate_floor),
            )
        else:
            self.vmr_cgp = None

        self.use_query_cgp = bool(use_query_cgp)
        enabled_cgp_modules = sum(
            int(enabled)
            for enabled in (self.use_tcgp, self.use_vmr_cgp, self.use_query_cgp)
        )
        if enabled_cgp_modules > 1:
            raise ValueError("T-CGP, token VMR-CGP, and DETR-Query CGP are mutually exclusive")
        if self.use_query_cgp:
            if self.transformer.decoder.num_layers < 2:
                raise ValueError("DETR-Query CGP requires at least two decoder layers")
            self.query_cgp = DETRQueryCGP(
                hidden_dim=hidden_dim,
                num_basis=int(query_cgp_num_basis),
                prompt_length=int(query_cgp_prompt_length),
                router_hidden_dim=int(query_cgp_router_hidden_dim),
                frf_hidden_dim=int(query_cgp_frf_hidden_dim),
                temperature=float(query_cgp_temperature),
                beta=float(query_cgp_beta),
            )
        else:
            self.query_cgp = None

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask,
                src_aud=None, src_aud_mask=None, src_txt_semantic_mask=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)

        query_semantic = None
        if self.query_cgp is not None:
            semantic_mask = (
                src_txt_mask.bool()
                if src_txt_semantic_mask is None
                else src_txt_semantic_mask.bool() & src_txt_mask.bool()
            )
            semantic_count = semantic_mask.sum(dim=1, keepdim=True)
            if bool((semantic_count == 0).any()):
                raise ValueError("DETR-Query CGP received a query with no valid semantic tokens")
            semantic_weights = semantic_mask.to(src_txt.dtype).unsqueeze(-1)
            query_semantic = (src_txt * semantic_weights).sum(dim=1)
            query_semantic = query_semantic / semantic_count.to(src_txt.dtype)

        vmr_cgp_output = None
        if self.vmr_cgp is not None:
            vmr_cgp_output = self.vmr_cgp(
                src_vid, src_vid_mask, src_txt, src_txt_mask
            )
            src_txt = vmr_cgp_output.enhanced_text
        if self.tcgp is not None:
            tcgp_output = self.tcgp(src_vid, src_vid_mask, src_txt, src_txt_mask)
            adapted_token = tcgp_output.adapted_query.unsqueeze(1)
            src_txt = torch.cat([src_txt, adapted_token], dim=1)
            adapted_mask = src_txt_mask.new_ones((src_txt_mask.shape[0], 1))
            src_txt_mask = torch.cat([src_txt_mask, adapted_mask], dim=1)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        if self.query_cgp is None:
            hs, memory = self.transformer(src, ~mask, self.query_embed.weight, pos)
        else:
            self.query_cgp.clear_diagnostics()
            hs, memory = self.transformer(
                src,
                ~mask,
                self.query_embed.weight,
                pos,
                decoder_interlayer_adapter=self.query_cgp,
                decoder_adapter_after_layer=0,
                decoder_adapter_kwargs={
                    "query_semantic": query_semantic,
                    "video_length": src_vid.shape[1],
                },
            )
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        outputs_coord = self.span_embed(hs)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        if vmr_cgp_output is not None:
            out["vmr_cgp_frame_logits"] = vmr_cgp_output.frame_logits
            out["vmr_cgp_basis_weights"] = vmr_cgp_output.basis_weights
            out["vmr_cgp_video_mask"] = src_vid_mask.bool()
            out["vmr_cgp_text_mask"] = src_txt_mask.bool()

        if self.query_cgp is not None and self.query_cgp.last_output is not None:
            query_cgp_output = self.query_cgp.last_output
            out["query_cgp_temporal_attention"] = query_cgp_output.temporal_attention
            out["query_cgp_basis_weights"] = query_cgp_output.basis_weights
            out["query_cgp_video_mask"] = src_vid_mask.bool()

        if self.exist_head is not None:
            out["pred_exist_logits"] = self.exist_head(hs[-1])

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)

        out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)  # (bsz, L_vid)

        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        return out

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, span_loss_type, max_v_l,
                 saliency_margin=1):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        # Empty GT batches contribute zero localization loss.
        if idx[0].numel() == 0:
            z = outputs["pred_spans"].sum() * 0.0
            return {"loss_span": z, "loss_giou": z}
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            if idx[0].numel() > 0:
                losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
        return {"loss_saliency": loss_saliency}

    def loss_exist(self, outputs, targets, indices=None, log=True):
        """Existence loss: whether this query-video pair contains any relevant moment.
        targets should contain key "exist_label": float tensor of shape (bsz,) with values in {0,1}.
        """
        return {"loss_exist": compute_existence_loss(outputs, targets)}

    def loss_vmr_cgp(self, outputs, targets, indices=None, log=True):
        """Directly supervise multi-evidence relevance and basis utilization."""
        del indices, log
        required = {
            "vmr_cgp_frame_logits",
            "vmr_cgp_basis_weights",
            "vmr_cgp_video_mask",
            "vmr_cgp_text_mask",
        }
        if targets is None or "span_labels" not in targets or not required.issubset(outputs):
            zero = outputs["pred_logits"].sum() * 0.0
            return {
                "loss_vmr_cgp_rel": zero,
                "loss_vmr_cgp_route": zero,
            }

        frame_logits = outputs["vmr_cgp_frame_logits"]
        video_mask = outputs["vmr_cgp_video_mask"].bool()
        sample_losses = []
        for batch_index, span_item in enumerate(targets["span_labels"]):
            valid_length = int(video_mask[batch_index].sum().item())
            if valid_length <= 0:
                continue
            logits = frame_logits[batch_index, :valid_length]
            centers = (
                torch.arange(valid_length, device=logits.device, dtype=logits.dtype) + 0.5
            ) / float(valid_length)
            spans = span_item["spans"]
            if spans.numel() == 0:
                union_target = torch.zeros_like(logits, dtype=torch.bool)
            else:
                span_xx = span_cxw_to_xx(spans).clamp(0.0, 1.0)
                union_target = (
                    (centers.unsqueeze(0) >= span_xx[:, :1])
                    & (centers.unsqueeze(0) <= span_xx[:, 1:])
                ).any(dim=0)

            positive_loss = F.softplus(-logits[union_target])
            negative_loss = F.softplus(logits[~union_target])
            if positive_loss.numel() and negative_loss.numel():
                sample_loss = 0.5 * (positive_loss.mean() + negative_loss.mean())
            elif positive_loss.numel():
                sample_loss = positive_loss.mean()
            else:
                sample_loss = negative_loss.mean()
            sample_losses.append(sample_loss)

        if sample_losses:
            relevance_loss = torch.stack(sample_losses).mean()
        else:
            relevance_loss = frame_logits.sum() * 0.0

        basis_weights = outputs["vmr_cgp_basis_weights"]
        text_mask = outputs["vmr_cgp_text_mask"].bool()
        valid_weights = basis_weights[text_mask]
        if valid_weights.numel() == 0:
            route_loss = basis_weights.sum() * 0.0
        else:
            eps = torch.finfo(valid_weights.dtype).eps
            marginal = valid_weights.mean(dim=0)
            target_usage = 1.0 / valid_weights.shape[-1]
            balance = valid_weights.shape[-1] * (
                (marginal - target_usage).square().sum()
            )
            conditional_entropy = -(
                valid_weights * valid_weights.clamp_min(eps).log()
            ).sum(dim=-1).mean() / math.log(valid_weights.shape[-1])
            route_loss = balance + 0.1 * conditional_entropy

        return {
            "loss_vmr_cgp_rel": relevance_loss,
            "loss_vmr_cgp_route": route_loss,
        }

    def loss_query_cgp(self, outputs, targets, indices=None, log=True):
        """Bind each matched DETR query to its own ground-truth temporal window."""
        del log
        required = {
            "query_cgp_temporal_attention",
            "query_cgp_basis_weights",
            "query_cgp_video_mask",
        }
        if (
            targets is None
            or "span_labels" not in targets
            or indices is None
            or not required.issubset(outputs)
        ):
            zero = outputs["pred_logits"].sum() * 0.0
            return {
                "loss_query_cgp_bind": zero,
                "loss_query_cgp_route": zero,
            }

        attention = outputs["query_cgp_temporal_attention"]
        basis_weights = outputs["query_cgp_basis_weights"]
        video_mask = outputs["query_cgp_video_mask"].bool()
        binding_terms = []
        matched_routes = []
        eps = torch.finfo(attention.dtype).eps

        for batch_index, (src_indices, target_indices) in enumerate(indices):
            if src_indices.numel() == 0:
                continue
            valid_length = int(video_mask[batch_index].sum().item())
            if valid_length <= 0:
                continue

            device = attention.device
            src_indices = src_indices.to(device)
            target_indices = target_indices.to(device)
            matched_attention = attention[batch_index, src_indices, :valid_length]
            target_spans = targets["span_labels"][batch_index]["spans"][target_indices]

            if self.span_loss_type == "l1":
                target_xx = span_cxw_to_xx(target_spans).clamp(0.0, 1.0)
                clip_starts = torch.arange(
                    valid_length, device=device, dtype=attention.dtype
                ) / float(valid_length)
                clip_ends = clip_starts + 1.0 / float(valid_length)
                overlap = (
                    (clip_starts.unsqueeze(0) < target_xx[:, 1:])
                    & (clip_ends.unsqueeze(0) > target_xx[:, :1])
                )
                empty_overlap = ~overlap.any(dim=1)
                if bool(empty_overlap.any()):
                    clip_centers = 0.5 * (clip_starts + clip_ends)
                    nearest = (
                        clip_centers.unsqueeze(0) - target_xx[:, :1]
                    ).abs().argmin(dim=1)
                    overlap[empty_overlap] = False
                    overlap[empty_overlap, nearest[empty_overlap]] = True
            else:
                clip_indices = torch.arange(valid_length, device=device).unsqueeze(0)
                overlap = (
                    (clip_indices >= target_spans[:, :1])
                    & (clip_indices <= target_spans[:, 1:])
                )

            target_mass = (matched_attention * overlap.to(attention.dtype)).sum(dim=1)
            binding_terms.append(-target_mass.clamp_min(eps).log())
            matched_routes.append(basis_weights[batch_index, src_indices])

        if binding_terms:
            binding_loss = torch.cat(binding_terms).mean()
            routes = torch.cat(matched_routes, dim=0)
            route_eps = torch.finfo(routes.dtype).eps
            conditional_entropy = -(
                routes * routes.clamp_min(route_eps).log()
            ).sum(dim=-1).mean()
            marginal = routes.mean(dim=0)
            marginal_entropy = -(
                marginal * marginal.clamp_min(route_eps).log()
            ).sum()
            route_loss = conditional_entropy - marginal_entropy
        else:
            binding_loss = attention.sum() * 0.0
            route_loss = basis_weights.sum() * 0.0

        return {
            "loss_query_cgp_bind": binding_loss,
            "loss_query_cgp_route": route_loss,
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "exist": self.loss_exist,
            "vmr_cgp": self.loss_vmr_cgp,
            "query_cgp": self.loss_query_cgp,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Match predictions to ground-truth windows.
        indices = self.matcher(outputs_without_aux, targets)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in {"saliency", "vmr_cgp", "query_cgp"}:  # only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)

def build_model(args):
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = MomentDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        aud_dim=args.a_feat_dim if "a_feat_dim" in args else 0,
        aux_loss=args.aux_loss,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        span_loss_type=args.span_loss_type,
        n_input_proj=args.n_input_proj,
        use_exist_head=bool(getattr(args, "use_exist_head", False)),
        exist_pool=str(getattr(args, "exist_pool", "max")),
        use_tcgp=bool(getattr(args, "use_tcgp", False)),
        tcgp_num_basis=int(getattr(args, "tcgp_num_basis", 16)),
        tcgp_prompt_length=int(getattr(args, "tcgp_prompt_length", 1)),
        tcgp_router_hidden_dim=int(getattr(args, "tcgp_router_hidden_dim", args.hidden_dim)),
        tcgp_temperature=float(getattr(args, "tcgp_temperature", 1.0)),
        tcgp_alpha_init=float(getattr(args, "tcgp_alpha_init", 0.1)),
        use_vmr_cgp=bool(getattr(args, "use_vmr_cgp", False)),
        vmr_cgp_num_basis=int(getattr(args, "vmr_cgp_num_basis", 16)),
        vmr_cgp_prompt_length=int(getattr(args, "vmr_cgp_prompt_length", 4)),
        vmr_cgp_router_hidden_dim=int(getattr(args, "vmr_cgp_router_hidden_dim", args.hidden_dim)),
        vmr_cgp_temperature=float(getattr(args, "vmr_cgp_temperature", 1.0)),
        vmr_cgp_alpha_init=float(getattr(args, "vmr_cgp_alpha_init", 0.01)),
        vmr_cgp_alpha_trainable=bool(getattr(args, "vmr_cgp_alpha_trainable", True)),
        vmr_cgp_gate_floor=float(getattr(args, "vmr_cgp_gate_floor", 0.0)),
        use_query_cgp=bool(getattr(args, "use_query_cgp", False)),
        query_cgp_num_basis=int(getattr(args, "query_cgp_num_basis", 16)),
        query_cgp_prompt_length=int(getattr(args, "query_cgp_prompt_length", 6)),
        query_cgp_router_hidden_dim=int(
            getattr(args, "query_cgp_router_hidden_dim", args.hidden_dim)
        ),
        query_cgp_frf_hidden_dim=int(
            getattr(args, "query_cgp_frf_hidden_dim", 2 * args.hidden_dim)
        ),
        query_cgp_temperature=float(getattr(args, "query_cgp_temperature", 1.0)),
        query_cgp_beta=float(getattr(args, "query_cgp_beta", 0.05)),
    )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']

    # Add existence supervision when the adapter is enabled.
    if bool(getattr(args, "use_exist_head", False)):
        weight_dict["loss_exist"] = float(getattr(args, "exist_loss_coef", 1.0))
        losses.append("exist")

    if bool(getattr(args, "use_vmr_cgp", False)):
        weight_dict["loss_vmr_cgp_rel"] = float(
            getattr(args, "vmr_cgp_temporal_loss_coef", 0.2)
        )
        weight_dict["loss_vmr_cgp_route"] = float(
            getattr(args, "vmr_cgp_route_loss_coef", 0.01)
        )
        losses.append("vmr_cgp")

    if bool(getattr(args, "use_query_cgp", False)):
        weight_dict["loss_query_cgp_bind"] = float(
            getattr(args, "query_cgp_binding_loss_coef", 0.2)
        )
        weight_dict["loss_query_cgp_route"] = float(
            getattr(args, "query_cgp_route_loss_coef", 0.01)
        )
        losses.append("query_cgp")

    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l, saliency_margin=args.saliency_margin
    )

    criterion.to(device)
    return model, criterion
