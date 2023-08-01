import copy
import math

import torch
from torch.nn.functional import relu, gelu, glu
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from models.ops.modules import MSDeformAttn


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DeformableTransformer(nn.Module):

    def __init__(self,
                 hidden_dim=256,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 feedforward_dim=1024,
                 dropout=0.0,
                 activation="relu",
                 return_intermediate_dec=True,
                 num_feature_levels=4,
                 dec_n_points=4,
                 enc_n_points=4,
                 two_stage=False,
                 two_stage_num_proposals=300):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_feature_levels = num_feature_levels
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.hda = []
        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_dim, feedforward_dim,
            dropout, activation,
            num_feature_levels, num_heads, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, self.hda)
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, feedforward_dim,
            dropout, activation,
            num_feature_levels, num_heads, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers,
                                                    return_intermediate_dec, self.hda)
        self.mae_decoder = None
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, hidden_dim))
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)
            self.pos_trans = nn.Linear(hidden_dim * 2, hidden_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(hidden_dim * 2)
        else:
            self.reference_points = nn.Linear(hidden_dim, 2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m.reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def build_mae_decoder(self, image_size, mae_layers, device, channel0=512):
        # Generate spatial shape according to image size
        h, w, c = math.ceil(image_size[0] / 8), math.ceil(image_size[1] / 8), channel0
        total_spatial_shapes = []
        for i in range(3):
            total_spatial_shapes.append([h, w, c])
            h = math.ceil(h / 2)
            w = math.ceil(w / 2)
            c *= 2
        # Build mae decoder
        self.mae_decoder = DeformableTransformerDecoderMAE(
            hidden_dim=self.hidden_dim,
            feedforward_dim=self.feedforward_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            num_feature_levels=self.num_feature_levels,
            mae_layers=mae_layers,
            total_spatial_shapes=total_spatial_shapes,
        )
        self.mae_decoder.to(device)

    @staticmethod
    def get_proposal_pos_embed(proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        n_, s_, c_ = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(n_, H_, W_, 1)
            valid_h = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_w = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
                                            indexing='ij')
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1).view(n_, 1, 1, 2)
            grid = torch.div((grid.unsqueeze(0).expand(n_, -1, -1, -1) + 0.5), scale)
            wh = torch.mul(torch.ones_like(grid) * 0.05, (2.0 ** lvl))
            proposal = torch.cat((grid, wh), -1).view(n_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = torch.all(torch.bitwise_and(torch.gt(output_proposals, 0.01), output_proposals < 0.99),
                                           dim=-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(torch.unsqueeze(memory_padding_mask, -1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        output_memory = memory
        output_memory = output_memory.masked_fill(torch.unsqueeze(memory_padding_mask, -1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_valid_ratio(mask):
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, enable_mae=False):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # encoder
        memory, inter_memory = self.encoder(src_flatten, spatial_shapes, level_start_index,
                                            valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # MAE decoder
        if enable_mae:
            assert self.mae_decoder is not None
            mae_output = self.mae_decoder(None, None, memory, spatial_shapes, level_start_index,
                                          valid_ratios, None, mask_flatten)
            return mae_output
        # prepare input for decoder
        assert self.two_stage or query_embed is not None
        bs, _, c = memory.shape
        enc_out_class, enc_out_coord_un_act = None, None
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            # hack implementation for two-stage Deformable DETR
            enc_out_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_out_coord_un_act = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_out_class[..., 0], topk, dim=1)[1]
            topk_coords_un_act = torch.gather(enc_out_coord_un_act, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_un_act = topk_coords_un_act.detach()
            reference_points = topk_coords_un_act.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_un_act)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
        # decoder
        hs, inter_ref, inter_object_query = self.decoder(tgt, reference_points, memory, spatial_shapes,
                                                         level_start_index, valid_ratios, query_embed, mask_flatten)
        return hs, init_reference_out, inter_ref, enc_out_class, enc_out_coord_un_act, inter_memory, inter_object_query


class DeformableTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src,
                              spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, hda=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.hda = hda + [num_layers]

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                                          indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        inter_memory = []
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer_idx, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            if self.hda is not None and (layer_idx+1) in self.hda:
                inter_memory.append(output)
        inter_memory = torch.stack(inter_memory, dim=1)
        return output, inter_memory


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes,
                level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False, hda=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.hda = hda + [num_layers]

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        intermediate_object_query = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,
                           src_level_start_index, src_padding_mask)
            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
            if self.hda is not None and (lid+1) in self.hda:
                intermediate_object_query.append(output)
        intermediate_object_query = torch.stack(intermediate_object_query, dim=1)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), intermediate_object_query
        return output, reference_points, intermediate_object_query


class DeformableTransformerDecoderMAE(DeformableTransformerDecoder):

    def __init__(self,
                 hidden_dim=256,
                 feedforward_dim=1024,
                 num_heads=8,
                 dropout=0.1,
                 activation='relu',
                 num_feature_levels=4,
                 n_points=4,
                 num_layers=2,
                 return_intermediate=False,
                 mae_layers=None,
                 total_spatial_shapes=None):
        mae_layers = [] if mae_layers is None else mae_layers
        total_spatial_shapes = [] if total_spatial_shapes is None else total_spatial_shapes
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, feedforward_dim,
            dropout, activation,
            num_feature_levels, num_heads, n_points
        )
        super(DeformableTransformerDecoderMAE, self).__init__(decoder_layer, num_layers, return_intermediate, hda=[])
        self.mae_layers = mae_layers
        self.spatial_shapes = [
            total_spatial_shapes[mae_layer] for mae_layer in mae_layers
        ]
        self.hidden_dim = hidden_dim
        self.mask_query = nn.Embedding(1, hidden_dim)
        self.query_embed_list = nn.ModuleList([
            nn.Embedding(h * w, hidden_dim * 2)
            for h, w, c in self.spatial_shapes
        ])
        self.reference_points = nn.Linear(hidden_dim, 2)
        self.output_proj = nn.ModuleList([
            nn.Linear(hidden_dim, c)
            for h, w, c in self.spatial_shapes
        ])

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, mask_flatten=None):
        bs = src.shape[0]
        mae_output = []
        for i, mae_layer in enumerate(self.mae_layers):
            query_pos, tgt = torch.split(self.query_embed_list[i].weight, self.hidden_dim, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            tgt_mask = mask_flatten[:, src_level_start_index[mae_layer]: src_level_start_index[mae_layer+1]]
            tgt_mask = torch.unsqueeze(tgt_mask, -1)
            h, w, c = self.spatial_shapes[i]
            tgt = tgt * (~tgt_mask).to(tgt.dtype) + self.mask_query.weight.\
                expand(bs, h * w, -1) * tgt_mask.to(tgt.dtype)
            reference_points = self.reference_points(query_pos).sigmoid()
            hs, _, _ = super(DeformableTransformerDecoderMAE, self).forward(
                tgt, reference_points, src,
                src_spatial_shapes, src_level_start_index,
                src_valid_ratios, query_pos, mask_flatten
            )
            output = self.output_proj[i](hs)
            mae_output.append(output.transpose(-2, -1).reshape(-1, c, h, w))
        return mae_output


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return relu
    if activation == "gelu":
        return gelu
    if activation == "glu":
        return glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
