import torch
import torch.nn as nn
from torch_geometric.data import Data

from transformers import ViTConfig
import torch.nn.functional as F

import time
from einops import rearrange

from typing import Dict, List, Optional, Set, Tuple, Union
import math

class Brightness_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Brightness_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map
    
class PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.IMAGE_SIZE, config.PATCH_SIZE
        num_channels, hidden_size = config.NUM_CHANNELS, config.HIDDEN_SIZE


        image_size = image_size 
        patch_size = (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, N, height, width = images.shape
        images = rearrange(images, 'b c n h w -> (b n) c h w')
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(images).flatten(2).transpose(1, 2)

        return embeddings
    
class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.HIDDEN_SIZE % config.NUM_ATTENTION_HEADS != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.HIDDEN_SIZE,} is not a multiple of the number of attention "
                f"heads {config.NUM_ATTENTION_HEADS}."
            )

        self.num_attention_heads = config.NUM_ATTENTION_HEADS
        self.attention_head_size = int(config.HIDDEN_SIZE / config.NUM_ATTENTION_HEADS)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.HIDDEN_SIZE, self.all_head_size, bias=config.QKV_BIAS)
        self.key = nn.Linear(config.HIDDEN_SIZE, self.all_head_size, bias=config.QKV_BIAS)
        self.value = nn.Linear(config.HIDDEN_SIZE, self.all_head_size, bias=config.QKV_BIAS)

        self.dropout = nn.Dropout(config.ATTENTION_PROBS_DROPOUT_PROB)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, illu_feas, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        illu_feas = self.transpose_for_scores(illu_feas)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        value_layer = illu_feas * value_layer

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
        self.dropout = nn.Dropout(config.HIDDEN_DROPOUT_PROB)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
    
def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
   
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
        self.pruned_heads = set()
    
    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        illu_feas : torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, illu_feas, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs



ACT2FN = {
    "relu": torch.nn.functional.relu,
    "gelu": torch.nn.functional.gelu,
    "tanh": torch.tanh,
    # Add other activation functions as needed
}

class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.HIDDEN_SIZE, config.INTERMEDIATE_SIZE)
        if isinstance(config.HIDDEN_ACT, str):
            self.intermediate_act_fn = ACT2FN[config.HIDDEN_ACT]
        else:
            self.intermediate_act_fn = config.HIDDEN_ACT

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.INTERMEDIATE_SIZE, config.HIDDEN_SIZE)
        self.dropout = nn.Dropout(config.HIDDEN_DROPOUT_PROB)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states

VIT_ATTENTION_CLASSES = {
    "eager": ViTAttention,
}


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.layernorm_before = nn.LayerNorm(config.HIDDEN_SIZE, eps=config.LAYER_NORM_EPS)
        self.attention = ViTAttention(config)
        self.layernorm_after = nn.LayerNorm(config.HIDDEN_SIZE, eps=config.LAYER_NORM_EPS)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        illu_feas : torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            illu_feas,  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs
    
class PREncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        image_size, patch_size = config.IMAGE_SIZE, config.PATCH_SIZE
        image_size = image_size 
        patch_size = (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.config = config
        self.gradient_checkpointing = False
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.HIDDEN_SIZE))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, config.HIDDEN_SIZE))
        self.patch_embed = PatchEmbeddings(config)
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.NUM_HIDDEN_LAYERS)])
        self.norm = nn.LayerNorm(config.HIDDEN_SIZE, eps=config.LAYER_NORM_EPS)
        self.batch_size = config.BATCH_SIZE
        

        self.brightness_estimator = Brightness_Estimator(n_fea_middle=config.HIDDEN_SIZE)
        self.illu_proj = nn.Conv2d(config.HIDDEN_SIZE, config.HIDDEN_SIZE, kernel_size=config.PATCH_SIZE, stride=config.PATCH_SIZE)
        self.illu_cls_token = nn.Parameter(torch.zeros(1, 1, config.HIDDEN_SIZE))
    def apply_attn(self, x, illu_feas,head_mask):                        
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    illu_feas,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(x, illu_feas, layer_head_mask)

            hidden_states = layer_outputs[0]
            # if output_attentions:
            #     all_self_attentions = (layer_outputs[1],)
            # all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states
    
    def illu_embed(self, images):
        b, c, T, h, w = images.shape
        br_fea_dim = self.config.HIDDEN_SIZE
        illu_feas = torch.zeros(b, br_fea_dim, T, int(h), int(w), device="cuda", dtype=torch.float)
        illu_maps = torch.zeros(b, c, T, int(h), int(w), device="cuda", dtype=torch.float)
        for i in range(images.size(2)):
            img = images[:, :, i:i+1, :, :]
            image = img.squeeze(2)
            illu_fea, illu_map = self.brightness_estimator(image)
            
            illu_fea = illu_fea.unsqueeze(2)
            illu_map = illu_map.unsqueeze(2)

            illu_feas[:, :, i:i+1, :, :] = illu_fea
            illu_maps[:, :, i:i+1, :, :] = illu_map
        illu_feas = rearrange(illu_feas, 'b c n h w -> (b n) c h w')
        illu_feas = self.illu_proj(illu_feas).flatten(2).transpose(1, 2)
        return illu_feas
    
    def forward(self, batch_dict):
        images = batch_dict['images']
        head_mask = batch_dict['head_mask']
        img_embed = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(img_embed.size(0), -1, -1)
        img_embed = torch.cat((cls_tokens, img_embed), dim=1)
        img_embed = img_embed + self.pos_embed
        reshaped_img_embed = rearrange(img_embed, '(a d) b c -> a (d b) c', d=2)

        illu_feas = self.illu_embed(images)
        illu_cls_tokens = self.illu_cls_token.expand(illu_feas.size(0), -1, -1)
        illu_embed = torch.cat((illu_cls_tokens, illu_feas), dim=1)
        illu_embed = rearrange(illu_embed, '(a d) b c -> a (d b) c', d=2)
        hidden_states = self.apply_attn(reshaped_img_embed, illu_embed, head_mask)
        # Take the cls token
        x = self.norm(hidden_states)
        return x[:,0]

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
    
class PRDecoder(nn.Module):
    def __init__(self, config):
        super(PRDecoder, self).__init__()
        self.hidden_dim = config.HIDDEN_SIZE
        self.output_dim = config.OUTPUT_SIZE
        self.mlp_ratio = config.MLP_RATIO
        self.norm_layer = nn.LayerNorm
        self.mlp = Mlp(self.hidden_dim, self.hidden_dim * int(self.mlp_ratio), self.hidden_dim, act_layer=nn.GELU)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.norm2 = self.norm_layer(self.hidden_dim)
        self.drop_path = nn.Identity()
    def forward(self, x):
        # hidden_dim = x.shape[-1]
        # mlp_channels = [4*hidden_dim, hidden_dim]
        # x = self.build_mlps(hidden_dim, mlp_channels=mlp_channels).to(img_feature.device)
        # hidden_states = img_feature
        # hidden_out = x(hidden_states)
        # output = self.output_layer(hidden_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.drop_path(self.output_layer(self.norm2(x)))
        return x
    

class VisualOdometryTransformer(nn.Module):
    def __init__(self, config):
        super(VisualOdometryTransformer, self).__init__()

        self.config = config

        self.encoder = PREncoder(config.PR_ENCODER)
        self.decoder = PRDecoder(config.PR_DECODER)
         
    def forward(self, batch_dict):        
        img_feature = self.encoder(batch_dict)
        refined_pose = self.decoder(img_feature)
        return refined_pose



