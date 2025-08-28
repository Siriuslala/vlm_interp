from typing import Optional, Tuple, Union, List

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.models.layers import DropPath
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, AutoProcessor
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPast,)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

from .internvl2_5_utils.modeling_internlm2 import InternLM2Config, InternLM2DecoderLayer, InternLM2Model, InternLM2RMSNorm, InternLM2ForCausalLM, InternLM2PreTrainedModel
from .internvl2_5_utils.modeling_internvl_chat import InternVLChatConfig, InternVLChatModel

logger = logging.get_logger(__name__)

import os
import copy
import string
import jsonlines

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import \
        flash_attn_varlen_qkvpacked_func
    has_flash_attn = True
except:
    print('FlashAttention2 is not installed.')
    has_flash_attn = False

def _import_flash_attn():
    global flash_attn_func, flash_attn_varlen_func
    global pad_input, index_first_axis, unpad_input
    try:
        from flash_attn import flash_attn_func as _flash_attn_func
        from flash_attn import \
            flash_attn_varlen_func as _flash_attn_varlen_func
        from flash_attn.bert_padding import \
            index_first_axis as _index_first_axis
        from flash_attn.bert_padding import pad_input as _pad_input
        from flash_attn.bert_padding import unpad_input as _unpad_input
        flash_attn_func, flash_attn_varlen_func = _flash_attn_func, _flash_attn_varlen_func
        pad_input, index_first_axis, unpad_input = _pad_input, _index_first_axis, _unpad_input
    except ImportError:
        raise ImportError('flash_attn is not installed.')

class InternVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternVisionModel`]. It is used to
    instantiate a vision encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            Number of color channels in the input images (e.g., 3 for RGB).
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries and values in the self-attention layers.
        hidden_size (`int`, *optional*, defaults to 3200):
            Dimensionality of the encoder layers and the pooler layer.
        num_attention_heads (`int`, *optional*, defaults to 25):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 12800):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        qk_normalization (`bool`, *optional*, defaults to `True`):
            Whether to normalize the queries and keys in the self-attention layers.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        use_flash_attn (`bool`, *optional*, defaults to `True`):
            Whether to use flash attention mechanism.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate for stochastic depth.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 0.1):
            A factor for layer scale.
    """

    model_type = 'intern_vit_6b'

    def __init__(
            self,
            num_channels=3,
            patch_size=14,
            image_size=224,
            qkv_bias=False,
            hidden_size=3200,
            num_attention_heads=25,
            intermediate_size=12800,
            qk_normalization=True,
            num_hidden_layers=48,
            use_flash_attn=True,
            hidden_act='gelu',
            norm_type='rms_norm',
            layer_norm_eps=1e-6,
            dropout=0.0,
            drop_path_rate=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.norm_type = norm_type
        self.qkv_bias = qkv_bias
        self.qk_normalization = qk_normalization
        self.use_flash_attn = use_flash_attn

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'vision_config' in config_dict:
            config_dict = config_dict['vision_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)
        
class InternLM2PreTrainedModel(PreTrainedModel):
    config_class = InternLM2Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternLM2DecoderLayer']
    _skip_keys_device_placement = 'past_key_values'
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


 
def intern2_5_vl_vit_forward_discard_layers(
        self,
        inputs_embeds,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        layer_ids_to_delete: Optional[Tuple[int]] = None,
) -> Union[Tuple, BaseModelOutput]:
    r"""
    Args:
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Embedded representation of the inputs. Should be float, not int tokens.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    encoder_states = () if output_hidden_states else None
    hidden_states = inputs_embeds

    for idx, encoder_layer in enumerate(self.layers):
        # print("xiix")
        # print(layer_ids_to_delete)
        # import pdb; pdb.set_trace()
        if idx in layer_ids_to_delete:
            continue
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(
                encoder_layer,
                hidden_states)
        else:
            layer_outputs = encoder_layer(
                hidden_states,
            )
        hidden_states = layer_outputs

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_states
    )

def intern2_5_vl_embedding_delete_pos_emed(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
    batch_size, _, height, width = patch_embeds.shape
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
    class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    return embeddings

@torch.no_grad()
def intern2_5_vl_generate_shuffle_image_tokens(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_ids: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    visual_features: Optional[torch.FloatTensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    output_hidden_states: Optional[bool] = None,
    **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
        
        # shuffle image tokens
        mask = copy.deepcopy(selected)
        mask = mask.reshape(B, N)
        for i in range(B):
            visual_embeddings = input_embeds[i][mask[i]]   
            shuffled_visual_embeddings = visual_embeddings[torch.randperm(len(visual_embeddings))]     
            input_embeds[i][mask[i]] = shuffled_visual_embeddings
        # print("yeah")
        # import pdb; pdb.set_trace()
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs

def intern2_5_vl_chat_model_forward_pass_image_mask(
    self,
    pixel_values: torch.FloatTensor,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    image_flags: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    image_flags = image_flags.squeeze(-1)
    input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

    vit_embeds = self.extract_feature(pixel_values)
    vit_embeds = vit_embeds[image_flags == 1]
    vit_batch_size = pixel_values.shape[0]

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == self.img_context_token_id)
    mask = selected.reshape(B, N)
    try:
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
    except Exception as e:
        vit_embeds = vit_embeds.reshape(-1, C)
        print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                f'vit_embeds.shape={vit_embeds.shape}')
        n_token = selected.sum()
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

    input_embeds = input_embeds.reshape(B, N, C)

    outputs = self.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        image_mask=mask,
    )
    logits = outputs.logits

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def InternLM2ForCausalLM_forward_receive_image_mask_llm_image_bidirectional_attention(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    image_mask: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, InternLM2ForCausalLM

    >>> model = InternLM2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        image_mask=image_mask,
    )

    hidden_states = outputs[0]
    logits = self.output(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    device = input_ids.device if input_ids is not None else inputs_embeds.device
    output = CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    output['logits'] = output['logits'].to(device)
    return output

def InternLM2Model_forward_llm_image_bidirectional_attention(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    image_mask: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if self.config.attn_implementation == 'flash_attention_2':
        _import_flash_attn()

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError('You have to specify either input_ids or inputs_embeds')

    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.tok_embeddings(input_ids)

    if self.config.attn_implementation == 'flash_attention_2':
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, image_mask=image_mask
        )

    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def InternLM2Model_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length, image_mask=None):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    
    if image_mask:
        
        _, seq_length = image_mask.shape
        
        # [batch_size, seq_length] -> [batch_size, 1, seq_length, seq_length]
        img_mask_expanded = image_mask.unsqueeze(1).unsqueeze(3).expand(-1, 1, seq_length, input_shape[-1])
        img_mask_expanded_t = image_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, seq_length, input_shape[-1])
        
        visual_bidirectional_mask = img_mask_expanded & img_mask_expanded_t
        
        bidirectional_attention = torch.zeros_like(combined_attention_mask, dtype=torch.bool)
        
        combined_attention_mask = torch.where(
            visual_bidirectional_mask,
            bidirectional_attention,
            combined_attention_mask
        )

    return combined_attention_mask

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def InternVLChatModel_generate_pass_image_mask(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_ids: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    visual_features: Optional[torch.FloatTensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    output_hidden_states: Optional[bool] = None,
    **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    mask = None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        mask = selected.reshape(B, N).squeeze()
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    # self.language_model.forward = InternLM2ForCausalLM_forward_pass_image_mask.__get__(self.language_model, InternLM2ForCausalLM)
    import types
    self.language_model.forward = types.MethodType(InternLM2ForCausalLM_forward_pass_image_mask, self.language_model)
    original_forward = self.language_model.forward
    def custom_forward(*args, **kwargs):
        kwargs["image_mask"] = mask
        return original_forward(*args, **kwargs)
    
    self.language_model.forward = custom_forward
    
    self.language_model.model.lm_head = self.language_model.output
    self.language_model.model.processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL2_5-8B", padding_side='left', use_fast=True, trust_remote_code=True)
    self.language_model.model.forward = types.MethodType(InternLM2Model_forward_token_truncation, self.language_model.model)
    
    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs

def InternLM2ForCausalLM_forward_pass_image_mask(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    image_mask: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        image_mask=image_mask,
    )

    hidden_states = outputs[0]
    logits = self.output(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    device = input_ids.device if input_ids is not None else inputs_embeds.device
    output = CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    output['logits'] = output['logits'].to(device)
    return output

def InternLM2Model_forward_token_truncation(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    image_mask: Optional[torch.Tensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if self.config.attn_implementation == 'flash_attention_2':
        _import_flash_attn()

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError('You have to specify either input_ids or inputs_embeds')
    
    if inputs_embeds is None:
        inputs_embeds = self.tok_embeddings(input_ids)

    # print(seq_length)
    # print(image_mask)
    if seq_length > 1:  # pre-filling
        use_cache = False
        past_key_values = None
        
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.config.attn_implementation == 'flash_attention_2':
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        
    # truncate hidden states by logit lens
    if seq_length > 1:  # pre-filling
        # get text tokens for image tokens
        logits = self.lm_head(hidden_states)  # (batch_size, seq_len, vocab_size)
        # text_token_ids = torch.argmax(logits[0], dim=-1)  # (seq_len)
        _, text_token_ids_top_3 = torch.topk(logits[0], k=3, dim=-1)  # (seq_len, top_k)
        text_tokens = [self.processor.decode(text_id[0]) for text_id in text_token_ids_top_3]
        text_tokens_second = [self.processor.decode(text_id[1]) for text_id in text_token_ids_top_3]
        text_tokens_third = [self.processor.decode(text_id[2]) for text_id in text_token_ids_top_3]
                        
        image_ids = [i for i in range(logits.shape[1]) if image_mask[i]]
        image_text_tokens = [text_tokens[i] for i in image_ids]
        
        # run-length encoding
        compressed_tokens = []
        start_indices = []
        run_lengths = []

        current_token = image_text_tokens[0]
        current_start_index = 0
        current_count = 1

        for i in range(1, len(image_text_tokens)):
            if image_text_tokens[i] == current_token:
            # whether to compress useful tokens
            # if image_text_tokens[i] in meaningless_tokens and image_text_tokens[i] == current_token:
                current_count += 1
            else:
                compressed_tokens.append(current_token)
                start_indices.append(current_start_index)
                run_lengths.append(current_count)

                current_token = image_text_tokens[i]
                current_start_index = i
                current_count = 1

        compressed_tokens.append(current_token)
        start_indices.append(current_start_index)
        run_lengths.append(current_count)
    
    # Turnback:
    # decoder layers, the second time
    # meaningless tokens
    meaningless_tokens = []
    meaningless_tokens.extend(string.punctuation)
    meaningless_tokens.extend(string.whitespace)
    
    if seq_length > 1:
        # compress inputs_embeds
        inputs_embeds_image = inputs_embeds[0, image_ids, :]  # (576, 4096)
        image_id_st = image_ids[0]
        image_id_ed = image_ids[-1]
        # run-length encoding
        new_len = 0
        compressed_embeds_list = []
        for i in range(len(start_indices)):
            
            # whether to del meaningless tokens
            if compressed_tokens[i] in meaningless_tokens:
                continue
            
            start = start_indices[i]
            length = run_lengths[i]
            current_run_embeds = inputs_embeds_image[start : start + length]  # [n, 4096]
            
            # meaning pooling
            averaged_embed = torch.mean(current_run_embeds, dim=0, keepdim=True)
            
            # random select
            # averaged_embed = random.choice(current_run_embeds).unsqueeze(0)
            
            new_len += 1
            compressed_embeds_list.append(averaged_embed)
        
        all_embeds = [inputs_embeds[0, :image_id_st]] + compressed_embeds_list + [inputs_embeds[0, image_id_ed + 1:]]
        inputs_embeds = torch.cat(all_embeds).unsqueeze(0)  # (1, new_seq_len, hidden_size)
        
        ori_len = len(image_ids)
        truncation_left_ratio = new_len / ori_len
        # print(truncation_left_ratio)
        # import pdb; pdb.set_trace()
        save_dir = "/raid_sdd/lyy/Interpretability/lyy/mm/eval/share/test_llm_image_token_truncation-method_5/internvl2_5_8b"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "truncation_ratio.jsonl")
        with jsonlines.open(save_path, "a") as f:
            f.write({"truncation_ratio": truncation_left_ratio})
    
    # if seq_length > 1:
    #     seq_length = inputs_embeds.shape[1]
    # else:
    #     seq_length = inputs_embeds.shape[1] + past_key_values_length
    
    use_cache = True
    seq_length = inputs_embeds.shape[1]
    seq_length_with_past = inputs_embeds.shape[1]
    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0)
    # print(f"position_ids.shape: {position_ids.shape}, seq_length: {seq_length}, past_key_values_length: {past_key_values_length}")
    # import pdb; pdb.set_trace()
    
    if self.config.attn_implementation == 'flash_attention_2':
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )      

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    # print(f"hidden_states.shape after norm: {hidden_states.shape}, seq_length: {seq_length}, past_key_values_length: {past_key_values_length}")
    # import pdb; pdb.set_trace()
    
    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )