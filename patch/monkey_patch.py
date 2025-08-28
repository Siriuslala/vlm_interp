"""
Modify libraries
"""

from .qwen2_5_vl_hijack import *
from .llava1_5_vl_hijack import *
from .intern_2_5_hijack import *

import transformers

from functools import partial
import types


"""
Shuffle the order of image tokens in LLM.
Used in:
test_pos_embed.py
"""
def replace_qwen2_5_vl_shuffle_image_token_orders(delete_vision_token=False):
    _partial = partial(shuffle_llm_image_order_forward, delete_vision_token=delete_vision_token)
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)

def replace_llava1_5_vl_shuffle_image_token_orders(delete_vision_token=False):
    # _partial = partial(llava1_5_forward_shuffle_llm_image_order, delete_vision_token=delete_vision_token)
    # transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = llava1_5_forward_shuffle_llm_image_order

def replace_intern2_5_vl_shuffle_image_token_orders(model):
    model._original_generate = model.generate
    model.generate = types.MethodType(intern2_5_vl_generate_shuffle_image_tokens, model)

def replace_llava1_5_ori():
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = llava1_5_forward_ori
    # _partial = partial(llava1_5_forward_ori)
    # transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)


"""
Delete the position embeddings in LLM.
Used in:
test_pos_embed.py
"""
def replace_qwen2_5_vl_delete_llm_pos_embed(layer_ids_to_delete):
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = new_apply_multi_rope
    _partial = partial(delete_llm_pos_embed_forward, layer_ids_to_delete=layer_ids_to_delete)
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)


"""
Delete the position embeddings in ViT.
Used in:
test_pos_embed.py
"""
def replace_qwen2_5_vl_delete_vit_pos_embed(layer_ids_to_delete):
    # transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_vision = qwen2_5_vl_new_apply_rope
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = qwen2_5_vl_new_apply_rope_flashatt
    _partial = partial(qwen2_5_vl_delete_vit_pos_embed_forward, layer_ids_to_delete=layer_ids_to_delete)
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)

def replace_qwen2_vl_delete_vit_pos_embed(layer_ids_to_delete):
    transformers.models.qwen2_vl.modeling_qwen2_vl.apply_rotary_pos_emb_vision = qwen2_vl_new_apply_rotary_pos_emb_vision
    _partial = partial(qwen2_vl_delete_vit_pos_embed_forward, layer_ids_to_delete=layer_ids_to_delete)
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)
    
def replace_llava1_5_vl_delete_vit_pos_embed():
    transformers.models.clip.modeling_clip.CLIPVisionEmbeddings.forward = clip_embedding_delete_pos_embed
    
def replace_llava1_6_vl_delete_vit_pos_embed():
    transformers.models.clip.modeling_clip.CLIPVisionEmbeddings.forward = clip_embedding_delete_pos_embed

def replace_intern2_5_delete_vit_pos_embed(model):
    model.vision_model.embeddings._original_forward = model.vision_model.embeddings.forward
    model.vision_model.embeddings.forward = types.MethodType(intern2_5_vl_embedding_delete_pos_emed, model.vision_model.embeddings)

# add pos embed
def replace_llava1_5_vl_add_vit_pos_embed():
    transformers.models.clip.modeling_clip.CLIPVisionEmbeddings.forward = llava_1_5_clip_embeddings_forward_add_pos_embed
    transformers.models.clip.modeling_clip.CLIPEncoder.forward = llava_1_5_clip_encoder_forward_add_pos_embed
    transformers.models.clip.modeling_clip.CLIPVisionTransformer.forward = llava_1_5_clip_vit_forward_add_pos_embed

"""
Delete the position embeddings of image patches in LLM.
Used in:
test_pos_embed.py
"""
def replace_qwen2_5_vl_delete_llm_image_pos_embed(layer_ids_to_delete):
    # transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = new_apply_multi_rope_without_image
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = Qwen2_5_VLFlashAttention2_forward_delete_image_rope
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = Qwen2_5_VLDecoderLayer_forward_pass_image_mask
    _partial = partial(Qwen2_5_VLModel_forward_delete_image_rope, layer_ids_to_delete=layer_ids_to_delete)
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)
    # transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_rope_index = get_rope_index_delete_image_rope
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_forward_delete_image_rope

def replace_llava1_5_vl_delete_llm_image_pos_embed(layer_ids_to_delete):
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = llava1_5_forward_delete_image_pos_embed
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_forward_delete_image_pos_embed
    _partial = partial(LlamaModel_forward_delete_image_pos_embed, layer_ids_to_delete=layer_ids_to_delete)
    transformers.models.llama.modeling_llama.LlamaModel.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = LlamaDecoderLayer_delete_image_pos_embed
    transformers.models.llama.modeling_llama.LlamaAttention.forward = LlamaAttention_delete_image_pos_embed
    # transformers.models.llama.modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb_without_image

"""
Shuffle the position ids of vision tokens in LLM.
Used in:
test_pos_embed.py
"""
def replace_qwen2_5_vl_shuffle_llm_image_tokens_pos_ids():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_rope_index = qwen2_5_get_rope_index_shuffle_image_positional_ids

def replace_llava1_5_vl_shuffle_llm_image_tokens_pos_ids():
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = llava1_5_forward_delete_image_pos_embed
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_forward_delete_image_pos_embed
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_forward_shuffle_image_pos_ids


"""
Return indices.
Used in:
test_direction.py
"""
def replace_qwen2_5_vl_test_directions_processor_return_indices():
    transformers.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor.__call__ = qwen2_5_processor_call_return_indices

def replace_qwen2_vl_test_directions_processor_return_indices():
    transformers.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor.__call__ = qwen2_processor_call_return_indices

def replace_llava1_5_test_directions_processor_return_indices():
    transformers.models.llava.processing_llava.LlavaProcessor.__call__ = llava1_5_processor_forward_return_image_size
    
"""
Delete ViT layers.
Used in:
test_direction.py
"""
def replace_qwen2_5_vl_test_directions_vit_discard_layers(layer_ids_to_delete):
    _partial = partial(qwen2_5_vl_vit_forward_discard_layers, layer_ids_to_delete=layer_ids_to_delete)
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)

def replace_qwen2_5_vl_test_directions_vit_discard_layers_and_delete_pos_embed(layer_ids_to_delete):
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_vision = qwen2_5_vl_new_apply_rope
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_flashatt = qwen2_5_vl_new_apply_rope_flashatt
    _partial = partial(qwen2_5_vl_vit_forward_discard_layers_and_delete_vit_pos_embed, layer_ids_to_delete=layer_ids_to_delete)
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)
    
def replace_qwen2_vl_test_directions_vit_discard_layers(layer_ids_to_delete):
    _partial = partial(qwen2_vl_vit_forward_discard_layers, layer_ids_to_delete=layer_ids_to_delete)
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)

def replace_llava1_5_test_directions_vit_discard_layers(layer_ids_to_delete):
    _partial = partial(llava1_5_clip_encoder_forward_discard_layers, layer_ids_to_delete=layer_ids_to_delete)
    transformers.models.clip.modeling_clip.CLIPEncoder.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)

def replace_intern2_5_vl_test_directions_vit_discard_layers(model, layer_ids_to_delete):
    _partial = partial(intern2_5_vl_vit_forward_discard_layers, layer_ids_to_delete=layer_ids_to_delete)
    model.vision_model.encoder._original_forward = model.vision_model.encoder.forward
    model.vision_model.encoder.forward = types.MethodType(lambda self, *args, **kwargs: _partial(self, *args, **kwargs), model.vision_model.encoder)


def replace_qwen2_5_vl_test_seg_vit_output_hidden_states():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward = qwen2_5_vl_vit_output_hidden_states

def replace_qwen2_5_vl_return_image_mask():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_forward_return_image_mask

def replace_qwen2_vl_return_image_mask():
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = Qwen2VLForConditionalGeneration_forward_return_image_mask

def replace_llava1_5_return_image_mask():
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = llava1_5_forward_return_image_mask
    
def replace_llava1_5_processor_forward_return_image_size():
    transformers.models.llava.processing_llava.LlavaProcessor.__call__ = llava1_5_processor_forward_return_image_size

   
"""
Pass ViT output(after connector) to LLM forward function.
Used in:
test_direction.py
    intervene_in_spatial_reasoning
    erase_object_in_llm
"""
def replace_qwen2_5_vl_receive_vit_output():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_forward_receive_vit_output

def replace_qwen2_5_vl_receive_vit_output_and_return_image_mask_and_specify_hidden_states():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_forward_receive_vit_output_and_return_image_mask

def replace_llava1_5_receive_vit_output():
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = llava1_5_forward_receive_vit_output

def replace_llava1_5_receive_vit_output_and_return_image_mask_and_specify_hidden_states():
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = llava1_5_forward_receive_vit_output_and_return_image_mask
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_specify_hidden_states

"""
Get ViT outputs of all layers, before connector.
Used in:
test_direction.py
    explore_relation_in_rope
"""
def replace_qwen2_5_vl_test_directions_vit_return_hidden_states():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward = qwen2_5_vl_vit_return_hidden_states
    
def replace_qwen2_vl_test_directions_vit_return_hidden_states():
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward = qwen2_vl_vit_return_hidden_states
    

"""
Change the attention at the LLM stage at the image positions to bidirectional attention.
Used in:
test_interaction.py
"""
def replace_qwen2_5_vl_llm_image_bidirectional_attention(include_vision_tokens=False):
    _partial = partial(Qwen2_5_VLForConditionalGeneration_forward_pass_image_mask, include_vision_tokens=include_vision_tokens)
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward_receive_image_mask_llm_image_bidirectional_attention
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = Qwen2_5_VLModel_update_causal_mask_llm_image_bidirectional_attention
    # transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = Qwen2_5_VLDecoderLayer_forward_pass_image_mask
    # transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = Qwen2_5_VLFlashAttention2_forward_delete_image_rope
    
def replace_qwen2_vl_llm_image_bidirectional_attention(include_vision_tokens=False):
    _partial = partial(Qwen2VLForConditionalGeneration_forward_pass_image_mask, include_vision_tokens=include_vision_tokens)
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = Qwen2_VLModel_forward_receive_image_mask_llm_image_bidirectional_attention
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = Qwen2_VLModel_update_causal_mask_llm_image_bidirectional_attention

def replace_llava1_5_llm_image_bidirectional_attention(include_vision_tokens=False):
    # def new_forward(self, *args, **kwargs):
    #     kwargs['include_vision_tokens'] = include_vision_tokens
    #     return llava1_5_forward_receive_image_mask_llm_image_bidirectional_attention(self, *args, **kwargs)
    # transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = new_forward
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = llava1_5_forward_receive_image_mask_llm_image_bidirectional_attention
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_forward_llm_image_bidirectional_attention
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_forward_llm_image_bidirectional_attention
    transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask = LlamaModel_update_causal_mask_llm_image_bidirectional_attention

def replace_llava1_5_llm_image_bidirectional_attention_and_return_image_mask(include_vision_tokens=False):
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = llava1_5_forward_receive_image_mask_llm_image_bidirectional_attention_and_return_image_mask
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_forward_llm_image_bidirectional_attention
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_forward_llm_image_bidirectional_attention
    transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask = LlamaModel_update_causal_mask_llm_image_bidirectional_attention
        
def replace_intern2_5_vl_llm_image_bidirectional_attention(model):
    model.forward = types.MethodType(intern2_5_vl_chat_model_forward_pass_image_mask, model)
    model.language_model.forward = types.MethodType(InternLM2ForCausalLM_forward_receive_image_mask_llm_image_bidirectional_attention, model.language_model)
    model.language_model.model.forward = types.MethodType(InternLM2Model_forward_llm_image_bidirectional_attention, model.language_model.model)
    model.language_model.model._prepare_decoder_attention_mask = types.MethodType(InternLM2Model_prepare_decoder_attention_mask, model.language_model.model)

"""
Change the attention at the LLM stage at the image positions to no attention.
Used in:
test_interaction.py
"""
def replace_qwen2_5_vl_llm_image_no_attention(include_vision_tokens=False):
    _partial = partial(Qwen2_5_VLForConditionalGeneration_forward_pass_image_mask, include_vision_tokens=include_vision_tokens)
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward_receive_image_mask_llm_image_bidirectional_attention
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = Qwen2_5_VLModel_update_causal_mask_llm_image_no_attention
    
def replace_qwen2_vl_llm_image_no_attention(include_vision_tokens=False):
    _partial = partial(Qwen2VLForConditionalGeneration_forward_pass_image_mask, include_vision_tokens=include_vision_tokens)
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = Qwen2_VLModel_forward_receive_image_mask_llm_image_bidirectional_attention
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = Qwen2_VLModel_update_causal_mask_llm_image_no_attention

def replace_llava1_5_llm_image_no_attention(include_vision_tokens=False):
    # def new_forward(self, *args, **kwargs):
    #     kwargs['include_vision_tokens'] = include_vision_tokens
    #     return llava1_5_forward_receive_image_mask_llm_image_bidirectional_attention(self, *args, **kwargs)
    # transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = new_forward
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = llava1_5_forward_receive_image_mask_llm_image_bidirectional_attention
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_forward_llm_image_bidirectional_attention
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_forward_llm_image_bidirectional_attention
    transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask = LlamaModel_update_causal_mask_llm_image_bidirectional_attention

def replace_llava1_5_llm_image_no_attention_and_return_image_mask(include_vision_tokens=False):
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = llava1_5_forward_receive_image_mask_llm_image_bidirectional_attention_and_return_image_mask
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_forward_llm_image_bidirectional_attention
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_forward_llm_image_bidirectional_attention
    transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask = LlamaModel_update_causal_mask_llm_image_bidirectional_attention
        
def replace_intern2_5_vl_llm_image_no_attention(model):
    model.forward = types.MethodType(intern2_5_vl_chat_model_forward_pass_image_mask, model)
    model.language_model.forward = types.MethodType(InternLM2ForCausalLM_forward_receive_image_mask_llm_image_bidirectional_attention, model.language_model)
    model.language_model.model.forward = types.MethodType(InternLM2Model_forward_llm_image_bidirectional_attention, model.language_model.model)
    model.language_model.model._prepare_decoder_attention_mask = types.MethodType(InternLM2Model_prepare_decoder_attention_mask, model.language_model.model)   

"""
Explore the RoPE attention by dimension group.
Used in:
test_direction.py
"""
def replace_qwen2_5_vl_attention_pattern_dimension_group(head_id=None):
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel = Qwen2_5_VisionTransformerPretrainedModel_return_attention_pattern
    _partial = partial(Qwen2_5_VLVisionAttention_forward_by_rope_dimension_group, head_id=head_id)
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)

def replace_qwen2_vl_attention_pattern_dimension_group(head_id=None):
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel = Qwen2VisionTransformerPretrainedModel_return_attention_pattern
    _partial = partial(Qwen2VL_VisionAttention_forward_by_rope_dimension_group, head_id=head_id)
    transformers.models.qwen2_vl.modeling_qwen2_vl.VisionAttention.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)

def replace_qwen2_vl_rope_attention_h_w_separate(method=None):
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLVisionBlock = Qwen2VLVisionBlock_eager_attention
    _partial = partial(Qwen2VL_VisionAttention_forward_rope_attention_h_w_separate, method=method)
    transformers.models.qwen2_vl.modeling_qwen2_vl.VisionAttention.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)

def replace_qwen2_vl_scaling_rope(scaling_type="linear", alpha=1.0, gamma=2.0, beta=0.1, base=512, poly_p=8, poly_alpha=99, sig_alpha=99, sig_mid_point=0.5, sig_k=20.0):
    _partial = partial(Qwen2VisionTransformerPretrainedModel_init_scaling_rope, scaling_type=scaling_type, alpha=alpha, gamma=gamma, beta=beta, base=base, poly_p=poly_p, poly_alpha=poly_alpha, sig_alpha=sig_alpha, sig_mid_point=sig_mid_point, sig_k=sig_k)
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.__init__ = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)

    
"""
Explore RoPE by sensitivity (gradient).
Used in:
test_direction.py
"""
def replace_qwen2_vl_rope_sensitivity(activation_name=None):
    _partial = partial(Qwen2VLForConditionalGeneration_forward_return_activations, activation_name=activation_name)
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = lambda self, *args, **kwargs: _partial(self, *args, **kwargs)
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward = Qwen2VisionTransformerPretrainedModel_forward_return_activations
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLVisionBlock.forward = Qwen2VLVisionBlock_forward_return_activations
    transformers.models.qwen2_vl.modeling_qwen2_vl.VisionSdpaAttention.forward = VisionSdpaAttention_forward_return_activations
    
"""
Token truncation using logit lens.
"""
def replace_llava_1_5_token_truncation_by_logit_lens():
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = LlavaForConditionalGeneration_forward_pass_image_mask
    transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM_token_truncation_by_logit_lens
    transformers.models.llama.modeling_llama.LlamaModel = LlamaModel_token_truncation_by_logit_lens

def replace_llava_1_5_logit_lens_adaptive():
    # image info density
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = LlavaForConditionalGeneration_forward_pass_image_mask
    transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM_token_truncation_by_logit_lens_adaptive

def replace_llava_1_5_token_truncation_by_logit_lens_turnback():
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = LlavaForConditionalGeneration_forward_pass_image_mask
    transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM_token_truncation_by_logit_lens_turnback
    transformers.models.llama.modeling_llama.LlamaModel = LlamaModel_token_truncation_by_logit_lens_turnback
    
def replace_llava_1_5_token_truncation_by_logit_lens_runlength():
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = LlavaForConditionalGeneration_forward_pass_image_mask
    transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM_token_truncation_by_logit_lens_runlength
    transformers.models.llama.modeling_llama.LlamaModel = LlamaModel_token_truncation_by_logit_lens_runlength

def replace_llava_1_5_token_truncation_by_logit_lens_runlength_adaptive():
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.__init__ = LlavaForConditionalGeneration_init_token_truncation_by_logit_lens_runlength_adaptive
    transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.forward = LlavaForConditionalGeneration_forward_pass_image_mask
    # transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM_token_truncation_by_logit_lens_runlength_adaptive
    transformers.models.llama.modeling_llama.LlamaModel = LlamaModel_token_truncation_by_logit_lens_runlength_adaptive

def replace_qwen2_5_vl_token_truncation_by_logit_lens_runlength_adaptive():
    # transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration_runlength_adaptive  # failed, maybe bacause Qwen2_5_VLForConditionalGeneration uses `from_pretrained`
    
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel = Qwen2_5_VLModel_token_truncation_by_logit_lens_runlength_adaptive
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.__init__ = Qwen2_5_VLForConditionalGeneration_init_runlength_adaptive
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_forward_pass_image_mask

def replace_qwen2_vl_token_truncation_by_logit_lens_runlength_adaptive():    
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel = Qwen2VLModel_token_truncation_by_logit_lens_runlength_adaptive
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.__init__ = Qwen2VLForConditionalGeneration_init_runlength_adaptive
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = Qwen2VLForConditionalGeneration_forward_pass_image_mask

def replace_intern2_5_vl_token_truncation_by_logit_lens_runlength_adaptive(model):
    model.generate = types.MethodType(InternVLChatModel_generate_pass_image_mask, model)
    model.language_model.forward = types.MethodType(InternLM2ForCausalLM_forward_pass_image_mask, model.language_model)
    # model.language_model.model.lm_head = model.language_model.output
    # model.language_model.model.processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL2_5-8B", padding_side='left', use_fast=True, trust_remote_code=True)
    # model.language_model.model.forward = types.MethodType(InternLM2Model_forward_token_truncation, model.language_model.model)
    