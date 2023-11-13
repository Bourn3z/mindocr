from dataclasses import dataclass


@dataclass
class LayoutXLMPretrainedConfig():
    def __init__(self, mode='base', use_float16=False):
        self.mode = mode
        self.use_float16 = use_float16
        self.attention_probs_dropout_prob = 0.1
        self.bos_token_id = 0
        self.coordinate_size = 128
        self.eos_token_id = 2
        self.fast_qkv = False
        self.gradient_checkpointing = False
        self.has_relative_attention_bias = False
        self.has_spatial_attention_bias = False
        self.has_visual_segment_embedding = True
        self.use_visual_backbone = True
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.image_feature_pool_shape = [7, 7, 256]
        self.initializer_range = 0.02
        self.intermediate_size = 3072
        self.layer_norm_eps = 1e-05
        self.max_2d_position_embeddings = 1024
        self.max_position_embeddings = 514
        self.max_rel_2d_pos = 256
        self.max_rel_pos = 128
        self.model_type = "layoutxlm"
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.output_past = True
        self.pad_token_id = 1
        self.shape_size = 128
        self.rel_2d_pos_bins = 64
        self.rel_pos_bins = 32
        self.type_vocab_size = 1
        self.vocab_size = 250002

        if mode == 'vi':
            self.attention_probs_dropout_prob = 0
            self.bos_token_id = 0
            self.coordinate_size = 128
            self.eos_token_id = 2
            self.fast_qkv = False
            self.gradient_checkpointing = False
            self.has_relative_attention_bias = False
            self.has_spatial_attention_bias = False
            self.has_visual_segment_embedding = True
            self.use_visual_backbone = False
            self.hidden_act = "gelu"
            self.hidden_dropout_prob = 0.1
            self.hidden_size = 768
            self.image_feature_pool_shape = [7, 7, 256]
            self.initializer_range = 0.02
            self.intermediate_size = 3072
            self.layer_norm_eps = 1e-05
            self.max_2d_position_embeddings = 1024
            self.max_position_embeddings = 514
            self.max_rel_2d_pos = 256
            self.max_rel_pos = 128
            self.model_type = "layoutxlm"
            self.num_attention_heads = 12
            self.num_hidden_layers = 12
            self.output_past = True
            self.pad_token_id = 1
            self.shape_size = 128
            self.rel_2d_pos_bins = 64
            self.rel_pos_bins = 32
            self.type_vocab_size = 1
            self.vocab_size = 250002
