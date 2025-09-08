import numpy as np
import inout 
import position_embedding as PE
import basic_layer

def layer_norm(x: np.ndarray, weight: np.ndarray , bias: np.ndarray , eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * weight + bias

def Swin_Transformer(x):
    H, W, pos_emb_x = PE.position_embedding_3D(x)

    Stage0_x, Stage0_H, Stage0_W , Stage0_down_x, Stage0_down_H, Stage0_down_W = basic_layer.BasicLayer(pos_emb_x, H, W, 3,
                                                                                                        'swin_1_layers_0_blocks_0_norm1_weight', (96),
                                                                                                        'swin_1_layers_0_blocks_0_norm1_bias'  , (96),
                                                                                                        'swin_1_layers_0_blocks_0_attn_relative_position_bias_table', (169, 3),
                                                                                                        'swin_1_layers_0_blocks_0_attn_relative_position_index', (49, 49), 
                                                                                                        'swin_1_layers_0_blocks_0_attn_qkv_weight', (288, 96),
                                                                                                        'swin_1_layers_0_blocks_0_attn_qkv_bias', (288),
                                                                                                        'swin_1_layers_0_blocks_0_attn_proj_weight', (96, 96),
                                                                                                        'swin_1_layers_0_blocks_0_attn_proj_bias', (96),
                                                                                                        'swin_1_layers_0_blocks_0_norm2_weight', (96),
                                                                                                        'swin_1_layers_0_blocks_0_norm2_bias', (96),
                                                                                                        'swin_1_layers_0_blocks_0_mlp_fc1_weight', (384, 96),
                                                                                                        'swin_1_layers_0_blocks_0_mlp_fc1_bias', (384),
                                                                                                        'swin_1_layers_0_blocks_0_mlp_fc2_weight', (96, 384),
                                                                                                        'swin_1_layers_0_blocks_0_mlp_fc2_bias', (96),

                                                                                                        'swin_1_layers_0_blocks_1_norm1_weight', (96),
                                                                                                        'swin_1_layers_0_blocks_1_norm1_bias'  , (96),
                                                                                                        'swin_1_layers_0_blocks_1_attn_relative_position_bias_table', (169, 3),
                                                                                                        'swin_1_layers_0_blocks_1_attn_relative_position_index', (49, 49), 
                                                                                                        'swin_1_layers_0_blocks_1_attn_qkv_weight', (288, 96),
                                                                                                        'swin_1_layers_0_blocks_1_attn_qkv_bias', (288),
                                                                                                        'swin_1_layers_0_blocks_1_attn_proj_weight', (96, 96),
                                                                                                        'swin_1_layers_0_blocks_1_attn_proj_bias', (96),
                                                                                                        'swin_1_layers_0_blocks_1_norm2_weight', (96),
                                                                                                        'swin_1_layers_0_blocks_1_norm2_bias', (96),
                                                                                                        'swin_1_layers_0_blocks_1_mlp_fc1_weight', (384, 96),
                                                                                                        'swin_1_layers_0_blocks_1_mlp_fc1_bias', (384),
                                                                                                        'swin_1_layers_0_blocks_1_mlp_fc2_weight', (96, 384),
                                                                                                        'swin_1_layers_0_blocks_1_mlp_fc2_bias', (96),

                                                                                                        'swin_1_layers_0_downsample_reduction_weight', (192, 384),
                                                                                                        'swin_1_layers_0_downsample_norm_weight', (384),
                                                                                                        'swin_1_layers_0_downsample_norm_bias', (384)
                                                                                                        )

    Stage1_x, Stage1_H, Stage1_W , Stage1_down_x, Stage1_down_H, Stage1_down_W = basic_layer.BasicLayer(Stage0_down_x, Stage0_down_H, Stage0_down_W, 6,
                                                                                                        'swin_1_layers_1_blocks_0_norm1_weight', (192),
                                                                                                        'swin_1_layers_1_blocks_0_norm1_bias'  , (192),
                                                                                                        'swin_1_layers_1_blocks_0_attn_relative_position_bias_table', (169, 6),
                                                                                                        'swin_1_layers_1_blocks_0_attn_relative_position_index', (49, 49), 
                                                                                                        'swin_1_layers_1_blocks_0_attn_qkv_weight', (576, 192),
                                                                                                        'swin_1_layers_1_blocks_0_attn_qkv_bias', (576),
                                                                                                        'swin_1_layers_1_blocks_0_attn_proj_weight', (192, 192),
                                                                                                        'swin_1_layers_1_blocks_0_attn_proj_bias', (192),
                                                                                                        'swin_1_layers_1_blocks_0_norm2_weight', (192),
                                                                                                        'swin_1_layers_1_blocks_0_norm2_bias', (192),
                                                                                                        'swin_1_layers_1_blocks_0_mlp_fc1_weight', (768, 192),
                                                                                                        'swin_1_layers_1_blocks_0_mlp_fc1_bias', (768),
                                                                                                        'swin_1_layers_1_blocks_0_mlp_fc2_weight', (192, 768),
                                                                                                        'swin_1_layers_1_blocks_0_mlp_fc2_bias', (192),

                                                                                                        'swin_1_layers_1_blocks_1_norm1_weight', (192),
                                                                                                        'swin_1_layers_1_blocks_1_norm1_bias'  , (192),
                                                                                                        'swin_1_layers_1_blocks_1_attn_relative_position_bias_table', (169, 6),
                                                                                                        'swin_1_layers_1_blocks_1_attn_relative_position_index', (49, 49), 
                                                                                                        'swin_1_layers_1_blocks_1_attn_qkv_weight', (576, 192),
                                                                                                        'swin_1_layers_1_blocks_1_attn_qkv_bias', (576),
                                                                                                        'swin_1_layers_1_blocks_1_attn_proj_weight', (192, 192),
                                                                                                        'swin_1_layers_1_blocks_1_attn_proj_bias', (192),
                                                                                                        'swin_1_layers_1_blocks_1_norm2_weight', (192),
                                                                                                        'swin_1_layers_1_blocks_1_norm2_bias', (192),
                                                                                                        'swin_1_layers_1_blocks_1_mlp_fc1_weight', (768, 192),
                                                                                                        'swin_1_layers_1_blocks_1_mlp_fc1_bias', (768),
                                                                                                        'swin_1_layers_1_blocks_1_mlp_fc2_weight', (192, 768),
                                                                                                        'swin_1_layers_1_blocks_1_mlp_fc2_bias', (192),

                                                                                                        'swin_1_layers_1_downsample_reduction_weight', (384, 768),
                                                                                                        'swin_1_layers_1_downsample_norm_weight', (768),
                                                                                                        'swin_1_layers_1_downsample_norm_bias', (768)
                                                                                                        )

    Stage2_x, Stage2_H, Stage2_W , Stage2_down_x, Stage2_down_H, Stage2_down_W = basic_layer.BasicLayer(Stage1_down_x, Stage1_down_H, Stage1_down_W, 12,
                                                                                                        'swin_1_layers_2_blocks_0_norm1_weight', (384),
                                                                                                        'swin_1_layers_2_blocks_0_norm1_bias'  , (384),
                                                                                                        'swin_1_layers_2_blocks_0_attn_relative_position_bias_table', (169, 12),
                                                                                                        'swin_1_layers_2_blocks_0_attn_relative_position_index', (49, 49), 
                                                                                                        'swin_1_layers_2_blocks_0_attn_qkv_weight', (1152, 384),
                                                                                                        'swin_1_layers_2_blocks_0_attn_qkv_bias', (1152),
                                                                                                        'swin_1_layers_2_blocks_0_attn_proj_weight', (384, 384),
                                                                                                        'swin_1_layers_2_blocks_0_attn_proj_bias', (384),
                                                                                                        'swin_1_layers_2_blocks_0_norm2_weight', (384),
                                                                                                        'swin_1_layers_2_blocks_0_norm2_bias', (384),
                                                                                                        'swin_1_layers_2_blocks_0_mlp_fc1_weight', (1536, 384),
                                                                                                        'swin_1_layers_2_blocks_0_mlp_fc1_bias', (1536),
                                                                                                        'swin_1_layers_2_blocks_0_mlp_fc2_weight', (384, 1536),
                                                                                                        'swin_1_layers_2_blocks_0_mlp_fc2_bias', (384),

                                                                                                        'swin_1_layers_2_blocks_1_norm1_weight', (384),
                                                                                                        'swin_1_layers_2_blocks_1_norm1_bias'  , (384),
                                                                                                        'swin_1_layers_2_blocks_1_attn_relative_position_bias_table', (169, 12),
                                                                                                        'swin_1_layers_2_blocks_1_attn_relative_position_index', (49, 49), 
                                                                                                        'swin_1_layers_2_blocks_1_attn_qkv_weight', (1152, 384),
                                                                                                        'swin_1_layers_2_blocks_1_attn_qkv_bias', (1152),
                                                                                                        'swin_1_layers_2_blocks_1_attn_proj_weight', (384, 384),
                                                                                                        'swin_1_layers_2_blocks_1_attn_proj_bias', (384),
                                                                                                        'swin_1_layers_2_blocks_1_norm2_weight', (384),
                                                                                                        'swin_1_layers_2_blocks_1_norm2_bias', (384),
                                                                                                        'swin_1_layers_2_blocks_1_mlp_fc1_weight', (1536, 384),
                                                                                                        'swin_1_layers_2_blocks_1_mlp_fc1_bias', (1536),
                                                                                                        'swin_1_layers_2_blocks_1_mlp_fc2_weight', (384, 1536),
                                                                                                        'swin_1_layers_2_blocks_1_mlp_fc2_bias', (384),
                                                                                                        )

    # inout.compare_arrays('Swin_Stage0_x',Stage0_x, inout.load_ans_from_npy('Swin_block/layer_0_x_out',(1, 66304, 96)))
    # inout.compare_arrays('Swin_Stage0_down_x',Stage0_down_x, inout.load_ans_from_npy('Swin_block/layer_0_x',(1, 16576, 192)))
    # inout.compare_arrays('Swin_Stage1_x',Stage1_x, inout.load_ans_from_npy('Swin_block/layer_1_x_out',(1, 16576, 192)))
    # inout.compare_arrays('Swin_Stage1_down_x',Stage1_down_x, inout.load_ans_from_npy('Swin_block/layer_1_x',(1, 4144, 384)))
    # inout.compare_arrays('Swin_Stage2_x',Stage2_x, inout.load_ans_from_npy('Swin_block/layer_2_x_out',(1, 4144, 384)))
    # inout.compare_arrays('Swin_Stage2_down_x',Stage2_down_x, inout.load_ans_from_npy('Swin_block/layer_2_x',(1, 4144, 384)))
    norm0_weight = inout.load_weight_from_txt('swin_1_norm0_weight',(96))
    norm0_bias   = inout.load_weight_from_txt('swin_1_norm0_bias',(96))
    norm1_weight = inout.load_weight_from_txt('swin_1_norm1_weight',(192))
    norm1_bias   = inout.load_weight_from_txt('swin_1_norm1_bias',(192))
    norm2_weight = inout.load_weight_from_txt('swin_1_norm2_weight',(384))
    norm2_bias   = inout.load_weight_from_txt('swin_1_norm2_bias',(384))

    layer_norm_Stage0_x = layer_norm(Stage0_x, norm0_weight, norm0_bias)
    layer_norm_Stage1_x = layer_norm(Stage1_x, norm1_weight, norm1_bias)
    layer_norm_Stage2_x = layer_norm(Stage2_x, norm2_weight, norm2_bias)
    
    reshape_Stage0_x = layer_norm_Stage0_x.reshape(Stage0_H, Stage0_W,  96).transpose(2, 0, 1)
    reshape_Stage1_x = layer_norm_Stage1_x.reshape(Stage1_H, Stage1_W, 192).transpose(2, 0, 1)
    reshape_Stage2_x = layer_norm_Stage2_x.reshape(Stage2_H, Stage2_W, 384).transpose(2, 0, 1)

    return reshape_Stage0_x, reshape_Stage1_x, reshape_Stage2_x