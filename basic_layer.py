import inout
import numpy as np
import math


def layer_norm(x: np.ndarray, weight: np.ndarray , bias: np.ndarray , eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * weight + bias

def window_partition(x, window_size):
    H, W, C = x.shape
    num_h = H // window_size
    num_w = W // window_size
    
    x = x.reshape(num_h, window_size, num_w, window_size, C)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    windows = x.reshape(num_h * num_w, window_size, window_size, C)
    x_windows = windows.reshape(num_h * num_w, window_size * window_size, C)
    
    return x_windows

def softmax(x, axis=-1):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def WindowAttention(x: np.ndarray,mask: np.ndarray, num_heads: int,
                    qkv_weight: np.ndarray, qkv_bias: np.ndarray, 
                    proj_weight: np.ndarray, proj_bias: np.ndarray, 
                    relative_position_bias_table: np.ndarray, relative_position_index: np.ndarray):

    num_windows, N, dim = x.shape
    head_dim = dim // num_heads
    scale = head_dim ** -0.5

    qkv = np.dot(x, qkv_weight.T) + qkv_bias
    qkv = qkv.reshape(num_windows, N, 3, num_heads, head_dim)
    qkv = np.transpose(qkv, (2, 0, 3, 1, 4))  
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * scale
    attn = np.matmul(q, np.transpose(k, (0, 1, 3, 2)))  

    relative_position_bias = relative_position_bias_table[relative_position_index.ravel()]
    relative_position_bias = relative_position_bias.reshape(N, N, num_heads)
    relative_position_bias = np.transpose(relative_position_bias, (2, 0, 1))
    attn = attn + np.expand_dims(relative_position_bias, axis=0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.reshape(num_windows // nW, nW, num_heads, N, N) + np.expand_dims(np.expand_dims(mask, axis=1), axis=0)
        attn = attn.reshape(-1, num_heads, N, N)
    attn = softmax(attn)

    weighted_v = np.matmul(attn, v) 
    weighted_v = np.transpose(weighted_v, (0, 2, 1, 3))
    weighted_v = weighted_v.reshape(num_windows, N, dim)
    out = np.dot(weighted_v, proj_weight.T) + proj_bias

    return out

def window_reverse(windows, window_size, H, W):
    num_h = H // window_size
    num_w = W // window_size
    x = windows.reshape(num_h, num_w, window_size, window_size, -1)
    x = x.transpose(0, 2, 1, 3, 4)
    x = x.reshape(H, W, -1)
    return x

def gelu(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)  
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))

def Mlp(x: np.ndarray, fc1_weight: np.ndarray, fc1_bias: np.ndarray, fc2_weight: np.ndarray, fc2_bias: np.ndarray) -> np.ndarray:
    x = np.dot(x, fc1_weight.T) + fc1_bias
    x = gelu(x)
    x = np.dot(x, fc2_weight.T) + fc2_bias
    return x

def swin_transformer_block(x: np.ndarray, H: int, W: int, window_size: int, num_heads: int, shift_size: int, 
                           attn_mask: np.ndarray,
                           norm1_weight_name        : str, norm1_weight_shape,
                           norm1_bias_name          : str, norm1_bias_shape,  
                           attn_rel_bias_table_name : str, attn_rel_bias_table_shape,
                           attn_rel_index_name      : str, attn_rel_index_shape, 
                           attn_qkv_weight_name     : str, attn_qkv_weight_shape,
                           attn_qkv_bias_name       : str, attn_qkv_bias_shape,
                           attn_proj_weight_name    : str, attn_proj_weight_shape,
                           attn_proj_bias_name      : str, attn_proj_bias_shape,
                           norm2_weight_name        : str, norm2_weight_shape,
                           norm2_bias_name          : str, norm2_bias_shape,  
                           mlp_fc1_weight_name      : str, mlp_fc1_weight_shape,
                           mlp_fc1_bias_name        : str, mlp_fc1_bias_shape,
                           mlp_fc2_weight_name      : str, mlp_fc2_weight_shape,
                           mlp_fc2_bias_name        : str, mlp_fc2_bias_shape,
                           ):

    norm1_weight        = inout.load_weight_from_txt(norm1_weight_name, norm1_weight_shape)
    norm1_bias          = inout.load_weight_from_txt(norm1_bias_name, norm1_bias_shape)
    layer_norm_1        = layer_norm(x, norm1_weight, norm1_bias)

    _, C = layer_norm_1.shape
    x_layer = layer_norm_1.reshape(H, W, C)
    pad_w = (window_size - W % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    x_pad = np.pad(x_layer, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    Hp, Wp, Wc = x_pad.shape

    if shift_size > 0:
        shifted_x = np.roll(x_pad, shift=(-shift_size, -shift_size), axis=(0, 1))
        block_mask = attn_mask
    else:
        shifted_x = x_pad
        block_mask = None

    x_windows = window_partition(shifted_x, window_size)

    attn_rel_bias_table = inout.load_weight_from_txt(attn_rel_bias_table_name, attn_rel_bias_table_shape)
    attn_rel_index      = inout.load_weight_from_txt(attn_rel_index_name, attn_rel_index_shape).astype(np.int64)
    attn_qkv_weight     = inout.load_weight_from_txt(attn_qkv_weight_name, attn_qkv_weight_shape)
    attn_qkv_bias       = inout.load_weight_from_txt(attn_qkv_bias_name, attn_qkv_bias_shape)
    attn_proj_weight    = inout.load_weight_from_txt(attn_proj_weight_name, attn_proj_weight_shape)
    attn_proj_bias      = inout.load_weight_from_txt(attn_proj_bias_name, attn_proj_bias_shape)
    attn_window         = WindowAttention(x_windows, block_mask, num_heads, 
                                          attn_qkv_weight, attn_qkv_bias,
                                          attn_proj_weight, attn_proj_bias,
                                          attn_rel_bias_table, attn_rel_index)


    attn_window = attn_window.reshape(-1, window_size, window_size, C)
    shifted_x = window_reverse(attn_window, window_size, Hp, Wp)

    if shift_size > 0:
        shifted_x = np.roll(shifted_x, shift=(shift_size, shift_size), axis=(0, 1))

    shifted_x = shifted_x[:H, :W, :].reshape(H * W, C)
    x_add_shift = x + shifted_x

    norm2_weight        = inout.load_weight_from_txt(norm2_weight_name, norm2_weight_shape)
    norm2_bias          = inout.load_weight_from_txt(norm2_bias_name, norm2_bias_shape)
    layer_norm_2        = layer_norm(x_add_shift, norm2_weight, norm2_bias)

    mlp_fc1_weight      = inout.load_weight_from_txt(mlp_fc1_weight_name, mlp_fc1_weight_shape)
    mlp_fc1_bias        = inout.load_weight_from_txt(mlp_fc1_bias_name,   mlp_fc1_bias_shape  )
    mlp_fc2_weight      = inout.load_weight_from_txt(mlp_fc2_weight_name, mlp_fc2_weight_shape)
    mlp_fc2_bias        = inout.load_weight_from_txt(mlp_fc2_bias_name,   mlp_fc2_bias_shape  )
    x_mlp = Mlp(layer_norm_2, mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias)

    result = x_add_shift + x_mlp

    # inout.compare_arrays('layer_norm',layer_norm_1, inout.load_ans_from_npy('Swin_block/layer_norm_96_0_0',(1, 66304, 96)))
    # inout.compare_arrays('window',x_windows, inout.load_ans_from_npy('Swin_block/windows_x_96_0',(1376, 49, 96)))
    # inout.compare_arrays('WindowAttention',attn_window, inout.load_ans_from_npy('Swin_block/attn_windows_96_0',(1376, 49, 96)))
    # inout.compare_arrays('layer_norm',layer_norm_2, inout.load_ans_from_npy('Swin_block/layer_norm_96_0_1',(1, 66304, 96)))
    # inout.compare_arrays('MLP',result, inout.load_ans_from_npy('Swin_block/MLP_96_0',(1, 66304, 96)))

    return result

def PatchMerging(x: np.ndarray, H: int, W: int, norm_weight: np.ndarray, norm_bias: np.ndarray, reduction_weight: np.ndarray) -> np.ndarray:
    dim = x.shape[-1]
    eps = 1e-5 

    x = x.reshape(H, W, dim)

    pad_h = 1 if H % 2 == 1 else 0
    pad_w = 1 if W % 2 == 1 else 0
    if pad_h or pad_w:
        x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    x0 = x[0::2, 0::2, :]  
    x1 = x[1::2, 0::2, :]  
    x2 = x[0::2, 1::2, :]  
    x3 = x[1::2, 1::2, :]  

    x = np.concatenate([x0, x1, x2, x3], axis=-1)  
    x = x.reshape(-1, 4 * dim)
    layer_norm_x = layer_norm(x, norm_weight, norm_bias)
    result = np.dot(layer_norm_x, reduction_weight.T)

    return result

def BasicLayer(x: np.ndarray, H: int, W: int, num_heads: int, 
               block0_norm1_weight_name        : str, block0_norm1_weight_shape,
               block0_norm1_bias_name          : str, block0_norm1_bias_shape,  
               block0_attn_rel_bias_table_name : str, block0_attn_rel_bias_table_shape,
               block0_attn_rel_index_name      : str, block0_attn_rel_index_shape, 
               block0_attn_qkv_weight_name     : str, block0_attn_qkv_weight_shape,
               block0_attn_qkv_bias_name       : str, block0_attn_qkv_bias_shape,
               block0_attn_proj_weight_name    : str, block0_attn_proj_weight_shape,
               block0_attn_proj_bias_name      : str, block0_attn_proj_bias_shape,
               block0_norm2_weight_name        : str, block0_norm2_weight_shape,
               block0_norm2_bias_name          : str, block0_norm2_bias_shape,  
               block0_mlp_fc1_weight_name      : str, block0_mlp_fc1_weight_shape,
               block0_mlp_fc1_bias_name        : str, block0_mlp_fc1_bias_shape,
               block0_mlp_fc2_weight_name      : str, block0_mlp_fc2_weight_shape,
               block0_mlp_fc2_bias_name        : str, block0_mlp_fc2_bias_shape,

               block1_norm1_weight_name        : str, block1_norm1_weight_shape,
               block1_norm1_bias_name          : str, block1_norm1_bias_shape,  
               block1_attn_rel_bias_table_name : str, block1_attn_rel_bias_table_shape,
               block1_attn_rel_index_name      : str, block1_attn_rel_index_shape, 
               block1_attn_qkv_weight_name     : str, block1_attn_qkv_weight_shape,
               block1_attn_qkv_bias_name       : str, block1_attn_qkv_bias_shape,
               block1_attn_proj_weight_name    : str, block1_attn_proj_weight_shape,
               block1_attn_proj_bias_name      : str, block1_attn_proj_bias_shape,
               block1_norm2_weight_name        : str, block1_norm2_weight_shape,
               block1_norm2_bias_name          : str, block1_norm2_bias_shape,  
               block1_mlp_fc1_weight_name      : str, block1_mlp_fc1_weight_shape,
               block1_mlp_fc1_bias_name        : str, block1_mlp_fc1_bias_shape,
               block1_mlp_fc2_weight_name      : str, block1_mlp_fc2_weight_shape,
               block1_mlp_fc2_bias_name        : str, block1_mlp_fc2_bias_shape,

               reduction_weight_name           : str = None, reduction_weight_shape = None, 
               norm_weight_name                : str = None, norm_weight_shape      = None,
               norm_bias_name                  : str = None, norm_bias_shape        = None,
               ):

    window_size = 7
    shift_size = window_size // 2

    Hp = math.ceil(H / window_size) * window_size
    Wp = math.ceil(W / window_size) * window_size
    img_mask = np.zeros((Hp, Wp, 1), dtype=np.float32)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.reshape(-1, window_size * window_size)
    attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    attn_mask = np.where(attn_mask != 0, float(-100.0), float(0.0))
    x_block0 = swin_transformer_block(x, H, W, window_size, num_heads, 0,attn_mask,
                                      block0_norm1_weight_name        , block0_norm1_weight_shape,
                                      block0_norm1_bias_name          , block0_norm1_bias_shape,  
                                      block0_attn_rel_bias_table_name , block0_attn_rel_bias_table_shape,
                                      block0_attn_rel_index_name      , block0_attn_rel_index_shape, 
                                      block0_attn_qkv_weight_name     , block0_attn_qkv_weight_shape,
                                      block0_attn_qkv_bias_name       , block0_attn_qkv_bias_shape,
                                      block0_attn_proj_weight_name    , block0_attn_proj_weight_shape,
                                      block0_attn_proj_bias_name      , block0_attn_proj_bias_shape,
                                      block0_norm2_weight_name        , block0_norm2_weight_shape,
                                      block0_norm2_bias_name          , block0_norm2_bias_shape,  
                                      block0_mlp_fc1_weight_name      , block0_mlp_fc1_weight_shape,
                                      block0_mlp_fc1_bias_name        , block0_mlp_fc1_bias_shape,
                                      block0_mlp_fc2_weight_name      , block0_mlp_fc2_weight_shape,
                                      block0_mlp_fc2_bias_name        , block0_mlp_fc2_bias_shape,
                                      )


    x_block1 = swin_transformer_block(x_block0, H, W, window_size, num_heads, 3,attn_mask,
                                      block1_norm1_weight_name        , block1_norm1_weight_shape,
                                      block1_norm1_bias_name          , block1_norm1_bias_shape,  
                                      block1_attn_rel_bias_table_name , block1_attn_rel_bias_table_shape,
                                      block1_attn_rel_index_name      , block1_attn_rel_index_shape, 
                                      block1_attn_qkv_weight_name     , block1_attn_qkv_weight_shape,
                                      block1_attn_qkv_bias_name       , block1_attn_qkv_bias_shape,
                                      block1_attn_proj_weight_name    , block1_attn_proj_weight_shape,
                                      block1_attn_proj_bias_name      , block1_attn_proj_bias_shape,
                                      block1_norm2_weight_name        , block1_norm2_weight_shape,
                                      block1_norm2_bias_name          , block1_norm2_bias_shape,  
                                      block1_mlp_fc1_weight_name      , block1_mlp_fc1_weight_shape,
                                      block1_mlp_fc1_bias_name        , block1_mlp_fc1_bias_shape,
                                      block1_mlp_fc2_weight_name      , block1_mlp_fc2_weight_shape,
                                      block1_mlp_fc2_bias_name        , block1_mlp_fc2_bias_shape,
                                      )

    if reduction_weight_name is not None:
        reduction_weight = inout.load_weight_from_txt(reduction_weight_name, reduction_weight_shape)
        norm_weight      = inout.load_weight_from_txt(norm_weight_name, norm_weight_shape)
        norm_bias        = inout.load_weight_from_txt(norm_bias_name, norm_bias_shape)
        x_patch_merging = PatchMerging(x_block1, H, W, norm_weight, norm_bias, reduction_weight)
        Wh, Ww = (H + 1) // 2, (W + 1) // 2
        return x_block1, H, W, x_patch_merging, Wh, Ww
    
    return x_block1, H, W, x_block1, H, W

    

    