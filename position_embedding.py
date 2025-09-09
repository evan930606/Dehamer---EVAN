import numpy as np
import inout
import math
from numpy.lib.stride_tricks import as_strided

def dark_channal(x: np.ndarray, window_size: int = 15) -> np.ndarray:
    C, H, W = x.shape
    # pad -------
    pad = window_size // 2
    x_pad = np.pad(x, pad_width=((0, 0), (pad, pad),(pad, pad)), mode='edge')
    # out -------
    result = np.zeros((H,W))

    for i in range(H):
        for j in range(W):
            window = x_pad[:, i:i+window_size, j:j+window_size]
            result[i, j] = window.min()

    return result 

def PatchEmbed(x: np.ndarray, 
               weight: np.ndarray,
               bias: np.ndarray,
               norm_weight: np.ndarray,
               norm_bias: np.ndarray):
    
    patch_size = 2
    embed_dim = 96
    eps = 1e-5  
    C, H, W = x.shape
    pad_h = patch_size - (H % patch_size) if H % patch_size != 0 else 0
    pad_w = patch_size - (W % patch_size) if W % patch_size != 0 else 0

    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    _, H_padded, W_padded = x.shape
    Wh = H_padded // patch_size
    Ww = W_padded // patch_size

    strides = (x.strides[0], x.strides[1] * patch_size, x.strides[2] * patch_size, x.strides[1], x.strides[2])
    patches = as_strided(x, shape=(C, Wh, Ww, patch_size, patch_size), strides=strides)

    proj = np.einsum('chwko,dcko->dhw', patches, weight) + bias[:, None, None]
    flattened = proj.reshape(embed_dim, Wh * Ww).transpose(1, 0)
    mean = np.mean(flattened, axis=-1, keepdims=True)
    var = np.var(flattened, axis=-1, keepdims=True)
    normalized = ((flattened - mean) / np.sqrt(var + eps)) * norm_weight[None, :] + norm_bias[None, :]
    result = normalized.transpose(1, 0).reshape(embed_dim, Wh, Ww)

    return result

def bicubic_kernel(x):
    x = np.abs(x)
    if x <= 1:
        return 1.25 * x**3 - 2.25 * x**2 + 1
    elif x < 2:
        return -0.75 * x**3 + 3.75 * x**2 - 6 * x + 3
    else:
        return 0

def interpolate(depth_map: np.ndarray) -> np.ndarray:
    depth_map = depth_map.astype(np.float32)

    h, w = depth_map.shape
    Wh = int(h / 2)
    Ww = int(w / 2)
    output = np.zeros((Wh, Ww), dtype=np.float32)
    
    for i in range(Wh):
        for j in range(Ww):
            iy = i * 2
            ix = j * 2
            
            weights_x = np.array([-0.09375, 0.59375, 0.59375, -0.09375])
            weights_y = np.array([-0.09375, 0.59375, 0.59375, -0.09375])
            
            patch = np.zeros((4, 4), dtype=np.float32)
            for my in range(4):
                for mx in range(4):
                    py = min(max(iy + my - 1, 0), h - 1)
                    px = min(max(ix + mx - 1, 0), w - 1)
                    patch[my, mx] = depth_map[py, px]
            
            value = np.dot(weights_y, np.dot(patch, weights_x))
            output[i, j] = value
    
    return output

def positional_encoding(x: np.ndarray, depth: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    h_d, w_d = depth.shape
    assert h == h_d and w == w_d, "Input and depth dimensions must match."

    num_pos_feats_x = 32
    num_pos_feats_y = 32
    num_pos_feats_z = 32
    num_pos_feats = max(num_pos_feats_x, num_pos_feats_y, num_pos_feats_z)
    temperature = 10000
    normalize = True
    scale = 2 * math.pi
    eps = 1e-6

    y_embed = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    x_embed = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    z_embed = depth.astype(np.float32)

    if normalize:
        y_embed = y_embed / (np.max(y_embed) + eps) * scale
        x_embed = x_embed / (np.max(x_embed) + eps) * scale
        z_embed_max = np.max(z_embed.reshape(-1))
        z_embed = z_embed / (z_embed_max + eps) * scale

    dim_t = np.arange(num_pos_feats, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t[:num_pos_feats_x]
    pos_y = y_embed[:, :, None] / dim_t[:num_pos_feats_y]
    pos_z = z_embed[:, :, None] / dim_t[:num_pos_feats_z]

    pos_x = np.stack((np.sin(pos_x[:, :, 0::2]), np.cos(pos_x[:, :, 1::2])), axis=3).reshape(h, w, num_pos_feats_x)
    pos_y = np.stack((np.sin(pos_y[:, :, 0::2]), np.cos(pos_y[:, :, 1::2])), axis=3).reshape(h, w, num_pos_feats_y)
    pos_z = np.stack((np.sin(pos_z[:, :, 0::2]), np.cos(pos_z[:, :, 1::2])), axis=3).reshape(h, w, num_pos_feats_z)

    pos = np.concatenate((pos_x, pos_y, pos_z), axis=2)
    pos = pos.transpose(2, 0, 1)

    return pos

def reconstruct(x: np.ndarray, absolute_pos_embed: np.ndarray) -> np.ndarray:
    x = x + absolute_pos_embed
    original_shape = x.shape
    flattened_dim = np.prod(original_shape[1:])
    x = x.reshape(original_shape[0], flattened_dim)
    x = np.transpose(x, axes=(1, 0))
    return x

def position_embedding_3D(x: np.ndarray) -> np.ndarray:
    x_dcp              = dark_channal(x)
    x_dcp_down         = interpolate(x_dcp)
    pat_emb_weight     = inout.load_weight_from_txt('swin_1_patch_embed_proj_weight', (96, 3, 2, 2))
    pat_emb_bias       = inout.load_weight_from_txt('swin_1_patch_embed_proj_bias', (96))
    lay_pat_emb_weight = inout.load_weight_from_txt('swin_1_patch_embed_norm_weight', (96))
    lay_pat_emb_bias   = inout.load_weight_from_txt('swin_1_patch_embed_norm_bias', (96))
    x_pat_emb          = PatchEmbed(x, pat_emb_weight, pat_emb_bias, lay_pat_emb_weight,lay_pat_emb_bias)
    x_pos_enc          = positional_encoding(x_pat_emb, x_dcp_down)
    x_add_pos_enc      = reconstruct(x_pat_emb, x_pos_enc)
    result = x_pat_emb + x_pos_enc
    result = np.transpose(result, (1, 2, 0))

    # #DeBug
    # inout.compare_arrays('darkchannal',x_dcp, np.squeeze(inout.load_ans_from_npy('pos_embed/depth_map',(1, 1, 592, 448)),axis=0))
    # inout.compare_arrays('darkchannal_pool',x_dcp_down, np.squeeze(inout.load_ans_from_npy('pos_embed/depth_pool',(1, 1, 296, 224)),axis=0))
    # inout.compare_arrays('PatchEmbed',x_pat_emb, inout.load_ans_from_npy('pos_embed/patch_embed',(1, 96, 296, 224)))
    # inout.compare_arrays('Positional_Encoding',x_pos_enc, inout.load_ans_from_npy('pos_embed/absolute_pos_embed',(1, 96, 296, 224)))
    # inout.compare_arrays('Add_Positional_Encoding',x_add_pos_enc, inout.load_ans_from_npy('pos_embed/add_absolute_pos_embed',(1, 66304, 96)))
    # inout.compare_arrays('Pos_Drop',x_add_pos_enc, inout.load_ans_from_npy('pos_embed/pos_drop',(1, 66304, 96)))
    C, H ,W = x_pos_enc.shape
    return  H, W, x_add_pos_enc # output shape (H * W, C)