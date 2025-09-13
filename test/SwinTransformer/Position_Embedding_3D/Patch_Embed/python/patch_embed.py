import numpy as np
import inout_test
import error
from numpy.lib.stride_tricks import as_strided

def PatchEmbed_verilog(x, weight, bias, norm_weight, norm_bias):
    C, H, W = x.shape
    o_H = H // 2
    o_W = W // 2
    result = np.zeros((o_H, o_W, 96))
    for w in range(o_W):
        for h in range(o_H):
            oh = h * 2
            ow = w * 2
            # Conv2d
            # Y (96 x 1) = W (96 x 12) * X (12 x 1) + b (96 x 1)
            result[h][w] = np.dot(weight, x[:, oh:oh+2, ow:ow+2].reshape(-1)) + bias
            # Normlayer
            mean = np.mean(result[h][w])
            var = np.var(result[h][w])
            result[h][w] = ((result[h][w] - mean) / np.sqrt(var + 1e-5)) * norm_weight + norm_bias

    return result.transpose(2, 0, 1)


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

if __name__ == '__main__':
    # input
    input         = inout_test.txt_to_numpy_array('../data/input_3_480_640.txt', (3, 480, 640))

    # weight
    pat_emb_weight     = inout_test.load_weight_from_npy('swin_1_patch_embed_proj_weight', (96, 3, 2, 2)) 
    pat_emb_bias       = inout_test.load_weight_from_npy('swin_1_patch_embed_proj_bias', (96))
    lay_pat_emb_weight = inout_test.load_weight_from_npy('swin_1_patch_embed_norm_weight', (96))
    lay_pat_emb_bias   = inout_test.load_weight_from_npy('swin_1_patch_embed_norm_bias', (96))

    output        = PatchEmbed_verilog(input, pat_emb_weight.reshape(96, -1), pat_emb_bias, lay_pat_emb_weight, lay_pat_emb_bias)
    ans           = PatchEmbed(input, pat_emb_weight, pat_emb_bias, lay_pat_emb_weight, lay_pat_emb_bias)

    inout_test.numpy_array_to_txt(output ,'../data/output')
    error.compare_numpy_arrays(output, ans, '../data/error.txt')