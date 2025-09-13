import numpy as np
import inout_test
import math
import error

def positional_encoding_verilog(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    H, W = z.shape
    z    = (z / np.max(z) * 2 * math.pi)
    result = np.zeros((H, W, 32))
    for h in range(H):
        for w in range(W):
            result[h, w, 0 ] = np.sin(z[h, w])
            result[h, w, 1 ] = np.cos(z[h, w])
            result[h, w, 2 ] = np.sin(z[h, w] * 0.56234132)
            result[h, w, 3 ] = np.cos(z[h, w] * 0.56234132)
            result[h, w, 4 ] = np.sin(z[h, w] * 0.31622777)
            result[h, w, 5 ] = np.cos(z[h, w] * 0.31622777)
            result[h, w, 6 ] = np.sin(z[h, w] * 0.17782794)
            result[h, w, 7 ] = np.cos(z[h, w] * 0.17782794)
            result[h, w, 8 ] = np.sin(z[h, w] * 0.1)
            result[h, w, 9 ] = np.cos(z[h, w] * 0.1)
            result[h, w, 10] = np.sin(z[h, w] * 0.056234132)
            result[h, w, 11] = np.cos(z[h, w] * 0.056234132)
            result[h, w, 12] = np.sin(z[h, w] * 0.031622777)
            result[h, w, 13] = np.cos(z[h, w] * 0.031622777)
            result[h, w, 14] = np.sin(z[h, w] * 0.017782794)
            result[h, w, 15] = np.cos(z[h, w] * 0.017782794)
            result[h, w, 16] = np.sin(z[h, w] * 0.01)
            result[h, w, 17] = np.cos(z[h, w] * 0.01)
            result[h, w, 18] = np.sin(z[h, w] * 0.0056234132)
            result[h, w, 19] = np.cos(z[h, w] * 0.0056234132)
            result[h, w, 20] = np.sin(z[h, w] * 0.0031622777)
            result[h, w, 21] = np.cos(z[h, w] * 0.0031622777)
            result[h, w, 22] = np.sin(z[h, w] * 0.0017782794)
            result[h, w, 23] = np.cos(z[h, w] * 0.0017782794)
            result[h, w, 24] = np.sin(z[h, w] * 0.001)
            result[h, w, 25] = np.cos(z[h, w] * 0.001)
            result[h, w, 26] = np.sin(z[h, w] * 0.00056234132)
            result[h, w, 27] = np.cos(z[h, w] * 0.00056234132)
            result[h, w, 28] = np.sin(z[h, w] * 0.00031622777)
            result[h, w, 29] = np.cos(z[h, w] * 0.00031622777)
            result[h, w, 30] = np.sin(z[h, w] * 0.00017782794)
            result[h, w, 31] = np.cos(z[h, w] * 0.00017782794)
    result = np.concatenate((x, y, result), axis=2)
    return result.transpose(2, 0, 1)


def positional_encoding(depth: np.ndarray) -> np.ndarray:
    c, h, w = 96, 240, 320
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

    return pos_x, pos_y, pos

if __name__ == '__main__':
    # input
    input         = inout_test.txt_to_numpy_array('../data/input_1_240_320.txt', (1, 240, 320))
    input         = np.squeeze(input)

    x, y, ans     = positional_encoding(input)
    output        = positional_encoding_verilog(x, y, input)

    inout_test.numpy_array_to_txt(output ,'../data/output')
    error.compare_numpy_arrays(output, ans, '../data/error.txt')