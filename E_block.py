import numpy as np
import math
import inout
from numpy.lib.stride_tricks import as_strided

def Conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None, padding: int = 1,stride: int = 1) -> np.ndarray:
    C, H, W = x.shape
    out_chans, _, kernel_size, _ = weight.shape
    x_padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)

    out_h = (H + 2 * padding - kernel_size) // stride + 1
    out_w = (W + 2 * padding - kernel_size) // stride + 1

    strides = (x_padded.strides[0], x_padded.strides[1] * stride, x_padded.strides[2] * stride, x_padded.strides[1], x_padded.strides[2])
    patches = as_strided(x_padded, shape=(C, out_h, out_w, kernel_size, kernel_size), strides=strides)
    out = np.einsum('chwko,dcko->dhw', patches, weight)
    if bias is not None:
        out += bias[:, None, None]
    return out

def ReLU(x: np.ndarray) -> np.ndarray:
    np.maximum(x, 0, out=x)
    return x

def MaxPool2d(x: np.ndarray) -> np.ndarray:
    C, H, W = x.shape
    kernel_size = 2
    stride = 2
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1
    strides = (x.strides[0], x.strides[1] * stride, x.strides[2] * stride, x.strides[1], x.strides[2])
    patches = as_strided(x, shape=(C, out_h, out_w, kernel_size, kernel_size), strides=strides)
    out = np.max(patches, axis=(3, 4))

    return out

def E_block(x: np.ndarray,
            weight_0_name: str, weight_0_shape, 
            bias_0_name  : str, bias_0_shape  ,
            weight_2_name: str, weight_2_shape,
            bias_2_name  : str, bias_2_shape,
            ):
    E_block_0_weight = inout.load_weight_from_txt(weight_0_name, weight_0_shape)
    E_block_0_bias   = inout.load_weight_from_txt(bias_0_name, bias_0_shape)
    E_block_2_weight = inout.load_weight_from_txt(weight_2_name, weight_2_shape)
    E_block_2_bias   = inout.load_weight_from_txt(bias_2_name,bias_2_shape)
    E_block_0 = Conv2d(x, E_block_0_weight , E_block_0_bias)
    E_block_1 = ReLU(E_block_0)
    E_block_2 = Conv2d(E_block_1, E_block_2_weight , E_block_2_bias)
    E_block_3 = ReLU(E_block_2)
    E_block_4 = MaxPool2d(E_block_3)
    return E_block_4

def AdaptiveAvgPool2d(x: np.ndarray, bin: int):
    C, H, W = x.shape
    out = np.empty((C, bin, bin), dtype=x.dtype)

    for i in range(bin):
        h0 = int(math.floor(i * H / bin))
        h1 = int(math.ceil((i + 1) * H / bin))
        if h1 <= h0:
            h1 = min(h0 + 1, H)

        for j in range(bin):
            w0 = int(math.floor(j * W / bin))
            w1 = int(math.ceil((j + 1) * W / bin))
            if w1 <= w0:
                w1 = min(w0 + 1, W)

            window = x[:, h0:h1, w0:w1]
            out[:, i, j] = window.mean(axis=(1, 2))

    return out

def bilinear_interpolate(x: np.ndarray, size: tuple) -> np.ndarray:
    C, H_in, W_in = x.shape
    H_out, W_out = size

    scale_h = (H_in - 1) / (H_out - 1) if H_out > 1 else 0
    scale_w = (W_in - 1) / (W_out - 1) if W_out > 1 else 0

    y_out, x_out = np.meshgrid(np.arange(H_out), np.arange(W_out), indexing='ij')
    y_out = y_out.astype(np.float32)
    x_out = x_out.astype(np.float32)

    y_in = y_out * scale_h
    x_in = x_out * scale_w

    y_in = np.clip(y_in, 0, H_in - 1)
    x_in = np.clip(x_in, 0, W_in - 1)

    y0 = np.floor(y_in).astype(int)
    y1 = y0 + 1
    x0 = np.floor(x_in).astype(int)
    x1 = x0 + 1

    y1 = np.clip(y1, 0, H_in - 1)
    x1 = np.clip(x1, 0, W_in - 1)

    dy = y_in - y0
    dx = x_in - x0

    output = np.zeros((C, H_out, W_out), dtype=np.float32)
    for c in range(C):
        Ia = x[c, y0, x0]
        Ib = x[c, y1, x0]
        Ic = x[c, y0, x1]
        Id = x[c, y1, x1]

        wa = (1 - dx) * (1 - dy)
        wb = (1 - dx) * dy
        wc = dx * (1 - dy)
        wd = dx * dy

        output[c] = Ia * wa + Ib * wb + Ic * wc + Id * wd

    return output

def PPM_branch(x: np.ndarray, weight: np.ndarray, dim, bin):
    C, H, W = x.shape
    pooled = AdaptiveAvgPool2d(x, bin)
    conved = Conv2d(pooled, weight, None, 0, 1)
    activated = ReLU(conved)
    interpolated = bilinear_interpolate(activated, (H, W))

    return interpolated

def PPM(x :np.ndarray, dim: int,
        weight_0_name: str, weight_0_shape,
        weight_1_name: str, weight_1_shape,
        weight_2_name: str, weight_2_shape,
        weight_3_name: str, weight_3_shape
        ):
    weight_0 = inout.load_weight_from_txt(weight_0_name, weight_0_shape)
    weight_1 = inout.load_weight_from_txt(weight_1_name, weight_1_shape)
    weight_2 = inout.load_weight_from_txt(weight_2_name, weight_2_shape)
    weight_3 = inout.load_weight_from_txt(weight_3_name, weight_3_shape)
    PPM_branch0 = PPM_branch(x, weight_0, dim, 1)
    PPM_branch1 = PPM_branch(x, weight_1, dim, 2)
    PPM_branch2 = PPM_branch(x, weight_2, dim, 3)
    PPM_branch3 = PPM_branch(x, weight_3, dim, 4)
    result = np.concatenate((x, PPM_branch0, PPM_branch1, PPM_branch2, PPM_branch3), axis=0)
    return result


def Encoder(x: np.ndarray):
    E_block1 = E_block(x,
                       'E_block1_0_weight', (32, 3, 3, 3),
                       'E_block1_0_bias'  , (32),
                       'E_block1_2_weight', (32, 32, 3, 3),
                       'E_block1_2_bias'  , (32))

    PPM1 = PPM(E_block1, 8,
               'PPM1_features_0_1_weight', (8, 32, 1, 1),
               'PPM1_features_1_1_weight', (8, 32, 1, 1),
               'PPM1_features_2_1_weight', (8, 32, 1, 1),
               'PPM1_features_3_1_weight', (8, 32, 1, 1))
    
    E_block2 = E_block(PPM1,
                       'E_block2_0_weight', (64, 64, 3, 3),
                       'E_block2_0_bias'  , (64),
                       'E_block2_2_weight', (64, 64, 3, 3),
                       'E_block2_2_bias'  , (64))

    PPM2 = PPM(E_block2, 16,
               'PPM2_features_0_1_weight', (16, 64, 1, 1),
               'PPM2_features_1_1_weight', (16, 64, 1, 1),
               'PPM2_features_2_1_weight', (16, 64, 1, 1),
               'PPM2_features_3_1_weight', (16, 64, 1, 1))

    E_block3 = E_block(PPM2,
                       'E_block3_0_weight', (128, 128, 3, 3),
                       'E_block3_0_bias'  , (128),
                       'E_block3_2_weight', (128, 128, 3, 3),
                       'E_block3_2_bias'  , (128))

    PPM3 = PPM(E_block3, 32,
               'PPM3_features_0_1_weight', (32, 128, 1, 1),
               'PPM3_features_1_1_weight', (32, 128, 1, 1),
               'PPM3_features_2_1_weight', (32, 128, 1, 1),
               'PPM3_features_3_1_weight', (32, 128, 1, 1))

    E_block4 = E_block(PPM3,
                       'E_block4_0_weight', (256, 256, 3, 3),
                       'E_block4_0_bias'  , (256),
                       'E_block4_2_weight', (256, 256, 3, 3),
                       'E_block4_2_bias'  , (256))

    PPM4 = PPM(E_block4, 64,
               'PPM4_features_0_1_weight', (64, 256, 1, 1),
               'PPM4_features_1_1_weight', (64, 256, 1, 1),
               'PPM4_features_2_1_weight', (64, 256, 1, 1),
               'PPM4_features_3_1_weight', (64, 256, 1, 1))

    # inout.compare_arrays('E_block1', E_block1, inout.load_ans_from_npy('E_block1',(1, 32, 296, 224)))
    # inout.compare_arrays('PPM1', PPM1, inout.load_ans_from_npy('PPM1',(1, 64, 296, 224)))
    # inout.compare_arrays('E_block2', E_block2, inout.load_ans_from_npy('E_block2',(1, 64, 148, 112)))
    # inout.compare_arrays('PPM2', PPM2, inout.load_ans_from_npy('PPM2',(1, 128, 148, 112)))
    # inout.compare_arrays('E_block3', E_block3, inout.load_ans_from_npy('E_block3',(1, 128,  74,  56)))
    # inout.compare_arrays('PPM3', PPM3, inout.load_ans_from_npy('PPM3',(1, 256, 74, 56)))
    # inout.compare_arrays('E_block4', E_block4, inout.load_ans_from_npy('E_block4',(1, 256,  37,  28)))
    # inout.compare_arrays('PPM4', PPM4, inout.load_ans_from_npy('PPM4',(1, 512, 37, 28)))

    return PPM1, PPM2, PPM3, PPM4
