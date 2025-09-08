import numpy as np 
import inout
import math
from numpy.lib.stride_tricks import as_strided

def ConvTranspose2d(x: np.ndarray, weight: np.ndarray, bias=None, stride=2, padding=1, output_padding=1):
    C_in, H, W = x.shape
    _, C_out, K_h, K_w = weight.shape
    S  = int(stride)
    P  = int(padding)
    OP = int(output_padding)

    # PyTorch 的輸出尺寸公式
    H_out = (H - 1) * S - 2 * P + K_h + OP
    W_out = (W - 1) * S - 2 * P + K_w + OP

    out = np.zeros((C_out, H_out, W_out), dtype=x.dtype)

    # 逐點散佈加總（與 PyTorch 一致的索引）
    for i in range(H):
        base_i = i * S - P
        for j in range(W):
            base_j = j * S - P
            # 取出這個輸入像素對所有輸出通道的權重切片： (C_out, K_h, K_w)
            # 與 val 相乘後散佈到 out
            for c_in in range(C_in):
                val = x[c_in, i, j]
                if val == 0:
                    continue  # 小優化：0 就不必加
                W_c = weight[c_in]  # 形狀 (C_out, K_h, K_w)
                # 對 kernel 每個 (kh,kw) 計算輸出座標
                for kh in range(K_h):
                    ii = base_i + kh
                    if ii < 0 or ii >= H_out:
                        continue
                    for kw in range(K_w):
                        jj = base_j + kw
                        if jj < 0 or jj >= W_out:
                            continue
                        # out[:, ii, jj] += val * W_c[:, kh, kw]
                        out[:, ii, jj] += W_c[:, kh, kw] * val

    if bias is not None:
        out += bias[:, None, None]

    return out

def upsample_bilinear(x:np.ndarray , scale_factor=2, align_corners=False):
    C, H_in, W_in = x.shape
    H_out = H_in * scale_factor
    W_out = W_in * scale_factor

    # ---- 對齊 PyTorch 的座標映射 ----
    # align_corners=False 時用比例映射 (yy+0.5)*(H_in/H_out)-0.5
    yy, xx = np.meshgrid(np.arange(H_out), np.arange(W_out), indexing='ij')
    in_y = (yy.astype(np.float64) + 0.5) * (H_in / float(H_out)) - 0.5
    in_x = (xx.astype(np.float64) + 0.5) * (W_in / float(W_out)) - 0.5

    i0 = np.floor(in_y).astype(np.int64)
    j0 = np.floor(in_x).astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1

    # 夾住到合法索引（關鍵：不丟棄，重疊時會在同一像素累加）
    i0c = np.clip(i0, 0, H_in - 1)
    i1c = np.clip(i1, 0, H_in - 1)
    j0c = np.clip(j0, 0, W_in - 1)
    j1c = np.clip(j1, 0, W_in - 1)

    di = in_y - i0
    dj = in_x - j0

    # 權重，四個權重總和恆為 1（含邊界重疊情形）
    wa = (1.0 - di) * (1.0 - dj)   # (i0 , j0)
    wb = (1.0 - di) * dj           # (i0 , j1)
    wc = di * (1.0 - dj)           # (i1 , j0)
    wd = di * dj                   # (i1 , j1)

    # 用 float64 計算再回寫，避免累加誤差
    out = (x[:, i0c, j0c] * wa[None, :, :] +
           x[:, i0c, j1c] * wb[None, :, :] +
           x[:, i1c, j0c] * wc[None, :, :] +
           x[:, i1c, j1c] * wd[None, :, :])

    return out.astype(x.dtype, copy=False)

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

def PReLU(x: np.ndarray, a):
    a = np.asarray(a, dtype=x.dtype)
    if a.size == 1: 
        a = float(a.reshape(-1)[0])
    return np.where(x >= 0, x, a * x)

def Sigmoid(x: np.ndarray):
    x = np.where(x >= 102, 102, x)
    x = np.where(x <= -87, -87, x)
    return 1 / (1 + np.exp(-x))

def Softmax(x: np.ndarray):
    m = x.max(axis=0, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=0, keepdims=True)

def Spatial_Attn_Layer(x: np.ndarray, weight: np.ndarray):
    max_pool    = np.max(x, axis=0, keepdims=True)
    avg_pool    = np.mean(x, axis=0, keepdims=True)
    concat_pool = np.concatenate((max_pool, avg_pool), axis=0)
    conv_x      = Conv2d(concat_pool, weight, None, 2, 1)
    sigmod_x    = Sigmoid(conv_x)
    return x * sigmod_x

def Ca_Layer(x: np.ndarray, weight0: np.ndarray, weight1: np.ndarray):
    x_avg = AdaptiveAvgPool2d(x, 1)
    x_conv_du_0 = Conv2d(x_avg, weight0, None, 0, 1)
    x_conv_du_1 = ReLU(x_conv_du_0)
    x_conv_du_2 = Conv2d(x_conv_du_1, weight1, None, 0, 1)
    x_conv_du_3 = Sigmoid(x_conv_du_2)
    return x * x_conv_du_3

def DAU(x: np.ndarray,
        body_0_weight_name   : str, body_0_weight_shape,
        body_1_weight_name   : str, body_1_weight_shape,
        body_2_weight_name   : str, body_2_weight_shape,
        sa_weight_name       : str, sa_weight_shape,
        ca_0_weight_name     : str, ca_0_weight_shape,
        ca_2_weight_name     : str, ca_2_weight_shape,
        conv1x1_weight_name  : str, conv1x1_weight_shape
        ):

    body_0_weight  = inout.load_weight_from_txt(body_0_weight_name, body_0_weight_shape)
    body_1_weight  = inout.load_weight_from_txt(body_1_weight_name, body_1_weight_shape)
    body_2_weight  = inout.load_weight_from_txt(body_2_weight_name, body_2_weight_shape)
    sa_weight      = inout.load_weight_from_txt(sa_weight_name, sa_weight_shape)
    ca_0_weight    = inout.load_weight_from_txt(ca_0_weight_name, ca_0_weight_shape)
    ca_2_weight    = inout.load_weight_from_txt(ca_2_weight_name, ca_2_weight_shape)
    conv1x1_weight = inout.load_weight_from_txt(conv1x1_weight_name, conv1x1_weight_shape)

    DAU_body1 = Conv2d(x, body_0_weight, None, 1, 1)
    DAU_body2 = PReLU(DAU_body1, body_1_weight)
    DAU_body3 = Conv2d(DAU_body2, body_2_weight, None, 1, 1)
    DAU_SA    = Spatial_Attn_Layer(DAU_body3, sa_weight)
    DAU_CA    = Ca_Layer(DAU_body3, ca_0_weight, ca_2_weight)
    DAU_cat   = np.concatenate([DAU_SA, DAU_CA], axis=0) 
    DAU_con   = Conv2d(DAU_cat, conv1x1_weight, None, 0, 1)
    # inout.compare_arrays('MSRB2_DAU_body', DAU_body3, inout.load_ans_from_npy('MSRB/DAU_2_0_body',(1, 128, 148, 112)))
    # inout.compare_arrays('MSRB2_DAU_SA', DAU_SA, inout.load_ans_from_npy('MSRB/DAU_2_0_sa',(1, 128, 148, 112)))
    # inout.compare_arrays('MSRB2_DAU_CA', DAU_CA, inout.load_ans_from_npy('MSRB/DAU_2_0_ca',(1, 128, 148, 112)))
    # inout.compare_arrays('MSRB2_DAU_cat', DAU_cat, inout.load_ans_from_npy('MSRB/DAU_2_0_cat',(1, 256, 148, 112)))
    # inout.compare_arrays('MSRB2_DAU_conv', DAU_con, inout.load_ans_from_npy('MSRB/DAU_2_0_conv',(1, 128, 148, 112)))
    return x + DAU_con

def down(inp, filt, pad_type='reflect', stride=2, pad_off=0):
    filt_size = 3
    a = np.array([1., 2., 1.])
    filt = np.outer(a, a)
    filt = filt / np.sum(filt)
    
    pad_left = int((filt_size - 1) / 2) + pad_off
    pad_right = int(np.ceil((filt_size - 1) / 2)) + pad_off
    pad_top = pad_left
    pad_bottom = pad_right
    
    if filt_size == 1:
        if pad_off == 0:
            return inp[:, ::stride, ::stride]
        else:
            padded = np.pad(inp, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode=pad_type)
            return padded[:, ::stride, ::stride]
    else:
        padded = np.pad(inp, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode=pad_type)
        C, Hp, Wp = padded.shape
        fh, fw = filt.shape
        out_h = (Hp - fh) // stride + 1
        out_w = (Wp - fw) // stride + 1
        
        shape = (C, out_h, out_w, fh, fw)
        strides = (padded.strides[0], padded.strides[1] * stride, padded.strides[2] * stride,
                   padded.strides[1], padded.strides[2])
        view = as_strided(padded, shape=shape, strides=strides)
        
        out = np.einsum('cijkl,kl->cij', view, filt)
        return out

def DownSample(x: np.ndarray,
               top_0_weight_name : str, top_0_weight_shape,
               top_1_weight_name : str, top_1_weight_shape,
               top_2_weight_name : str, top_2_weight_shape,
               top_3_weight_name : str, top_3_weight_shape,
               top_4_weight_name : str, top_4_weight_shape,
               top_5_weight_name : str, top_5_weight_shape,
               bot_0_weight_name : str, bot_0_weight_shape,
               bot_1_weight_name : str, bot_1_weight_shape,
               ):
    top_0_weight = inout.load_weight_from_txt(top_0_weight_name, top_0_weight_shape)
    top_1_weight = inout.load_weight_from_txt(top_1_weight_name, top_1_weight_shape)
    top_2_weight = inout.load_weight_from_txt(top_2_weight_name, top_2_weight_shape)
    top_3_weight = inout.load_weight_from_txt(top_3_weight_name, top_3_weight_shape)
    top_4_weight = inout.load_weight_from_txt(top_4_weight_name, top_4_weight_shape)
    top_5_weight = inout.load_weight_from_txt(top_5_weight_name, top_5_weight_shape)
    bot_0_weight = inout.load_weight_from_txt(bot_0_weight_name, bot_0_weight_shape)
    bot_1_weight = inout.load_weight_from_txt(bot_1_weight_name, bot_1_weight_shape)

    top0 = Conv2d(x   , top_0_weight, None, 0, 1)
    top1 = PReLU (top0, top_1_weight)
    top2 = Conv2d(top1, top_2_weight, None, 1, 1)
    top3 = PReLU (top2, top_3_weight)
    top4 = down(top3, top_4_weight)
    top5 = Conv2d(top4, top_5_weight, None, 0, 1)
    bot0 = down(x   , bot_0_weight)
    bot1 = Conv2d(bot0, bot_1_weight, None, 0, 1)
    return top5 + bot1

def UpSample(x: np.ndarray,
             top_0_weight_name : str, top_0_weight_shape,
             top_1_weight_name : str, top_1_weight_shape,
             top_2_weight_name : str, top_2_weight_shape,
             top_3_weight_name : str, top_3_weight_shape,
             top_4_weight_name : str, top_4_weight_shape,
             bot_1_weight_name : str, bot_1_weight_shape,
             ):
    top_0_weight = inout.load_weight_from_txt(top_0_weight_name, top_0_weight_shape)
    top_1_weight = inout.load_weight_from_txt(top_1_weight_name, top_1_weight_shape)
    top_2_weight = inout.load_weight_from_txt(top_2_weight_name, top_2_weight_shape)
    top_3_weight = inout.load_weight_from_txt(top_3_weight_name, top_3_weight_shape)
    top_4_weight = inout.load_weight_from_txt(top_4_weight_name, top_4_weight_shape)
    bot_1_weight = inout.load_weight_from_txt(bot_1_weight_name, bot_1_weight_shape)
    
    top0 = Conv2d(x   , top_0_weight, None, 0, 1)
    top1 = PReLU (top0, top_1_weight)
    top2 = ConvTranspose2d(top1, top_2_weight)
    top3 = PReLU (top2, top_3_weight)
    top4 = Conv2d(top3, top_4_weight, None, 0, 1)

    bot0 = upsample_bilinear(x)
    bot1 = Conv2d(bot0, bot_1_weight, None, 0, 1)

    # inout.compare_arrays('up_top_2_1', top4, inout.load_ans_from_npy('MSRB/up_top_2_1',(1, 128, 148, 112)))
    # inout.compare_arrays('up_bot_2_1', bot1, inout.load_ans_from_npy('MSRB/up_bot_2_1',(1, 128, 148, 112)))

    return top4 + bot1

def SKFF(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray,
         feat_Z_0_weight_name   : str, feat_Z_0_weight_shape,
         feat_Z_1_weight_name   : str, feat_Z_1_weight_shape,
         att_vec_0_weight_name  : str, att_vec_0_weight_shape,
         att_vec_1_weight_name  : str, att_vec_1_weight_shape,
         att_vec_2_weight_name  : str, att_vec_2_weight_shape,
         ):
    
    feat_Z_0_weight  = inout.load_weight_from_txt(feat_Z_0_weight_name , feat_Z_0_weight_shape)
    feat_Z_1_weight  = inout.load_weight_from_txt(feat_Z_1_weight_name , feat_Z_1_weight_shape)
    att_vec_0_weight = inout.load_weight_from_txt(att_vec_0_weight_name, att_vec_0_weight_shape)
    att_vec_1_weight = inout.load_weight_from_txt(att_vec_1_weight_name, att_vec_1_weight_shape)
    att_vec_2_weight = inout.load_weight_from_txt(att_vec_2_weight_name, att_vec_2_weight_shape)

    feats = np.stack([x0, x1, x2], axis=0)
    feats_U = feats.sum(axis=0)
    feats_S = feats_U.mean(axis=(1, 2), keepdims=True)
    feats_Z = Conv2d(feats_S, feat_Z_0_weight, None,0, 1)
    feats_Z = PReLU(feats_Z, feat_Z_1_weight)

    att_vec0 = Conv2d(feats_Z, att_vec_0_weight, None, 0, 1)
    att_vec1 = Conv2d(feats_Z, att_vec_1_weight, None, 0, 1)
    att_vec2 = Conv2d(feats_Z, att_vec_2_weight, None, 0, 1)

    att_vec  = np.stack([att_vec0, att_vec1, att_vec2], axis=0)
    att_vec  = Softmax(att_vec)
    feats_V  = (feats * att_vec).sum(axis=0)
    return feats_V

def MSRB(x: np.ndarray, n_feat: int,
         block0_body_0_weight_name          : str, block0_body_0_weight_shape,
         block0_body_1_weight_name          : str, block0_body_1_weight_shape,
         block0_body_2_weight_name          : str, block0_body_2_weight_shape,
         block0_sa_weight_name              : str, block0_sa_weight_shape,
         block0_ca_0_weight_name            : str, block0_ca_0_weight_shape,
         block0_ca_2_weight_name            : str, block0_ca_2_weight_shape,
         block0_conv1x1_weight_name         : str, block0_conv1x1_weight_shape,

         down_128_2_body_top_0_weight_name  : str, down_128_2_body_top_0_weight_shape,
         down_128_2_body_top_1_weight_name  : str, down_128_2_body_top_1_weight_shape,
         down_128_2_body_top_2_weight_name  : str, down_128_2_body_top_2_weight_shape,
         down_128_2_body_top_3_weight_name  : str, down_128_2_body_top_3_weight_shape,
         down_128_2_body_top_4_weight_name  : str, down_128_2_body_top_4_weight_shape,
         down_128_2_body_top_5_weight_name  : str, down_128_2_body_top_5_weight_shape,
         down_128_2_body_bot_0_weight_name  : str, down_128_2_body_bot_0_weight_shape,
         down_128_2_body_bot_1_weight_name  : str, down_128_2_body_bot_1_weight_shape,

         block1_body_0_weight_name          : str, block1_body_0_weight_shape,
         block1_body_1_weight_name          : str, block1_body_1_weight_shape,
         block1_body_2_weight_name          : str, block1_body_2_weight_shape,
         block1_sa_weight_name              : str, block1_sa_weight_shape,
         block1_ca_0_weight_name            : str, block1_ca_0_weight_shape,
         block1_ca_2_weight_name            : str, block1_ca_2_weight_shape,
         block1_conv1x1_weight_name         : str, block1_conv1x1_weight_shape,

         down_256_2_body_top_0_weight_name  : str, down_256_2_body_top_0_weight_shape,
         down_256_2_body_top_1_weight_name  : str, down_256_2_body_top_1_weight_shape,
         down_256_2_body_top_2_weight_name  : str, down_256_2_body_top_2_weight_shape,
         down_256_2_body_top_3_weight_name  : str, down_256_2_body_top_3_weight_shape,
         down_256_2_body_top_4_weight_name  : str, down_256_2_body_top_4_weight_shape,
         down_256_2_body_top_5_weight_name  : str, down_256_2_body_top_5_weight_shape,
         down_256_2_body_bot_0_weight_name  : str, down_256_2_body_bot_0_weight_shape,
         down_256_2_body_bot_1_weight_name  : str, down_256_2_body_bot_1_weight_shape,

         block2_body_0_weight_name          : str, block2_body_0_weight_shape,
         block2_body_1_weight_name          : str, block2_body_1_weight_shape,
         block2_body_2_weight_name          : str, block2_body_2_weight_shape,
         block2_sa_weight_name              : str, block2_sa_weight_shape,
         block2_ca_0_weight_name            : str, block2_ca_0_weight_shape,
         block2_ca_2_weight_name            : str, block2_ca_2_weight_shape,
         block2_conv1x1_weight_name         : str, block2_conv1x1_weight_shape,

         last_up_1_0_body_top_0_weight_name : str, last_up_1_0_body_top_0_weight_shape,
         last_up_1_0_body_top_1_weight_name : str, last_up_1_0_body_top_1_weight_shape,
         last_up_1_0_body_top_2_weight_name : str, last_up_1_0_body_top_2_weight_shape,
         last_up_1_0_body_top_3_weight_name : str, last_up_1_0_body_top_3_weight_shape,
         last_up_1_0_body_top_4_weight_name : str, last_up_1_0_body_top_4_weight_shape,
         last_up_1_0_body_bot_1_weight_name : str, last_up_1_0_body_bot_1_weight_shape,

         last_up_2_0_body_top_0_weight_name : str, last_up_2_0_body_top_0_weight_shape,
         last_up_2_0_body_top_1_weight_name : str, last_up_2_0_body_top_1_weight_shape,
         last_up_2_0_body_top_2_weight_name : str, last_up_2_0_body_top_2_weight_shape,
         last_up_2_0_body_top_3_weight_name : str, last_up_2_0_body_top_3_weight_shape,
         last_up_2_0_body_top_4_weight_name : str, last_up_2_0_body_top_4_weight_shape,
         last_up_2_0_body_bot_1_weight_name : str, last_up_2_0_body_bot_1_weight_shape,

         last_up_2_1_body_top_0_weight_name : str, last_up_2_1_body_top_0_weight_shape,
         last_up_2_1_body_top_1_weight_name : str, last_up_2_1_body_top_1_weight_shape,
         last_up_2_1_body_top_2_weight_name : str, last_up_2_1_body_top_2_weight_shape,
         last_up_2_1_body_top_3_weight_name : str, last_up_2_1_body_top_3_weight_shape,
         last_up_2_1_body_top_4_weight_name : str, last_up_2_1_body_top_4_weight_shape,
         last_up_2_1_body_bot_1_weight_name : str, last_up_2_1_body_bot_1_weight_shape,

         skff_feat_Z_0_weight_name          : str, skff_feat_Z_0_weight_shape,
         skff_feat_Z_1_weight_name          : str, skff_feat_Z_1_weight_shape,
         skff_att_vec_0_weight_name         : str, skff_att_vec_0_weight_shape,
         skff_att_vec_1_weight_name         : str, skff_att_vec_1_weight_shape,
         skff_att_vec_2_weight_name         : str, skff_att_vec_2_weight_shape,

         conv_weight_name                   : str, conv_weight_shape,
          ):

    tmp0 = DAU(x, 
               block0_body_0_weight_name , block0_body_0_weight_shape,
               block0_body_1_weight_name , block0_body_1_weight_shape,
               block0_body_2_weight_name , block0_body_2_weight_shape,
               block0_sa_weight_name     , block0_sa_weight_shape,
               block0_ca_0_weight_name   , block0_ca_0_weight_shape,
               block0_ca_2_weight_name   , block0_ca_2_weight_shape,
               block0_conv1x1_weight_name, block0_conv1x1_weight_shape,
               )

    down1 = DownSample(tmp0,
                       down_128_2_body_top_0_weight_name, down_128_2_body_top_0_weight_shape,
                       down_128_2_body_top_1_weight_name, down_128_2_body_top_1_weight_shape,
                       down_128_2_body_top_2_weight_name, down_128_2_body_top_2_weight_shape,
                       down_128_2_body_top_3_weight_name, down_128_2_body_top_3_weight_shape,
                       down_128_2_body_top_4_weight_name, down_128_2_body_top_4_weight_shape,
                       down_128_2_body_top_5_weight_name, down_128_2_body_top_5_weight_shape,
                       down_128_2_body_bot_0_weight_name, down_128_2_body_bot_0_weight_shape,
                       down_128_2_body_bot_1_weight_name, down_128_2_body_bot_1_weight_shape
                       )
 
    tmp1 = DAU(down1, 
           block1_body_0_weight_name , block1_body_0_weight_shape,
           block1_body_1_weight_name , block1_body_1_weight_shape,
           block1_body_2_weight_name , block1_body_2_weight_shape,
           block1_sa_weight_name     , block1_sa_weight_shape,
           block1_ca_0_weight_name   , block1_ca_0_weight_shape,
           block1_ca_2_weight_name   , block1_ca_2_weight_shape,
           block1_conv1x1_weight_name, block1_conv1x1_weight_shape,
           )

    down2 = DownSample(tmp1,
                       down_256_2_body_top_0_weight_name, down_256_2_body_top_0_weight_shape,
                       down_256_2_body_top_1_weight_name, down_256_2_body_top_1_weight_shape,
                       down_256_2_body_top_2_weight_name, down_256_2_body_top_2_weight_shape,
                       down_256_2_body_top_3_weight_name, down_256_2_body_top_3_weight_shape,
                       down_256_2_body_top_4_weight_name, down_256_2_body_top_4_weight_shape,
                       down_256_2_body_top_5_weight_name, down_256_2_body_top_5_weight_shape,
                       down_256_2_body_bot_0_weight_name, down_256_2_body_bot_0_weight_shape,
                       down_256_2_body_bot_1_weight_name, down_256_2_body_bot_1_weight_shape
                       )

    tmp2 = DAU(down2, 
           block2_body_0_weight_name , block2_body_0_weight_shape,
           block2_body_1_weight_name , block2_body_1_weight_shape,
           block2_body_2_weight_name , block2_body_2_weight_shape,
           block2_sa_weight_name     , block2_sa_weight_shape,
           block2_ca_0_weight_name   , block2_ca_0_weight_shape,
           block2_ca_2_weight_name   , block2_ca_2_weight_shape,
           block2_conv1x1_weight_name, block2_conv1x1_weight_shape,
           )

    last_up0 = tmp0

    last_up1 = UpSample(tmp1,
                        last_up_1_0_body_top_0_weight_name, last_up_1_0_body_top_0_weight_shape,
                        last_up_1_0_body_top_1_weight_name, last_up_1_0_body_top_1_weight_shape,
                        last_up_1_0_body_top_2_weight_name, last_up_1_0_body_top_2_weight_shape,
                        last_up_1_0_body_top_3_weight_name, last_up_1_0_body_top_3_weight_shape,
                        last_up_1_0_body_top_4_weight_name, last_up_1_0_body_top_4_weight_shape,
                        last_up_1_0_body_bot_1_weight_name, last_up_1_0_body_bot_1_weight_shape
                        )

    last_up2 = UpSample(tmp2,
                        last_up_2_0_body_top_0_weight_name, last_up_2_0_body_top_0_weight_shape,
                        last_up_2_0_body_top_1_weight_name, last_up_2_0_body_top_1_weight_shape,
                        last_up_2_0_body_top_2_weight_name, last_up_2_0_body_top_2_weight_shape,
                        last_up_2_0_body_top_3_weight_name, last_up_2_0_body_top_3_weight_shape,
                        last_up_2_0_body_top_4_weight_name, last_up_2_0_body_top_4_weight_shape,
                        last_up_2_0_body_bot_1_weight_name, last_up_2_0_body_bot_1_weight_shape
                        )

    last_up2 = UpSample(last_up2,
                        last_up_2_1_body_top_0_weight_name, last_up_2_1_body_top_0_weight_shape,
                        last_up_2_1_body_top_1_weight_name, last_up_2_1_body_top_1_weight_shape,
                        last_up_2_1_body_top_2_weight_name, last_up_2_1_body_top_2_weight_shape,
                        last_up_2_1_body_top_3_weight_name, last_up_2_1_body_top_3_weight_shape,
                        last_up_2_1_body_top_4_weight_name, last_up_2_1_body_top_4_weight_shape,
                        last_up_2_1_body_bot_1_weight_name, last_up_2_1_body_bot_1_weight_shape
                        )

    selective_kernel = SKFF(last_up0, last_up1, last_up2,
                            skff_feat_Z_0_weight_name , skff_feat_Z_0_weight_shape,
                            skff_feat_Z_1_weight_name , skff_feat_Z_1_weight_shape,
                            skff_att_vec_0_weight_name, skff_att_vec_0_weight_shape,
                            skff_att_vec_1_weight_name, skff_att_vec_1_weight_shape,
                            skff_att_vec_2_weight_name, skff_att_vec_2_weight_shape,
                            )

    conv_weight = inout.load_weight_from_txt(conv_weight_name, conv_weight_shape)
    conv = Conv2d(selective_kernel, conv_weight, None, 1, 1)
    result = conv + x

    # inout.compare_arrays('MSRB2_DAU_0', tmp0, inout.load_ans_from_npy('MSRB/DAU_2_0',(1, 128, 148, 112)))
    # inout.compare_arrays('MSRB2_DAU_down_1', down1, inout.load_ans_from_npy('MSRB/DAU_down_2_1',(1, 256,  74,  56)))
    # inout.compare_arrays('MSRB2_DAU_1', tmp1, inout.load_ans_from_npy('MSRB/DAU_2_1',(1, 256,  74,  56)))
    # inout.compare_arrays('MSRB2_DAU_2', tmp2, inout.load_ans_from_npy('MSRB/DAU_2_2',(1, 512,  37,  28)))
    # inout.compare_arrays('MSRB2_up_0', last_up0, inout.load_ans_from_npy('MSRB/up_2_0',(1, 128, 148, 112)))
    # inout.compare_arrays('MSRB2_up_1', last_up1, inout.load_ans_from_npy('MSRB/up_2_1',(1, 128, 148, 112)))
    # inout.compare_arrays('MSRB2_up_2', last_up2, inout.load_ans_from_npy('MSRB/up_2_2',(1, 128, 148, 112)))
    # inout.compare_arrays('MSRB2_skff_2', selective_kernel, inout.load_ans_from_npy('MSRB/skff_2',(1, 128, 148, 112)))
    # inout.compare_arrays('MSRB2_conv_2', conv, inout.load_ans_from_npy('MSRB/conv_2',(1, 128, 148, 112)))

    return result


