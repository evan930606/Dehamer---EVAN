import numpy as np
import inout
import MSRB
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

def Instance_Norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=(1, 2), keepdims=True)                     
    var  = x.var(axis=(1, 2), keepdims=True)                      
    y = (x - mean) / np.sqrt(var + eps)
    return y.astype(x.dtype, copy=False)

def upsample_bilinear2d(x: np.ndarray, scale_factor: int = 2, align_corners: bool = False) -> np.ndarray:
    C, H, W = x.shape
    H_out = int(H * scale_factor)
    W_out = int(W * scale_factor)

    y_in = np.linspace(0, max(H - 1, 0), H_out, dtype=np.float32) if H_out > 1 else np.zeros(1, np.float32)
    x_in = np.linspace(0, max(W - 1, 0), W_out, dtype=np.float32) if W_out > 1 else np.zeros(1, np.float32)

    yy, xx = np.meshgrid(y_in, x_in, indexing='ij') 

    y0 = np.floor(yy).astype(np.int64)
    x0 = np.floor(xx).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1

    dy = yy - y0
    dx = xx - x0

    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)

    wa = (1.0 - dx) * (1.0 - dy)
    wb = (1.0 - dx) * dy        
    wc = dx * (1.0 - dy)        
    wd = dx * dy                

    Ia = x[:, y0c, x0c] 
    Ib = x[:, y1c, x0c]
    Ic = x[:, y0c, x1c]
    Id = x[:, y1c, x1c]

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out.astype(x.dtype, copy=False)

def D_block(x: np.ndarray, 
            weight_0_name: str, weight_0_shape,
            bias_0_name  : str, bias_0_shape,
            weight_2_name: str, weight_2_shape,
            bias_2_name  : str, bias_2_shape,
            ):
    D_block_0_weight = inout.load_weight_from_txt(weight_0_name, weight_0_shape)
    D_block_0_bias   = inout.load_weight_from_txt(bias_0_name, bias_0_shape)
    D_block_2_weight = inout.load_weight_from_txt(weight_2_name, weight_2_shape)
    D_block_2_bias   = inout.load_weight_from_txt(bias_2_name,bias_2_shape)

    D_block_0 = Conv2d(x, D_block_0_weight, D_block_0_bias)
    D_block_1 = ReLU(D_block_0)
    D_block_2 = Conv2d(D_block_1, D_block_2_weight, D_block_2_bias)
    D_block_3 = ReLU(D_block_2)
    D_block_4 = upsample_bilinear2d(D_block_3)
    return D_block_4

def D_Block_7(x: np.ndarray,
            weight_0_name: str, weight_0_shape,
            bias_0_name  : str, bias_0_shape,
            weight_2_name: str, weight_2_shape,
            bias_2_name  : str, bias_2_shape,
            ):
    D_block_0_weight = inout.load_weight_from_txt(weight_0_name, weight_0_shape)
    D_block_0_bias   = inout.load_weight_from_txt(bias_0_name, bias_0_shape)
    D_block_2_weight = inout.load_weight_from_txt(weight_2_name, weight_2_shape)
    D_block_2_bias   = inout.load_weight_from_txt(bias_2_name,bias_2_shape)

    D_block_0 = Conv2d(x, D_block_0_weight, D_block_0_bias)
    D_block_1 = ReLU(D_block_0)
    D_block_2 = Conv2d(D_block_1, D_block_2_weight, D_block_2_bias)
    return D_block_2


def Decoder(S_block_0: np.ndarray,
            S_block_1: np.ndarray,
            S_block_2: np.ndarray,
            E_block_0: np.ndarray,
            E_block_1: np.ndarray,
            E_block_2: np.ndarray,
            E_block_3: np.ndarray):
    
    upsample1 = D_block(E_block_3,
                        '_block1_0_weight',(256, 512, 3, 3),
                        '_block1_0_bias'  ,(256),
                        '_block1_2_weight',(256, 256, 3, 3),
                        '_block1_2_bias'  ,(256))
    
    conv1_weight   = inout.load_weight_from_txt('conv1_weight',(256, 768, 3, 3))
    conv1_bias     = inout.load_weight_from_txt('conv1_bias',(256))
    conv1_1_weight = inout.load_weight_from_txt('conv1_1_weight',(256, 384, 3, 3))
    conv1_1_bias   = inout.load_weight_from_txt('conv1_1_bias',(256))
    conv1_2_weight = inout.load_weight_from_txt('conv1_2_weight',(256, 384, 3, 3))
    conv1_2_bias   = inout.load_weight_from_txt('conv1_2_bias',(256))

    beta1     = Conv2d(S_block_2, conv1_1_weight, conv1_1_bias)
    gamma1    = Conv2d(S_block_2, conv1_2_weight, conv1_2_bias)
    instance1 = Instance_Norm(E_block_2)
    feature1  = instance1 * beta1 + gamma1
    concat1   = np.concatenate([E_block_2, feature1, upsample1], axis=0)

    conv1     = Conv2d(concat1, conv1_weight, conv1_bias)
    relu1     = ReLU(conv1)

    upsample2 = D_block(relu1,
                        '_block3_0_weight',(128, 256, 3, 3),
                        '_block3_0_bias'  ,(128),
                        '_block3_2_weight',(128, 128, 3, 3),
                        '_block3_2_bias'  ,(128))

    MSRB_2 = MSRB.MSRB(upsample2, 128,
                       'MSRB2_blocks_0_0_body_0_weight'         , (128, 128, 3, 3),
                       'MSRB2_blocks_0_0_body_1_weight'         , (1),
                       'MSRB2_blocks_0_0_body_2_weight'         , (128, 128, 3, 3),
                       'MSRB2_blocks_0_0_SA_spatial_conv_weight', (1, 2, 5, 5),
                       'MSRB2_blocks_0_0_CA_conv_du_0_weight'   , (16, 128, 1, 1),
                       'MSRB2_blocks_0_0_CA_conv_du_2_weight'   , (128, 16, 1, 1),
                       'MSRB2_blocks_0_0_conv1x1_weight'        , (128, 256, 1, 1),

                       'MSRB2_down_128_2_body_0_top_0_weight'   , (128, 128, 1, 1),
                       'MSRB2_down_128_2_body_0_top_1_weight'   , (1),
                       'MSRB2_down_128_2_body_0_top_2_weight'   , (128, 128, 3, 3),
                       'MSRB2_down_128_2_body_0_top_3_weight'   , (1),
                       'MSRB2_down_128_2_body_0_top_4_filt'     , (128, 1, 3, 3),
                       'MSRB2_down_128_2_body_0_top_5_weight'   , (256, 128, 1, 1),
                       'MSRB2_down_128_2_body_0_bot_0_filt'     , (128, 1, 3, 3),
                       'MSRB2_down_128_2_body_0_bot_1_weight'   , (256, 128, 1, 1),

                       'MSRB2_blocks_1_0_body_0_weight'         , (256, 256, 3, 3),
                       'MSRB2_blocks_1_0_body_1_weight'         , (1),
                       'MSRB2_blocks_1_0_body_2_weight'         , (256, 256, 3, 3),
                       'MSRB2_blocks_1_0_SA_spatial_conv_weight', (1, 2, 5, 5),
                       'MSRB2_blocks_1_0_CA_conv_du_0_weight'   , (32, 256, 1, 1),
                       'MSRB2_blocks_1_0_CA_conv_du_2_weight'   , (256, 32, 1, 1),
                       'MSRB2_blocks_1_0_conv1x1_weight'        , (256, 512, 1, 1),

                       'MSRB2_down_256_2_body_0_top_0_weight'   , (256, 256, 1, 1),
                       'MSRB2_down_256_2_body_0_top_1_weight'   , (1),
                       'MSRB2_down_256_2_body_0_top_2_weight'   , (256, 256, 3, 3),
                       'MSRB2_down_256_2_body_0_top_3_weight'   , (1),
                       'MSRB2_down_256_2_body_0_top_4_filt'     , (256, 1, 3, 3),
                       'MSRB2_down_256_2_body_0_top_5_weight'   , (512, 256, 1, 1),
                       'MSRB2_down_256_2_body_0_bot_0_filt'     , (256, 1, 3, 3),
                       'MSRB2_down_256_2_body_0_bot_1_weight'   , (512, 256, 1, 1),

                       'MSRB2_blocks_2_0_body_0_weight'         , (512, 512, 3, 3),
                       'MSRB2_blocks_2_0_body_1_weight'         , (1),
                       'MSRB2_blocks_2_0_body_2_weight'         , (512, 512, 3, 3),
                       'MSRB2_blocks_2_0_SA_spatial_conv_weight', (1, 2, 5, 5),
                       'MSRB2_blocks_2_0_CA_conv_du_0_weight'   , (64, 512, 1, 1),
                       'MSRB2_blocks_2_0_CA_conv_du_2_weight'   , (512, 64, 1, 1),
                       'MSRB2_blocks_2_0_conv1x1_weight'        , (512, 1024, 1, 1),

                       'MSRB2_last_up_1_body_0_top_0_weight'    , (256, 256, 1, 1),
                       'MSRB2_last_up_1_body_0_top_1_weight'    , (1),
                       'MSRB2_last_up_1_body_0_top_2_weight'    , (256, 256, 3, 3),
                       'MSRB2_last_up_1_body_0_top_3_weight'    , (1),
                       'MSRB2_last_up_1_body_0_top_4_weight'    , (128, 256, 1, 1),
                       'MSRB2_last_up_1_body_0_bot_1_weight'    , (128, 256, 1, 1),

                       'MSRB2_last_up_2_body_0_top_0_weight'    , (512, 512, 1, 1),
                       'MSRB2_last_up_2_body_0_top_1_weight'    , (1),
                       'MSRB2_last_up_2_body_0_top_2_weight'    , (512, 512, 3, 3),
                       'MSRB2_last_up_2_body_0_top_3_weight'    , (1),
                       'MSRB2_last_up_2_body_0_top_4_weight'    , (256, 512, 1, 1),
                       'MSRB2_last_up_2_body_0_bot_1_weight'    , (256, 512, 1, 1),

                       'MSRB2_last_up_2_body_1_top_0_weight'    , (256, 256, 1, 1),
                       'MSRB2_last_up_2_body_1_top_1_weight'    , (1),
                       'MSRB2_last_up_2_body_1_top_2_weight'    , (256, 256, 3, 3),
                       'MSRB2_last_up_2_body_1_top_3_weight'    , (1),
                       'MSRB2_last_up_2_body_1_top_4_weight'    , (128, 256, 1, 1),
                       'MSRB2_last_up_2_body_1_bot_1_weight'    , (128, 256, 1, 1),

                       'MSRB2_selective_kernel_0_conv_du_0_weight', (16, 128, 1, 1),
                       'MSRB2_selective_kernel_0_conv_du_1_weight', (1),
                       'MSRB2_selective_kernel_0_fcs_0_weight'    , (128, 16, 1, 1),
                       'MSRB2_selective_kernel_0_fcs_1_weight'    , (128, 16, 1, 1),
                       'MSRB2_selective_kernel_0_fcs_2_weight'    , (128, 16, 1, 1),

                       'MSRB2_conv_out_weight'                    , (128, 128, 3, 3))


    conv2_weight   = inout.load_weight_from_txt('conv2_weight',(128, 384, 3, 3))
    conv2_bias     = inout.load_weight_from_txt('conv2_bias',(128))
    conv2_1_weight = inout.load_weight_from_txt('conv2_1_weight',(128, 192, 3, 3))
    conv2_1_bias   = inout.load_weight_from_txt('conv2_1_bias',(128))
    conv2_2_weight = inout.load_weight_from_txt('conv2_2_weight',(128, 192, 3, 3))
    conv2_2_bias   = inout.load_weight_from_txt('conv2_2_bias',(128))

    beta2     = Conv2d(S_block_1, conv2_1_weight, conv2_1_bias)
    gamma2    = Conv2d(S_block_1, conv2_2_weight, conv2_2_bias)
    instance2 = Instance_Norm(E_block_1)
    feature2  = instance2 * beta2 + gamma2
    concat2   = np.concatenate([E_block_1, feature2, MSRB_2], axis=0)
    conv2     = Conv2d(concat2, conv2_weight, conv2_bias)
    relu2     = ReLU(conv2)

    upsample3 = D_block(relu2,
                        '_block4_0_weight',(64, 128, 3, 3),
                        '_block4_0_bias'  ,(64),
                        '_block4_2_weight',(64, 64, 3, 3),
                        '_block4_2_bias'  ,(64))

    MSRB_3 = MSRB.MSRB(upsample3, 64,
                       'MSRB3_blocks_0_0_body_0_weight'         , (64, 64, 3, 3),
                       'MSRB3_blocks_0_0_body_1_weight'         , (1),
                       'MSRB3_blocks_0_0_body_2_weight'         , (64, 64, 3, 3),
                       'MSRB3_blocks_0_0_SA_spatial_conv_weight', (1, 2, 5, 5),
                       'MSRB3_blocks_0_0_CA_conv_du_0_weight'   , (8, 64, 1, 1),
                       'MSRB3_blocks_0_0_CA_conv_du_2_weight'   , (64, 8, 1, 1),
                       'MSRB3_blocks_0_0_conv1x1_weight'        , (64, 128, 1, 1),

                       'MSRB3_down_64_2_body_0_top_0_weight'    , (64, 64, 1, 1),
                       'MSRB3_down_64_2_body_0_top_1_weight'    , (1),
                       'MSRB3_down_64_2_body_0_top_2_weight'    , (64, 64, 3, 3),
                       'MSRB3_down_64_2_body_0_top_3_weight'    , (1),
                       'MSRB3_down_64_2_body_0_top_4_filt'      , (64, 1, 3, 3),
                       'MSRB3_down_64_2_body_0_top_5_weight'    , (128, 64, 1, 1),
                       'MSRB3_down_64_2_body_0_bot_0_filt'      , (64, 1, 3, 3),
                       'MSRB3_down_64_2_body_0_bot_1_weight'    , (128, 64, 1, 1),

                       'MSRB3_blocks_1_0_body_0_weight'         , (128, 128, 3, 3),
                       'MSRB3_blocks_1_0_body_1_weight'         , (1),
                       'MSRB3_blocks_1_0_body_2_weight'         , (128, 128, 3, 3),
                       'MSRB3_blocks_1_0_SA_spatial_conv_weight', (1, 2, 5, 5),
                       'MSRB3_blocks_1_0_CA_conv_du_0_weight'   , (16, 128, 1, 1),
                       'MSRB3_blocks_1_0_CA_conv_du_2_weight'   , (128, 16, 1, 1),
                       'MSRB3_blocks_1_0_conv1x1_weight'        , (128, 256, 1, 1),

                       'MSRB3_down_128_2_body_0_top_0_weight'    , (128, 128, 1, 1),
                       'MSRB3_down_128_2_body_0_top_1_weight'    , (1),
                       'MSRB3_down_128_2_body_0_top_2_weight'    , (128, 128, 3, 3),
                       'MSRB3_down_128_2_body_0_top_3_weight'    , (1),
                       'MSRB3_down_128_2_body_0_top_4_filt'      , (128, 1, 3, 3),
                       'MSRB3_down_128_2_body_0_top_5_weight'    , (256, 128, 1, 1),
                       'MSRB3_down_128_2_body_0_bot_0_filt'      , (128, 1, 3, 3),
                       'MSRB3_down_128_2_body_0_bot_1_weight'    , (256, 128, 1, 1),

                       'MSRB3_blocks_2_0_body_0_weight'         , (256, 256, 3, 3),
                       'MSRB3_blocks_2_0_body_1_weight'         , (1),
                       'MSRB3_blocks_2_0_body_2_weight'         , (256, 256, 3, 3),
                       'MSRB3_blocks_2_0_SA_spatial_conv_weight', (1, 2, 5, 5),
                       'MSRB3_blocks_2_0_CA_conv_du_0_weight'   , (32, 256, 1, 1),
                       'MSRB3_blocks_2_0_CA_conv_du_2_weight'   , (256, 32, 1, 1),
                       'MSRB3_blocks_2_0_conv1x1_weight'        , (256, 512, 1, 1),

                       'MSRB3_last_up_1_body_0_top_0_weight'    , (128, 128, 1, 1),
                       'MSRB3_last_up_1_body_0_top_1_weight'    , (1),
                       'MSRB3_last_up_1_body_0_top_2_weight'    , (128, 128, 3, 3),
                       'MSRB3_last_up_1_body_0_top_3_weight'    , (1),
                       'MSRB3_last_up_1_body_0_top_4_weight'    , (64, 128, 1, 1),
                       'MSRB3_last_up_1_body_0_bot_1_weight'    , (64, 128, 1, 1),

                       'MSRB3_last_up_2_body_0_top_0_weight'    , (256, 256, 1, 1),
                       'MSRB3_last_up_2_body_0_top_1_weight'    , (1),
                       'MSRB3_last_up_2_body_0_top_2_weight'    , (256, 256, 3, 3),
                       'MSRB3_last_up_2_body_0_top_3_weight'    , (1),
                       'MSRB3_last_up_2_body_0_top_4_weight'    , (128, 256, 1, 1),
                       'MSRB3_last_up_2_body_0_bot_1_weight'    , (128, 256, 1, 1),

                       'MSRB3_last_up_2_body_1_top_0_weight'    , (128, 128, 1, 1),
                       'MSRB3_last_up_2_body_1_top_1_weight'    , (1),
                       'MSRB3_last_up_2_body_1_top_2_weight'    , (128, 128, 3, 3),
                       'MSRB3_last_up_2_body_1_top_3_weight'    , (1),
                       'MSRB3_last_up_2_body_1_top_4_weight'    , (64, 128, 1, 1),
                       'MSRB3_last_up_2_body_1_bot_1_weight'    , (64, 128, 1, 1),

                       'MSRB3_selective_kernel_0_conv_du_0_weight', (8, 64, 1, 1),
                       'MSRB3_selective_kernel_0_conv_du_1_weight', (1),
                       'MSRB3_selective_kernel_0_fcs_0_weight'    , (64, 8, 1, 1),
                       'MSRB3_selective_kernel_0_fcs_1_weight'    , (64, 8, 1, 1),
                       'MSRB3_selective_kernel_0_fcs_2_weight'    , (64, 8, 1, 1),

                       'MSRB3_conv_out_weight'                    , (64, 64, 3, 3))
    
    conv3_weight   = inout.load_weight_from_txt('conv3_weight',(64, 192, 3, 3))
    conv3_bias     = inout.load_weight_from_txt('conv3_bias',(64))
    conv3_1_weight = inout.load_weight_from_txt('conv3_1_weight',(64, 96, 3, 3))
    conv3_1_bias   = inout.load_weight_from_txt('conv3_1_bias',(64))
    conv3_2_weight = inout.load_weight_from_txt('conv3_2_weight',(64, 96, 3, 3))
    conv3_2_bias   = inout.load_weight_from_txt('conv3_2_bias',(64))

    beta3     = Conv2d(S_block_0, conv3_1_weight, conv3_1_bias)
    gamma3    = Conv2d(S_block_0, conv3_2_weight, conv3_2_bias)
    instance3 = Instance_Norm(E_block_0)
    feature3  = instance3 * beta3 + gamma3
    concat3   = np.concatenate([E_block_0, feature3, MSRB_3], axis=0)
    conv3     = Conv2d(concat3, conv3_weight, conv3_bias)
    relu3     = ReLU(conv3)

    upsample4 = D_block(relu3,
                        '_block5_0_weight',(32, 64, 3, 3),
                        '_block5_0_bias'  ,(32),
                        '_block5_2_weight',(32, 32, 3, 3),
                        '_block5_2_bias'  ,(32))

    conv4_weight   = inout.load_weight_from_txt('conv4_weight',(32, 32, 3, 3))
    conv4_bias     = inout.load_weight_from_txt('conv4_bias',(32))

    conv4     = Conv2d(upsample4, conv4_weight, conv4_bias)
    relu4     = ReLU(conv4)

    upsample5 = D_Block_7(relu4,
                        '_block7_0_weight',(32, 32, 3, 3),
                        '_block7_0_bias'  ,(32),
                        '_block7_2_weight',(3, 32, 3, 3),
                        '_block7_2_bias'  ,(3))

    return upsample5