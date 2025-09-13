import numpy as np
import inout_test
import error

def interpolate_verilog(x):
    H, W = x.shape
    o_H = H // 2
    o_W = W // 2
    x_pad = np.pad(x, ((1,2),(1,2)), mode = 'edge')
    result = np.zeros((o_H, o_W))
    for w in range(o_W):
        for h in range(o_H):
            ow = w * 2 + 1
            oh = h * 2 + 1
            result[h][w] = (x_pad[oh-1, ow-1] *  0.0087890625 + x_pad[oh-1, ow  ] * -0.0556640625 + x_pad[oh-1, ow+1] * -0.0556640625 + x_pad[oh-1, ow+2] *  0.0087890625 + 
                            x_pad[oh  , ow-1] * -0.0556640625 + x_pad[oh  , ow  ] *  0.3525390625 + x_pad[oh  , ow+1] *  0.3525390625 + x_pad[oh  , ow+2] * -0.0556640625 + 
                            x_pad[oh+1, ow-1] * -0.0556640625 + x_pad[oh+1, ow  ] *  0.3525390625 + x_pad[oh+1, ow+1] *  0.3525390625 + x_pad[oh+1, ow+2] * -0.0556640625 + 
                            x_pad[oh+2, ow-1] *  0.0087890625 + x_pad[oh+2, ow  ] * -0.0556640625 + x_pad[oh+2, ow+1] * -0.0556640625 + x_pad[oh+2, ow+2] *  0.0087890625 )
    return result

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

if __name__ == '__main__':
    input         = inout_test.txt_to_numpy_array('../data/input_1_480_640.txt', (1, 480, 640))
    input         = np.squeeze(input).astype(np.float32)
    output        = interpolate_verilog(input)
    ans           = interpolate(input)
    ans           = ans[None, ...]
    output        = output[None, ...]
    inout_test.numpy_array_to_txt(output ,'../data/output')
    error.compare_numpy_arrays(output, ans, '../data/error.txt')