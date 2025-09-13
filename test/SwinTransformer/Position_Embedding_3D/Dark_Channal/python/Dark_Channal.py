import numpy as np
import inout_test

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

if __name__ == '__main__':
    input         = inout_test.txt_to_numpy_array('../data/input_binary_3_480_640.txt', (3, 480, 640), ' ', str)
    input         = inout_test.binary_to_float(input , 9, 5, True)
    output        = dark_channal(input)
    output        = output[None, ...]
    output_binary = inout_test.float_to_binary(output, 9, 5, True)
    inout_test.numpy_array_to_txt(output_binary,'../data/ans_binary')
    inout_test.numpy_array_to_txt(output,'../data/ans')