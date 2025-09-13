import numpy as np
import inout_test 

def Preprocessing(x: np.ndarray) -> np.ndarray:
    x = np.transpose(x, (1, 2, 0))
    H, W, _ = x.shape
    new_H = H - (H % 16)
    new_W = W - (W % 16)
    x_crop = x[:new_H, :new_W, :]
    x_crop = x_crop.astype(np.float32) / 256.0
    mean = np.array([0.62, 0.58, 0.56])
    std = np.array([0.125, 0.125, 0.125])
    x_crop = (x_crop - mean) / std
    result = np.transpose(x_crop, (2, 0, 1))
    
    return result

if __name__ == '__main__' :
    input     = inout_test.txt_to_numpy_array('../data/input_3_480_640.txt', (3, 480, 640))
    output    = Preprocessing(input)
    input_2   = inout_test.float_to_binary(input , 8, 0, False)
    output_2d = inout_test.float_to_binary(output, 9, 5, True)
    inout_test.numpy_array_to_txt(input_2  ,'../data/input_binary')
    inout_test.numpy_array_to_txt(output_2d,'../data/ans_binary')