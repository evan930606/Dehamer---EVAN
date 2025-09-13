import numpy as np
import inout_test
import error

if __name__ == '__main__':
    # input
    input0        = inout_test.txt_to_numpy_array('../data/input0_96_240_320.txt', (96, 240, 320))
    input1        = inout_test.txt_to_numpy_array('../data/input1_96_240_320.txt', (96, 240, 320))

    output       = input0 + input1

    inout_test.numpy_array_to_txt(output ,'../data/output')