import numpy as np 
import inout 
import Swin_Transformer as ST
import basic_layer
import E_block
import D_Block

test_input = inout.ppm_to_numpy('./pic/example.ppm')
test_input = inout.png_to_numpy_480x640('./pic/tree.jpg')
inout.save_png_3c(test_input, './pic/input.png')
inout.numpy_array_to_txt(test_input, './pic/input')
test_input = inout.Preprocessing(test_input)
swin_stage0_image ,swin_stage1_image ,swin_stage2_image = ST.Swin_Transformer(test_input)
Encoder1_PPM_image, Encoder2_PPM_image ,Encoder3_PPM_image, Encoder4_PPM_image=E_block.Encoder(test_input)

result = D_Block.Decoder(swin_stage0_image,
                         swin_stage1_image,
                         swin_stage2_image,
                         Encoder1_PPM_image,
                         Encoder2_PPM_image,
                         Encoder3_PPM_image,
                         Encoder4_PPM_image)

inout.save_png_3c(result, './pic/result.png')
inout.save_image(result, 'result','NH')


# inout.compare_arrays('Preprocessing',test_input, inout.load_ans_from_npy('x',(1, 3, 592, 448)))
# inout.compare_arrays('Swin_Stage_0',swin_stage0_image, inout.load_ans_from_npy('swin_out_1_0',(1, 96, 296, 224)))
# inout.compare_arrays('Swin_Stage_1',swin_stage1_image, inout.load_ans_from_npy('swin_out_1_1',(1, 192, 148, 112)))
# inout.compare_arrays('Swin_Stage_2',swin_stage2_image, inout.load_ans_from_npy('swin_out_1_2',(1, 384, 74, 56)))
# inout.compare_arrays('result', result, inout.load_ans_from_npy('result',(1, 3, 592, 448)))