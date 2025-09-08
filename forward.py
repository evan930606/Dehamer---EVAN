

swin_in = x #96,192,384,768
swin_out_1=self.swin_1(swin_in)      
swin_input_1=self.E_block1(swin_in)#32    
swin_input_1=self.PPM1(swin_input_1)  

swin_input_2=self.E_block2(swin_input_1)#64 
swin_input_2=self.PPM2(swin_input_2)     

swin_input_3=self.E_block3(swin_input_2)#128
swin_input_3=self.PPM3(swin_input_3) 
swin_input_4=self.E_block4(swin_input_3)#256
swin_input_4=self.PPM4(swin_input_4) 
upsample1 = self._block1(swin_input_4)#256

beta_1 = self.conv1_1(swin_out_1[2])
gamma_1 = self.conv1_2(swin_out_1[2])
swin_input_3_refine=self.IN_3(swin_input_3)*beta_1+gamma_1#128
concat3 = torch.cat((swin_input_3,swin_input_3_refine,upsample1), dim=1)#256+256+256==512
decoder_3 = self.ReLU(self.conv1(concat3)) #256
upsample3 = self._block3(decoder_3)#128
upsample3=self.MSRB2(upsample3)
beta_2 = self.conv2_1(swin_out_1[1])
gamma_2 = self.conv2_2(swin_out_1[1])
swin_input_2_refine=self.IN_2(swin_input_2)*beta_2+gamma_2 #64
concat2 = torch.cat((swin_input_2,swin_input_2_refine,upsample3), dim=1)#128+128+128=256
decoder_2 = self.ReLU(self.conv2(concat2))#128
upsample4 = self._block4(decoder_2)#64
upsample4=self.MSRB3(upsample4)
beta_3 = self.conv3_1(swin_out_1[0])
gamma_3 =self.conv3_2(swin_out_1[0]) 
swin_input_1_refine=self.IN_1(swin_input_1)*beta_3+gamma_3 #32
concat1 = torch.cat((swin_input_1,swin_input_1_refine,upsample4), dim=1)#64+64+64=128
decoder_1 = self.ReLU(self.conv3(concat1))#64
upsample5 = self._block5(decoder_1)#32
decoder_0 = self.ReLU(self.conv4(upsample5))
result=self._block7(decoder_0)
