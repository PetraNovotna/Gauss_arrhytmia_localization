import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import matplotlib.pyplot as plt
import os



class myAttention(nn.Module):
    
    def __init__(self,in_size, out_size,levels):
        super().__init__()
        
        self.levels = levels
        
        self.conv_tanh = nn.Conv1d(in_size, in_size, 1)
        self.conv_sigm = nn.Conv1d(in_size, in_size, 1)
        self.conv_w = nn.Conv1d(in_size, out_size, 1)
        
        self.conv_final = nn.Conv1d(in_size,out_size,1)
    
    def forward(self, inputs,remove_matrix):
        
        
        tanh = torch.tanh(self.conv_tanh(inputs))
        
        sigm = torch.sigmoid(self.conv_sigm(inputs))
        
        z = self.conv_w(tanh * sigm) 
        
        z[remove_matrix.repeat(1,list(z.size())[1],1)==1] = -np.Inf
        
        a = torch.softmax(z,dim=2)
        
        
        output = self.conv_final(inputs)
        output = output * a
        
        a2 = output
        
        output = torch.sum(output,dim=2)
        

        return output,a,a2






class myConv(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=None,do_batch=1,dov=None):
        super().__init__()
        
        pad=int((filter_size-1)/2)
        
        self.do_batch=do_batch
        self.dov=dov
        self.conv=nn.Conv1d(in_size, out_size,filter_size,stride,pad,bias=False)
        self.bn=nn.BatchNorm1d(out_size,momentum=0.1)
        
        
        if self.dov:
            self.do=nn.Dropout(dov)
            
    def swish(self,x):
        return x * F.sigmoid(x)
    
    def forward(self, inputs):
     
        outputs = self.conv(inputs)
        if self.do_batch:
            outputs = self.bn(outputs)  
        
        outputs=F.relu(outputs)
        # outputs=self.swish(outputs)
        
        
        if self.dov:
            outputs = self.do(outputs)
            
            
            
        
        return outputs


        
class Net_addition_grow(nn.Module):
    def set_ts(self,ts):
        self.ts=ts
        
    def get_ts(self):
        return self.ts
    
    
    def __init__(self,levels=7, lvl1_size=4, input_size=12, output_size=9, convs_in_layer=3, init_conv=4,
                 filter_size=13, mil_solution='max',do=None,blocks_in_lvl=2):
        
        
        
        
        
        
        super().__init__()
        self.levels=levels
        self.lvl1_size=lvl1_size
        self.input_size=input_size
        self.output_size=output_size
        self.convs_in_layer=convs_in_layer
        self.filter_size=filter_size
        self.blocks_in_lvl=blocks_in_lvl
        self.do = do
        self.mil_solution = mil_solution
        
        init_conv=lvl1_size
        
        
        
        self.init_conv=myConv(input_size,init_conv,filter_size=filter_size)
        
        
        self.layers=nn.ModuleList()
        for lvl_num in range(self.levels):
            
            for block_num in range(self.blocks_in_lvl):
            
                if block_num==0 and lvl_num>0:
                    self.layers.append(myConv(int(lvl1_size*(lvl_num)), int(lvl1_size*(lvl_num+1)),filter_size=1,dov=do))
                    
                    self.layers.append(myConv(int(lvl1_size*(lvl_num)), int(lvl1_size*(lvl_num+1)),filter_size=filter_size,dov=do))
                else:
                    self.layers.append(myConv(int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1)),filter_size=filter_size,dov=do))
                
                for conv_num_in_lvl in range(self.convs_in_layer-1):
                    self.layers.append(myConv(int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1)),filter_size=filter_size,dov=do))

        if self.mil_solution == 'max' or self.mil_solution == 'gauss':
            self.conv_final=myConv(int(lvl1_size*(self.levels)), output_size,filter_size=filter_size)   
        elif self.mil_solution == 'att1-nolenmul' or self.mil_solution == 'att1-lenmul' or self.mil_solution == 'att2-nolenmul' or self.mil_solution == 'att2-lenmul':
            self.conv_final=myConv(int(lvl1_size*(self.levels)), int(lvl1_size*self.levels),filter_size=filter_size)
            self.attention = myAttention(int(lvl1_size*self.levels),output_size,self.levels)
        else:
            raise Exception('No mill solution')
        
        
        
        ## weigths initialization wih xavier method
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.xavier_normal_(m.weight)
                if not m.bias == None:
                    init.constant_(m.bias, 0)
        
        
        
    def forward(self, x,lens,detection):
        
        
        
        
        shape = list(x.size())
        remove_matrix = torch.ones((shape[0],1,shape[2]),dtype=x.dtype)
        cuda_check = x.is_cuda
        if cuda_check:
            cuda_device = x.get_device()
            device = torch.device('cuda:' + str(cuda_device) )
            remove_matrix = remove_matrix.to(device)
        
        
        detection=F.avg_pool1d(detection, 2 ** self.levels, 2 ** self.levels)

        
        for signal_num in range(list(x.size())[0]):
            
            k = int(lens[signal_num])
            
            remove_matrix[signal_num,:,:k] = 0
        
        
        
        x=self.init_conv(x)
        x = x * (1 - remove_matrix)
        
        
        ## aply all convolutions
        layer_num=-1
        for lvl_num in range(self.levels):
            for block_num in range(self.blocks_in_lvl):
            
                y = x
                
                if block_num ==0 and lvl_num>0:
                    
                    layer_num+=1
                    y=self.layers[layer_num](y)
                    y = y * (1 - remove_matrix)
                    
                
                for conv_num_in_block in range(self.convs_in_layer):
                    
                    layer_num+=1
                    x=self.layers[layer_num](x)
                    x = x * (1 - remove_matrix)
                    
                ## skip conection to previous layer and to the input
                x = x + y
            
            x = F.max_pool1d(x, 2, 2)
            remove_matrix = F.max_pool1d(remove_matrix, 2, 2)
            x = x * (1 - remove_matrix)
            
            
        x=self.conv_final(x)
        x = x * (1 - remove_matrix)


        if self.mil_solution == 'max' or self.mil_solution == 'gauss':
            heatmap = x.clone()
            x[remove_matrix.repeat(1,list(x.size())[1],1)==1] = -np.Inf
            x = F.adaptive_max_pool1d(x, 1)
        elif self.mil_solution == 'att1-nolenmul':
            x, a1, a2 = self.attention(x,remove_matrix)
            heatmap = a1
        elif self.mil_solution == 'att1-lenmul':
            x, a1, a2 = self.attention(x,remove_matrix)
            heatmap = a1   
            heatmap = heatmap * lens.view(lens.size(0),1,1).repeat(1,heatmap.size(1),heatmap.size(2))
        elif self.mil_solution == 'att2-nolenmul':
            x, a1, a2 = self.attention(x,remove_matrix)
            heatmap = a2
        elif self.mil_solution == 'att2-lenmul':
            x, a1, a2 = self.attention(x,remove_matrix)
            heatmap = a2   
            heatmap = heatmap * lens.view(lens.size(0),1,1).repeat(1,heatmap.size(1),heatmap.size(2))    
        else:
            raise Exception('No mill solution')
        
        x = x.view(list(x.size())[:2])
        score = x
        
        x = torch.sigmoid(x)
        
        
        return x, heatmap, score, detection
    
    def save_log(self,log):
        self.log=log
        
    def save_config(self,config):  
        self.config=config
        
    def save_lens(self,lens):
        self.lens=lens
        
    def save_filename_train_valid(self,train_names,valid_names):
        
        self.train_names = train_names
        
        self.valid_names = valid_names