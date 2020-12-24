import numpy as np
import pandas as pd
import pickle as pkl
import json

import torch

#将数据tokens转化为张量
def tokens_str_to_list(tokens):
#字符串转列表  
    if tokens is None:
        print('Tokens have nothing!')
        return None
    else:
        #print(type(tokens.iloc[0][0].replace('[','').replace(']', '').split(', ')))
        res = np.array(tokens.iloc[0][0].replace('[','').replace(']', '').split(', '))
        
    for i in range(len(tokens)-1):
        temp = tokens.iloc[i+1][0].replace('[','').replace(']', '').split(', ')
        res = np.vstack((res,np.array(temp)))
        
    #转换数据类型
    res = res.astype(np.uint8)
    res = torch.from_numpy(res)
    return res

#上面的转换太慢，因此将张量保存一下方便读取
#此外，暂时不会直接保存tensor，所以用numpy过渡了一下

#保存
def tensor_save(tensor,path):
    save_np = tensor.cpu().numpy()
    np.save(path, save_np)

#加载
def tensor_load(path):
    load_np = np.load(path)
    tensor = torch.from_numpy(load_np)
    return tensor


