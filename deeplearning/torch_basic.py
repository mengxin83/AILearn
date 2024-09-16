import torch
print(torch.__version__)

# CUDA支持
print('CUDA版本:',torch.version.cuda)
print('Pytorch版本:',torch.__version__)
print('显卡是否可用:','可用' if(torch.cuda.is_available()) else '不可用')
print('显卡数量:',torch.cuda.device_count())
if torch.cuda.is_available()==True:
    print('当前显卡的CUDA算力:',torch.cuda.get_device_capability(0))
    print('当前显卡型号:',torch.cuda.get_device_name(0))
