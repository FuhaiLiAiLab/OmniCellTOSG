import os
import torch
import numpy as np

from pynvml import *


# GET [gpu_ids] OF ALL AVAILABLE GPUs
def get_available_devices():
    if torch.cuda.is_available():
        # GET AVAILABLE GPUs RESOURCES
        gpu_ids = []
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        # RANK THOSE GPUs
        nvmlInit()
        gpu_free_dict = {}
        for gpu_id in gpu_ids:
            h = nvmlDeviceGetHandleByIndex(gpu_id)
            info = nvmlDeviceGetMemoryInfo(h)
            gpu_free_dict[gpu_id] = info.free
            print(f'--- GPU{gpu_id} has {info.free/1024**3}G Free Memory ---')
            # import pdb; pdb.set_trace()
        sort_gpu_free = sorted(gpu_free_dict.items(), key=lambda x: x[1], reverse=True)
        sort_gpu_ids = []
        sort_gpu_ids += [gpu_id[0] for gpu_id in sort_gpu_free]
        # USE THE MAXIMAL GPU AS MAIN DEVICE
        max_free_gpu_id = sort_gpu_ids[0]
        device = torch.device(f'cuda:{gpu_ids[max_free_gpu_id]}') 
        torch.cuda.set_device(device)
        return device, sort_gpu_ids
    else:
        gpu_ids = []
        device = torch.device('cpu')
        return device, gpu_ids