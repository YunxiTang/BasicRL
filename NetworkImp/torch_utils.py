import torch
import torch.nn as nn

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    """
        put a variable to a tensor
    """
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    """
        convert a tensor to numpy variable
    """
    return tensor.to('cpu').detach().numpy()