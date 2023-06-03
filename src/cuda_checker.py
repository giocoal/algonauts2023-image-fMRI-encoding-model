import torch

def cuda_torch_check():
    # Check if GPU is available and torch is using it
    print("Check if GPU is available and if torch is using it ..")
    print("\n")
    print("Torch Cuda is available?")
    print(torch.cuda.is_available())
    print("Torch Cuda device count is :")
    print(torch.cuda.device_count())
    print("Torch Cuda current device is :")
    print(torch.cuda.current_device())
    print("Torch Cuda device is :")
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))
    print("Pytorch version：")
    print(torch.__version__)
    print("CUDA Version: ")
    print(torch.version.cuda)
    print("cuDNN version is :")
    print(torch.backends.cudnn.version())
    print("\n")
    
def memory_checker():
    print(f'Total GPU Memory: {torch.cuda.mem_get_info()[1] * 1e-6} MB')
    print(f'Free GPU Memory: {torch.cuda.mem_get_info()[0] * 1e-6} MB')
    print(f'Cuda Reserved Memory: {torch.cuda.memory_reserved() * 1e-6} MB' )
    print(f'Cuda Allocated Memory: {torch.cuda.memory_allocated() * 1e-6} MB')