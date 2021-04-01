from transfer.downstream.finetune.env.imports import torch
from transfer.downstream.finetune.io_.logger import printing


def use_gpu_(use_gpu, verbose=0):
    if use_gpu is not None and use_gpu:
      assert torch.cuda.is_available() , "ERROR : use_gpu was set to True but cuda not available "
    use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
    printing("HARDWARE : use_gpu set to {} ", var=[use_gpu], verbose=verbose, verbose_level=1)
    return use_gpu


def printout_allocated_gpu_memory(verbose, comment):

    if verbose == "gpu":
        try:
            printing("GPU {} {}",var=[comment, torch.cuda.memory_allocated()], verbose=verbose, verbose_level="gpu")
        except Exception as e:
            print(e)

