# utils.py
import torch
import gc

def clear_gpu():
    """
    Clears GPU memory to avoid OOM crashes.
    Call this after heavy operations or after each training step.
    """
    gc.collect()            # free Python memory
    torch.cuda.empty_cache() # free GPU memory
