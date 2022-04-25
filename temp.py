import gc
import torch
#del model
gc.collect()
torch.cuda.empty_cache()
