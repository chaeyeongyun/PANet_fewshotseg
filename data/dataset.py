import torch
from torch.utils.data import Dataset
import os

class FewshotData(Dataset):
    '''
    fewshot style Dataset class
    support-query pairs x num_episodes
    '''
    def __init__(self, data_path, mode, num_class) :
        super().__init__()
        self.mode = mode
        
        self.img_path = os.path.join(data_path, mode, 'imgs')
        self.mask_path = os.path.join(data_path, mode, 'mask')
        
        self.classes = range(1, num_class+1)
        
        
        
        