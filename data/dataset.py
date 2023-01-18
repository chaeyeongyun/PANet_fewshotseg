import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize
import os
from PIL import Image
import random
import numpy as np
from collections import defaultdict
from functools import reduce
from torchvision import transforms

class FewshotData(Dataset):
    """
    few-shot style dataset class for PASCALVOC2012-style dataset
    """
    def __init__(self, data_path, mode, n_ways, n_shots, n_query, resize=(512, 512), transforms=None) :
        super().__init__()
        self.data_path = data_path
        self.img_path = os.path.join(data_path, mode, 'imgs') 
        self.mask_path = os.path.join(data_path, mode, 'masks')
        
        self.mode = mode
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_query = n_query
        
        self.classes = list(range(1, n_ways+1)) # background 0, [1, n_way] foreground class
        self.cls_to_img = self._map_cls_to_img() # dict, key:cls number, value: a list of filename which include key class
        self.all_imgs = list(reduce(lambda x, y: x+y, list(self.cls_to_img.values())))
        self.transforms = transforms
        
        self.bilinear = Resize(size=resize, interpolation=Image.BILINEAR)
        self.nearest = Resize(size=resize, interpolation=Image.NEAREST)
        
    def __len__(self):
        ids = list(self.cls_to_img.keys())
        num_imgs = [len(self.cls_to_img[id]) for id in ids]
        return min(num_imgs)//self.n_shots
    
    def __getitem__(self, index):
        episode = {'support_imgs':[],
                   'support_fg_masks':[],
                   'support_bg_masks':[],
                   'query_imgs':[],
                   'query_labels':[]}
        support_sets = []
        for cls in self.classes:
            support_files = [self.cls_to_img[cls].pop() for _ in range(self.n_shots)] 
            support_sets += support_files
            if self.transforms == None:
                sup_imgs = [TF.to_tensor(self.bilinear(Image.open(os.path.join(self.img_path, file+'.jpg')).convert('RGB'))) for file in support_files]
                # [[bg mask, fg mask], [bg maks, fg mask], ...]
                sup_masks = [self._to_bg_fg_mask(torch.from_numpy(np.array(self.nearest(Image.open(os.path.join(self.mask_path, file+'.png'))))), cls, self.classes) for file in support_files]
            else: 
                sup_imgs = [TF.to_tensor(self.bilinear(self.transforms(Image.open(os.path.join(self.img_path, file+'.jpg')).convert('RGB')))) for file in support_files]
                # [[bg mask, fg mask], [bg maks, fg mask], ...]
                sup_masks = [self._to_bg_fg_mask(torch.from_numpy(np.array(self.nearest(self.transforms(Image.open(os.path.join(self.mask_path, file+'.png')))))), cls, self.classes) for file in support_files]
            #TODO: small object 제외
            episode['support_imgs'].append(sup_imgs)
            episode['support_bg_masks'].append([i[0]for i in sup_masks])
            episode['support_fg_masks'].append([i[1]for i in sup_masks])
        # support 가 바뀌므로 query는 중복의 가능성이 있어도 될 것 같아사 그냥 supp# jpg ort만 제외한 목록에서 랜덤 선택
        query_files =  random.sample(list(set(self.all_imgs) - set(support_sets)), self.n_query)
        if self.transforms == None:
            qry_imgs = [TF.to_tensor(self.bilinear(Image.open(os.path.join(self.img_path, file+'.jpg')).convert('RGB'))) for file in query_files]
            qry_labels = [torch.from_numpy(np.array(self.nearest(Image.open(os.path.join(self.mask_path, file+'.png'))))) for file in query_files]
        else:
            qry_imgs = [TF.to_tensor(self.bilinear(self.transforms(Image.open(os.path.join(self.img_path, file+'.jpg')).convert('RGB')))) for file in query_files]
            qry_labels = [torch.from_numpy(np.array(self.nearest(self.transforms(Image.open(os.path.join(self.mask_path, file+'.png')))))) for file in query_files]
        qry_labels = self._filter_qry_label(qry_labels)
        episode['query_imgs'].extend(qry_imgs)
        episode['query_labels'].extend(qry_labels)
        return episode
    
    def _to_bg_fg_mask(self, mask, cls, all_cls):
        """Comvert from VOC style segmentation mask to [back ground mask(binary), fore ground mask(binary)] list

        Args:
            mask (torch.Tensor): label map, 255 will be ignored 
        """
        # 해당 클래스와 같은 부분을 1로 하여 fg mask로 만들고
        # bg mask의 경우 에피소드 내의 모든 클래스에 해당하는 부분은 0으로 처리한다(배경아 아닌 것으로 처리)
        bg_mask = torch.where(mask!=cls, torch.ones_like(mask), torch.zeros_like(mask))
        fg_mask = torch.where(mask==cls, torch.ones_like(mask), torch.zeros_like(mask))
        for i in all_cls:
            bg_mask[mask==i] = 0
        return [bg_mask, fg_mask]
         
    def _map_cls_to_img(self):
        cls_to_img = defaultdict(list)
        filenames = os.listdir(self.img_path)
        for file in filenames:
            name = os.path.splitext(file)[0]
            mask = Image.open(os.path.join(self.mask_path, name+'.png'))
            cls_list = set(np.unique(mask)) - {0}
            for cls in cls_list:
                cls_to_img[cls].append(name)
        return cls_to_img
    def _filter_qry_label(self, qry_labels):
        '''qry label is a list of query images'''        
        for i in range(len(qry_labels)):
            qry_labels[i] = torch.where(qry_labels[i]<=self.n_ways, qry_labels[i], 255)
        return qry_labels
            
