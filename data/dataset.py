import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import os
from PIL import Image
import random
import numpy as np
from collections import defaultdict

class FewshotData(Dataset):
    '''
    나는 일단
    1. n_way n_shots 1 3 H W 그니까 모든클래스를 shot개만큼 개지고 있는 support set이랑
    2. n_query 3 H W 의 query set을 만들어보고자 한다. 
    batch size를 1이상으로 한다면 아마.. 각각 1 n_way n_shots 1 3 H W, 1 n_query 3 H W 이 되겠지만 그러면 오히려 좋게 하나의 에피소드로 취급하면된다. 맞제
    fewshot style Dataset class
    support-query pairs x num_episodes
    '''
    def __init__(self, data_path, mode, n_ways, n_shots, n_query, num_episode, transforms=None) :
        super().__init__()
        self.data_path = data_path
        self.img_path = os.path.join(data_path, 'imgs')
        self.mask_path = os.path.join(data_path, 'masks')
        
        self.mode = mode
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_query = n_query
        self.num_episode = num_episode    
        self.classes = list(range(1, n_ways+1))
        self.cls_to_img = self._map_cls_to_img() # dict, key:cls number, value: a list of filename which include key class
        
    
    def __getitem__(self, index):
        # support sets
        # 다 뽑고 여기에 안들어가는 이미지들 중에서 query 뽑아야해요            
        episode = {'support_imgs':[],
                   'support_fg_masks':[],
                   'support_bg_masks':[],
                   'query_imgs':[],
                   'query_labels':[]}
        support_sets = []
        for cls in self.classes:
            # support_files = random.sample(self.cls_to_img[cls], self.n_shots) # cls에 해당하는 파일들중 shot개수만큼 랜덤으로 뽑음
            support_files = [self.cls_to_img[cls].pop() for _ in range(self.n_shots)] # 다음 에피소드에서 겹치면 안되니까 pop으로 아예 제거
            support_set += support_files
            sup_imgs = [TF.to_tensor(Image.open(os.path.join(self.img_path, file)).convert('RGB')) for file in support_files]
            #TODO: fg bg 구별 함수
            sup_masks = [self._to_bg_fg_mask(Image.open(os.path.join(self.mask_path, file) for file in support_files))]
            #TODO: small object 제외
            episode['support_imgs'].append(sup_imgs)
            episode['support_bg_masks'].append([i[0]for i in sup_masks])
            episode['support_fg_masks'].append([i[1]for i in sup_masks])
        # support 가 바뀌므로 query는 중복의 가능성이 있어도 될 것 같아사 그냥 support만 제외한 목록에서 랜덤 선택
        query_files =  random.sample(list(set(os.listdir(self.img_path)) - set(support_sets)), self.num_query) 
        qry_imgs = [TF.to_tensor(Image.open(os.path.join(self.img_path, file)).convert('RGB')) for file in query_files]
        #TODO: label화 함수
        qry_labels = [self._to_labelmap(Image.open(os.path.join(self.mask_path, file) for file in query_files))]
        episode['query_imgs'].append(qry_imgs)
        episode['query_labels'].append(qry_labels)
        return episode
        
    #TODO
    def _to_labelmap(self):
        pass
    
    #TODO
    def _to_bg_fg_mask(self, cls):
        pass
         
    #TODO
    def _cls_in_mask(mask):
        pass
    def _map_cls_to_img(self):
        cls_to_img = defaultdict(list)
        filenames = os.listdir(self.img_path)
        for file in filenames:
            mask = Image.open(os.path.join(self.mask_path, file))
            #TODO: extract classes from mask
            cls_list = _cls_in_mask(mask)
            for cls in cls_list:
                cls_to_img[cls].append(file)
        return cls_to_img
        
        
        
        
    def __len__(self):
        return self.num_episode
    def __getitem__(self, index):
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # self.mode = mode
        
        # self.img_path = os.path.join(data_path, mode, 'imgs')
        # self.mask_path = os.path.join(data_path, mode, 'mask')
        
        # self.imgs = os.listdir(self.img_path)
        
        # self.ways = range(1, n_ways+1)
        # self.n_ways = n_ways
        # self.n_shots = n_shots
        # self.max_iters = max_iters
        
        # self.transforms = transforms
    
    
    
    
    # def __len__(self):
    #     len(self.imgs)
    # def __getitem__(self, idx):
    #     filename = self.imgs[idx]
    #     # query
    #     qry_img = Image.open(os.path.join(self.qry_img_path, filename)).convert('RGB')
    #     #TODO: grayscale여부
    #     qry_mask = Image.open(os.path.join(self.mask_path, filename))
    #     if self.transforms != None:
    #         qry_img = self.transforms(qry_img)
    #     qry_img = TF.to_tensor(qry_img)
    #     #TODO: target class, mask 내의 클래스 중에 random하게 추출
    #     cls = random.choice(sorted(set(np.unique(qry_mask)) & self.ways))
        
        # support set
        # cls에 속하는 것들 중에 랜덤하게 shot개수만큼 뽑으면 되겠죵
        
        
        
        
        
