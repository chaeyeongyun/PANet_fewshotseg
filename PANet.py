import models
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import FewshotData
from warmup_scheduler import GradualWarmupScheduler


class PANet():
    def __init__(self, opt):
        self.device_setting(opt.device)
        self.batchsize = opt.batchsize
        
        self.feature_extractor = models.MVGG16(init_weights=True)
        self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
        # self.lr_scheduler = 
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=opt.ignore_idx)
        #TODO: align loss
        # self.align_loss = 
    def train(self,):
        #TODO: load data
        dataset = make_data('...')
        trainloader = DataLoader(dataset, self.batchsize)
        for iter, batch in enumerate(trainloader):
            support_imgs = [[shot.to(self.device) for shot in way] for way in batch['support imgs']]
            support_fg_masks = [[shot.to(self.device) for shot in way]for way in batch['support_fg_masks']]
            support_bg_masks = [[shot.to(self.device) for shot in way] for way in batch['support_bg_masks']]
            query_imgs =  [query_img.to(self.device) for query_img in batch['query_imgs']]
            query_labels = [query_label.to(self.device) for query_label in batch['query_labels']]
            
            self.optimizer.zero_grad()
            # supp_imgs: support images
            #     way x shot x [B x 3 x H x W], list of lists of tensors
            # fore_mask: foreground masks for support images
            #     way x shot x [B x H x W], list of lists of tensors
            # back_mask: background masks for support images
            #     way x shot x [B x H x W], list of lists of tensors
            # qry_imgs: query images
            #     N x [B x 3 x H x W], list of tensors
            n_ways = len(support_imgs)  # 
            n_shots = len(support_imgs[0])
            n_query = len(query_imgs)
            
            # feature extractor의 input은 support에서 1-way 내 모든 shot 이미지들 배치방향 concat하고 -> (shotxB, 3, H, W) 이거를 n_way개에 다 해서 리스트를 만들고
            # [(shotxB, 3, H, W) x n_ways]여기에 쿼리이미지들까지 (n_queryxB, 3, H, W)해서 concat하면
            # ((n_waysxshotxB + n_queryxB, 3, H, W)의 input이 만들어짐. 이것을 input으로 넣음
            input = torch.cat([torch.cat(way, dim=0) for way in support_imgs]+[torch.cat(query_imgs, dim=0)], dim=0)
            features = self.feature_extractor(input)
            feat_h, feat_w = features.shape[-2:]
            support_features = features[:n_ways*n_shots*self.batchsize].view(n_ways, n_shots, self.batchsize, -1, feat_h, feat_w) #(n_ways, n_shots, B, C, feath, featw)
            query_features = features[n_ways*n_shots*self.batchsize:].view(n_query, self.batchsize, -1, feat_h, feat_w)
            fg_mask = torch.stack([torch.stack(way, dim=0) for way in support_fg_masks], dim=0) # (n_ways, n_shots, B, H, W)
            bg_mask = torch.stack([torch.stack(way, dim=0) for way in support_bg_masks], dim=0) # (n_ways, n_shots, B, H, W)
            # each batch is one episode
            outputs = []
            align_loss = 0
            for epi in range(self.batchsize):
                ### extract prototype ###
                map_fg_fts = [[self.map(support_features[way, shot, [epi]], fg_mask[way, shot, [epi]]) 
                               for shot in range(n_shots)]
                               for way in range(n_ways)]  # epi인덱스에서 [epi]로 넣어주면 차원을 축소하지 않고 epi 에 해당하는 차원은 1 차원으로 남는다
                map_bg_fts = [[self.map(support_features[way, shot, [epi]], bg_mask[way, shot, [epi]]) 
                               for shot in range(n_shots)]
                               for way in range(n_ways)]
                fg_prototypes, bg_prototype = self.get_prototype(map_fg_fts, map_bg_fts)
                prototypes = [bg_prototype] + fg_prototypes
            #TODO: 여기부터 시작해야쥬 프로토타입까지 계산했고 이제 distance 연산하는 부분!
            # distance = 
            # pred = 
            # outputs += [F.interpolate(pred, size=imgsize, mode='bilinear')]
            ### compute the distance ###
            
            
            # align_loss += self.align_loss(query_features[:, epi], pred, support_features[:, :, epi],
            #                               fg_mask[:, :, epi], bg_mask[:, :, epi])
            
            
    def map(self, features, mask):
        ''' 
        masked average pooling
        features shape: (1, C, H, W)
        mask shape: (1, H, W) -> unsqueeze is needed to multiply to features
        '''
        masked_features = F.interpolate(features, size=mask.shape[-2:], mode='bilinear') * \
            torch.unsqueeze(mask, dim=0)
        masked_features = torch.sum(masked_features, dim=(2,3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # (1, C)
    
    def get_prototype(self, map_fg_fts, map_bg_fts):
        """average the features that is applied map to extract prototype

        Args:
            map_fg_fts : a list of list of features applied map with fore ground mask n_ways x n_shots x Tensor(1, C)
            map_bg_fts : a list of list of features applied map with fore ground mask n_ways x n_shots x Tensor(1, C)
        """
        n_ways, n_shots = len(map_fg_fts), len(map_fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in map_fg_fts] # list of mean of shots in each way -> 각 클래스에 대한 prototype
        bg_prototype = sum([sum(way) / n_shots for way in map_bg_fts]) / n_ways # mean of mean of shots in ways -> 그냥 background 하나의 prototype
        return fg_prototypes, bg_prototype
    
    def device_setting(self, device):
        if device != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda'+device)
        elif device == '-1':
            self.device = torch.device('cpu')
        else: raise Exception('device number is not available or torch.cuda.is_available is False')
    
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, deault='0', help='gpu number to use. -1 is cpu')
    parser.add_argument('--init_lr', type=float, deault=1e-4, help='initial learning rate')
    parser.add_argument('--ignore_idx', type=float, deault=1e-4, help='index that you want to ignore')
    parser.add_argument('--batchsize', type=float, deault=1e-4, help='batch size')