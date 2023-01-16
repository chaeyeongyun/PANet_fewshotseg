import models
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import FewshotData

class PANet():
    def __init__(self, opt):
        if opt.mode == 'train':
            self.device_setting(opt.device)
            self.batchsize = opt.batchsize
            self.scaler = opt.scaler
            self.align_loss_scaler = opt.align_loss_scaler
            self.feature_extractor = models.MVGG16(init_weights=True)
            #TODO: load data
            self.dataset = make_data('...')
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(self.dataset), eta_min=1e-7, verbose=True)
            self.ce_loss = nn.CrossEntropyLoss()
            self.ckpoint_path = os.path.join(self.save_path, 'ckpoints')
        
    def train(self,):
        trainloader = DataLoader(dataset, self.batchsize)
        best_loss = 100
        for iter, batch in enumerate(trainloader):
            support_imgs = [[shot.to(self.device) for shot in way] for way in batch['support imgs']]
            support_fg_masks = [[shot.to(self.device) for shot in way]for way in batch['support_fg_masks']]
            support_bg_masks = [[shot.to(self.device) for shot in way] for way in batch['support_bg_masks']]
            query_imgs =  [query_img.to(self.device) for query_img in batch['query_imgs']]
            #TODO: query labels shape?
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
            inp_size = tuple(input.shape[-2:]) # input image size
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
                # distance between features (query) and prototypes
                distance = [F.cosine_similarity(query_features[:, epi], prototype, dim=1)*self.scaler for prototype in prototypes]
                # query_features[:, epi] 는 (N_query, C, H, W), 각 prototype은 (1, C) 일텐데...
                # (N_query, H, W)의 way+1 (클래스 수 + 배경 클래스) 개 리스트가 될 것이고 이것을 stack해준다
                pred = torch.stack(distance, dim=1) # (n_query, (1+way), H', W')
                outputs.append(F.interpolate(pred, size=inp_size, mode='bilinear')) 
                ## align loss 
                align_loss += self.align_loss(query_features[:, epi], pred, support_features[:, :, epi],
                                              fg_mask[:, :, epi], bg_mask[:, :, epi])
            query_pred = torch.stack(outputs, dim=1) # query 예측 결과들 (n_query, num_epi, 1+n_ways, H, W)
            query_pred = query_pred.view(-1, *query_pred.shape[2:]) # (n_query*num_epi, 1+n_ways, H, W)
            align_loss /= self.batchsize
            ce_loss = self.ce_loss(query_pred, query_labels)
            total_loss = ce_loss + align_loss * self.align_loss_scaler
            if best_loss >= total_loss.item():
                best_loss = total_loss.item()
                torch.save(self.feature_extractor.state_dict(), os.path.join(self.ckpoint_path, f'best_loss_{iter}iter.pth'))    
            total_loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            
            torch.save(self.feature_extractor.state_dict(), os.path.join(self.ckpoint_path, f'{iter}iter.pth'))
        torch.save(self.feature_extractor.state_dict(), os.path.join(self.ckpoint_path, f'model_last.pth'))
            
    def map(self, features, mask):
        ''' 
        masked average pooling
        features shape: (1, C, H', W')
        mask shape: (1, H, W) -> unsqueeze is needed to multiply to features
        '''
        masked_features = F.interpolate(features, size=mask.shape[-2:], mode='bilinear') * \
            torch.unsqueeze(mask, dim=0)
        masked_features = torch.sum(masked_features, dim=(2,3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # (1, C)
        return masked_features
    
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
    
    def align_loss(self, q_ft, pred, sup_ft, fg_mask, bg_mask):
        """
        Compute the loss for the prototype alignment branch
        Args:
            q_ft: embedding features for query images of one episode
                shape: N x C x H' x W'
            pred: predicted segmentation score for one episode
                shape: N x (1 + Wa) x H x W
            sup_ft: embedding features for support images of one episode
                shape: Wa x Sh x C x H' x W'
            fg_mask: foreground masks for support images of one episode
                shape: way x shot x H x W
            bg_mask: background masks for support images of one episode
                shape: way x shot x H x W
        """
        n_ways, n_shots = len(fg_mask), len(fg_mask[0])
        inp_size = tuple(fg_mask.shape[-2:])
        # get prototype from query
        pred_result = pred.argmax(dim=1, keepdim=True) # (N_query, 1, H', W')
        binary_masks = [pred_result == i for i in range(1+n_ways)] # a list of (N_query, 1, H', W') tensors, each tensor have 0 or 1 values.
        # pred되지 않은 classe들 (way들) 은 skip 합니다
        skip_ways = [i  for i in range(1, n_ways+1) if binary_masks[i].sum() == 0] # 예측결과는 bg + n_ways이므로
        query_masks = torch.stack(binary_masks, dim=1).float() #(N_query, 1+n_ways, 1, H', W')
        query_prototypes = (q_ft.unsqueeze(1) * query_masks) # (N, 1, C, H', W')와 (N_query, 1+n_ways, 1, H', W') 의 곱이 broad castint의해 (N, 1+n_ways, C, H', W') 의 결과를 낸다.
                                            # 각 쿼리별 각 클래스에 대한 masked feature들이다.
        query_prototypes = torch.sum(query_prototypes, dim=(0, -2, -1)) # (1+n_ways, C)
        query_prototypes = query_prototypes / query_masks.sum(dim=(0, -2, -1) + 1e-5) # average
        # compute loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways: continue
            bg_prototype, fg_prototype = query_prototypes[[0]], query_prototypes[[way+1]] # (1, C)
            for shot in range(n_shots):
                features = sup_ft[way, [shot]] #(1, C, H', W')
                supft_qproto_dist = [F.cosine_similarity(features, bg_prototype, dim=1)*self.scaler,
                                     F.cosine_similarity(features, fg_prototype, dim=1)*self.scaler]
                sup_pred = torch.stack(supft_qproto_dist, dim=1) # (1, 2, H', W')
                sup_pred = F.interpolate(sup_pred, size=inp_size, mode='bilinear') # (1, 2, H, W)
                # binary label (GT). fg:1, bg:0, (H, W)
                sup_label = torch.full_like(fg_mask[way, shot], -1).long().to(features.device)
                sup_label[fg_mask[way, shot]==1] = 1
                sup_label[bg_mask[way, shot]==1] = 0
                loss += F.cross_entropy(sup_pred, sup_label.unsqueeze(0), ignore_index=-1) / (n_shots*n_ways)
        return loss
    
    def device_setting(self, device):
        if device != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda'+device)
        elif device == '-1':
            self.device = torch.device('cpu')
        else: raise Exception('device number is not available or torch.cuda.is_available is False')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, deault='train', help='mode')
    parser.add_argument('--save_path', type=str, deault='./train', help='path for saving')
    parser.add_argument('--device', type=str, deault='0', help='gpu number to use. -1 is cpu')
    parser.add_argument('--init_lr', type=float, deault=1e-4, help='initial learning rate')
    parser.add_argument('--batchsize', type=float, deault=2, help='batch size')
    parser.add_argument('--scaler', type=float, deault=20, help='scaler for cosine similarity')
    parser.add_argument('--align_loss_scaler', type=float, deault=1, help='scaler for align loss')