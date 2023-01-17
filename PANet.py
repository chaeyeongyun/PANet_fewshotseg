import models
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import FewshotData
from tqdm import tqdm

class PANet():
    def __init__(self, opt):
        self.device_setting(opt.device)
        self.feature_extractor = models.MVGG16(init_weights=True).to(self.device)
        if opt.mode == 'train':
            self.half = opt.half
            self.batchsize = opt.batchsize
            self.scaler = opt.scaler
            self.align_loss_scaler = opt.align_loss_scaler
            self.dataset = FewshotData(data_path=opt.data_path, mode=opt.mode, 
                                       n_ways=opt.n_ways, n_shots=opt.n_shots,n_query=opt.n_query,
                                       resize=(512, 512), transforms=None)
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(self.dataset), eta_min=1e-7, verbose=True)
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
            self.ckpoint_path = os.path.join(opt.save_path, 'ckpoints'+str(len(os.listdir(opt.save_path))))
            os.makedirs(self.ckpoint_path)
        
    def train(self,):
        trainloader = DataLoader(self.dataset, self.batchsize)
        best_loss = 100
        for iter, batch in tqdm(enumerate(trainloader)):
            '''
            support_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            support_fg_masks: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            support_bg_masks: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            query_imgs: query images
                n_query x [B x 3 x H x W], list of tensors
            query_labels: query labels
                n_query x [B x H x W], list of tensors
            '''
            support_imgs, support_fg_masks, support_bg_masks, query_imgs, query_labels = \
                batch['support_imgs'], batch['support_fg_masks'], batch['support_bg_masks'], batch['query_imgs'], batch['query_labels']
            query_labels = torch.cat(query_labels, dim=0).long().to(self.device) # (n_query*B, H, W)
            self.optimizer.zero_grad()
    
            n_ways = len(support_imgs)  # 
            n_shots = len(support_imgs[0])
            n_query = len(query_imgs)
            
            
            input = torch.cat([torch.cat(way, dim=0) for way in support_imgs]+[torch.cat(query_imgs, dim=0)], dim=0)  # ((n_waysxshotxB + n_queryxB, 3, H, W)
            input = input.to(self.device)
            
            ### extrac features ###
            scaler = torch.cuda.amp.GradScaler(enabled=self.half)
            inp_size = tuple(input[0].shape[-2:]) # input image size
            with torch.cuda.amp.autocast(enabled=self.half):
                features = self.feature_extractor(input)
            
            feat_h, feat_w = features.shape[-2:]
            support_features = features[:n_ways*n_shots*self.batchsize].view(n_ways, n_shots, self.batchsize, -1, feat_h, feat_w) #(n_ways, n_shots, B, C, feath, featw)
            query_features = features[n_ways*n_shots*self.batchsize:].view(n_query, self.batchsize, -1, feat_h, feat_w)
            
            fg_mask = torch.stack([torch.stack(way, dim=0) for way in support_fg_masks], dim=0) # (n_ways, n_shots, B, H, W)
            bg_mask = torch.stack([torch.stack(way, dim=0) for way in support_bg_masks], dim=0) # (n_ways, n_shots, B, H, W)
            fg_mask, bg_mask = fg_mask.to(self.device), bg_mask.to(self.device)
            # each batch is one episode
            outputs = []
            align_loss = 0
            for epi in range(self.batchsize):
                ### extract prototype ###
                map_fg_fts = [[self.map(support_features[way, shot, [epi]], fg_mask[way, shot, [epi]]) 
                               for shot in range(n_shots)]
                               for way in range(n_ways)] # 
                map_bg_fts = [[self.map(support_features[way, shot, [epi]], bg_mask[way, shot, [epi]]) 
                               for shot in range(n_shots)]
                               for way in range(n_ways)]
                fg_prototypes, bg_prototype = self.get_prototype(map_fg_fts, map_bg_fts)
                prototypes = [bg_prototype] + fg_prototypes
                # distance between features (query) and prototypes
                # query feature -(n_query, C, H', W'), prototype (1, C) -> (1, C, 1, 1)
                distance = [F.cosine_similarity(query_features[:, epi], prototype[..., None, None], dim=1)*self.scaler for prototype in prototypes] # (1+n_ways) x (n_query, H', W')
                pred = torch.stack(distance, dim=1) # (n_query, 1+n_ways, H', W')
                outputs.append(F.interpolate(pred, size=inp_size, mode='bilinear'))
                ## align loss 
                with torch.cuda.amp.autocast(enabled=self.half):
                    align_loss += self.align_loss(query_features[:, epi], pred, support_features[:, :, epi],
                                                fg_mask[:, :, epi], bg_mask[:, :, epi])
            query_pred = torch.stack(outputs, dim=1) # (n_query, B, 1+n_ways, H, W)
            query_pred = query_pred.view(-1, *query_pred.shape[2:]) # (n_query*B, 1+n_ways, H, W)
            with torch.cuda.amp.autocast(enabled=self.half):
                align_loss /= self.batchsize
                ce_loss = self.ce_loss(query_pred, query_labels)
                total_loss = ce_loss + align_loss * self.align_loss_scaler
            iter_loss = total_loss.item()
            if best_loss >= iter_loss:
                best_loss = iter_loss
                torch.save(self.feature_extractor.state_dict(), os.path.join(self.ckpoint_path, f'best_loss_{iter}iter.pth'))    
            scaler.scale(total_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.lr_scheduler.step()
            print(f'{iter} iter loss: {iter_loss}')
            torch.save(self.feature_extractor.state_dict(), os.path.join(self.ckpoint_path, f'{iter}iter.pth'))
        torch.save(self.feature_extractor.state_dict(), os.path.join(self.ckpoint_path, f'model_last.pth'))
            
    def map(self, features, mask):
        ''' 
        masked average pooling
        features shape: (1, C, H', W')
        mask shape: (1, H, W) -> unsqueeze is needed to multiply to features
        '''
        mask_4d = torch.unsqueeze(mask, dim=0)
        fts = F.interpolate(features, size=mask.shape[-2:], mode='bilinear')
        masked_features =  torch.sum(fts * mask_4d, dim=(-1, -2)) / (mask_4d.sum(dim=(-2,-1)) + 1e-5)
        # masked_features = torch.nan_to_num(masked_features, nan = 0.0, posinf=1e10, neginf=-1e10)
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
        query_prototypes = query_prototypes / (query_masks.sum(dim=(0, -2, -1)) + 1e-5) # average
        # compute loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways: continue
            bg_prototype, fg_prototype = query_prototypes[[0]], query_prototypes[[way+1]] # (1, C)
            for shot in range(n_shots):
                features = sup_ft[way, [shot]] #(1, C, H', W')
                supft_qproto_dist = [F.cosine_similarity(features, bg_prototype[..., None, None], dim=1)*self.scaler,
                                     F.cosine_similarity(features, fg_prototype[..., None, None], dim=1)*self.scaler]
                sup_pred = torch.stack(supft_qproto_dist, dim=1) # (1, 2, H', W')
                sup_pred = F.interpolate(sup_pred, size=inp_size, mode='bilinear') # (1, 2, H, W)
                # binary label (GT). fg:1, bg:0, (H, W)
                sup_label = torch.full_like(fg_mask[way, shot], -1).long().to(features.device)
                sup_label[fg_mask[way, shot]==1] = 1
                sup_label[bg_mask[way, shot]==1] = 0
                loss += F.cross_entropy(sup_pred, sup_label.unsqueeze(0), ignore_index=255) / (n_shots*n_ways)
        return loss
    
    def device_setting(self, device):
        if device != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda:'+device)
        elif device == '-1':
            self.device = torch.device('cpu')
        else: raise Exception('device number is not available or torch.cuda.is_available is False')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='mode')
    parser.add_argument('--half', type=bool, default=False, help='set True to use mixed precision')
    parser.add_argument('--data_path', type=str, default='../data/voc2012', help='root path of dataset')
    parser.add_argument('--save_path', type=str, default='./train', help='path for saving')
    parser.add_argument('--device', type=str, default='-1', help='gpu number to use. -1 is cpu')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batchsize', type=int, default=1, help='batch size')
    parser.add_argument('--scaler', type=float, default=20, help='scaler for cosine similarity')
    parser.add_argument('--align_loss_scaler', type=float, default=1, help='scaler for align loss')
    parser.add_argument('--n_ways', type=int, default=3, help='the number of ways')
    parser.add_argument('--n_shots', type=int, default=2, help='the number of shots')
    parser.add_argument('--n_query', type=int, default=1, help='the number of queries')
    opt = parser.parse_args()
    panet = PANet(opt)
    panet.train()