import models
import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
from torch.utils.data import DataLoader

from data import FewshotData
from metric import Measurement
from config import make_config

class PANet():
    def __init__(self, opt):
        self.device_setting(opt.device)
        if opt.mode == 'train':
            self.feature_extractor = models.MVGG16(init_weights=True).to(self.device)
            self.half = opt.half
            self.batchsize = opt.batchsize
            self.num_workers = opt.num_workers
            self.scaler = opt.scaler
            self.align_loss_scaler = opt.align_loss_scaler
            self.dataset = FewshotData(data_path=opt.data_path, mode=opt.mode, 
                                       n_ways=opt.n_ways, n_shots=opt.n_shots,n_query=opt.n_query,
                                       resize=(512, 512), transforms=None)
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(self.dataset), eta_min=1e-7, verbose=False)
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
            self.ckpoint_path = os.path.join(opt.save_path, 'ckpoints'+str(len(os.listdir(opt.save_path))))
            os.makedirs(self.ckpoint_path)
        
        if opt.mode == 'test':
            self.feature_extractor = models.MVGG16(init_weights=False)
            self.feature_extractor.load_state_dict(torch.load(opt.weights))
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.n_ways = opt.n_ways
            self.batchsize = opt.batchsize
            self.scaler = opt.scaler
            self.dataset = FewshotData(data_path=opt.data_path, mode=opt.mode, 
                                       n_ways=opt.n_ways, n_shots=opt.n_shots,n_query=opt.n_query,
                                       resize=(512, 512), transforms=None)
            self.result_path = os.path.join(opt.save_path, 'test_results'+str(len(os.listdir(opt.save_path))))
            os.makedirs(self.result_path)
        
    def train(self,):
        torch.multiprocessing.set_sharing_strategy('file_system')
        trainloader = DataLoader(self.dataset, self.batchsize, shuffle=False, num_workers=self.num_workers)
        best_loss = 100
        self.feature_extractor.train()
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
                #W distance between features (query) and prototypes
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
    
    def test(self):
        testloader = DataLoader(self.dataset, 1, shuffle=False)
        self.feature_extractor.eval()
        measurement = Measurement(self.n_ways+1, ignore_idx=255)
        test_acc, test_miou = 0, 0
        test_precision, test_recall, test_f1score = 0, 0, 0
        iou_per_class = np.array([0]*(self.n_ways+1), dtype=np.float64)
        for batch in tqdm(testloader):
            '''
            B=1
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
            query_labels = torch.cat(query_labels, dim=0).long() # (n_query*B, H, W)
    
            n_ways = len(support_imgs)  # 
            n_shots = len(support_imgs[0])
            n_query = len(query_imgs)
            
            
            input = torch.cat([torch.cat(way, dim=0) for way in support_imgs]+[torch.cat(query_imgs, dim=0)], dim=0)  # ((n_waysxshotxB + n_queryxB, 3, H, W)
            input = input.to(self.device)
            
            ### extrac features ###
            inp_size = tuple(input[0].shape[-2:]) # input image size
            with torch.no_grad():
                features = self.feature_extractor(input)
            
            feat_h, feat_w = features.shape[-2:]
            support_features = features[:n_ways*n_shots*self.batchsize].view(n_ways, n_shots, self.batchsize, -1, feat_h, feat_w) #(n_ways, n_shots, B, C, feath, featw)
            query_features = features[n_ways*n_shots*self.batchsize:].view(n_query, self.batchsize, -1, feat_h, feat_w)
            
            fg_mask = torch.stack([torch.stack(way, dim=0) for way in support_fg_masks], dim=0) # (n_ways, n_shots, B, H, W)
            bg_mask = torch.stack([torch.stack(way, dim=0) for way in support_bg_masks], dim=0) # (n_ways, n_shots, B, H, W)
            fg_mask, bg_mask = fg_mask.to(self.device), bg_mask.to(self.device)
            # each batch is one episode
            outputs = []
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
                #W distance between features (query) and prototypes
                # query feature -(n_query, C, H', W'), prototype (1, C) -> (1, C, 1, 1)
                distance = [F.cosine_similarity(query_features[:, epi], prototype[..., None, None], dim=1)*self.scaler for prototype in prototypes] # (1+n_ways) x (n_query, H', W')
                pred = torch.stack(distance, dim=1) # (n_query, 1+n_ways, H', W')
                outputs.append(F.interpolate(pred, size=inp_size, mode='bilinear'))
                
            query_pred = torch.stack(outputs, dim=1) # (n_query, B, 1+n_ways, H, W)
            query_pred = query_pred.view(-1, *query_pred.shape[2:]) # (n_query*B, 1+n_ways, H, W)
            
            qrypred_cpu, qrylabels_cpu = query_pred.detach().cpu().numpy(), query_labels.cpu().numpy()
            acc_pixel, batch_miou, iou_ndarray, precision, recall, f1score = measurement(qrypred_cpu, qrylabels_cpu) 
            
            test_acc += acc_pixel
            test_miou += batch_miou
            iou_per_class += iou_ndarray
            
            test_precision += precision
            test_recall += recall
            test_f1score += f1score
         # test finish
        test_acc = test_acc / len(testloader)
        test_miou = test_miou / len(testloader)
        test_ious = np.round((iou_per_class / len(testloader)), 5).tolist()
        test_precision /= len(testloader)
        test_recall /= len(testloader)
        test_f1score /= len(testloader)
        
        result_txt = "load model(.pt) : %s \n Testaccuracy: %.8f, Test miou: %.8f" % (opt.weights,  test_acc, test_miou)       
        result_txt += f"\niou per class {test_ious}"
        result_txt += f"\nprecision : {test_precision}, recall : {test_recall}, f1score : {test_f1score} "
        print(result_txt)
        with open(os.path.join(self.result_path, 'metrics.txt'), 'w') as f:
            f.write(result_txt)
            
    def map(self, features, mask):
        ''' 
        masked average pooling
        features shape: (1, C, H', W')
        mask shape: (1, H, W) -> unsqueeze is needed to multiply to features
        '''
        mask_4d = torch.unsqueeze(mask, dim=0)
        fts = F.interpolate(features, size=mask.shape[-2:], mode='bilinear')
        masked_features =  torch.sum(fts * mask_4d, dim=(-1, -2)) / (mask_4d.sum(dim=(-2,-1)) + 1e-5)
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
    mode = parser.parse_args()
    mode = mode.mode
    
    if mode == 'train':
        opt = make_config(mode)
        panet = PANet(opt)
        panet.train()
    if mode in ['test']:
        opt = make_config(mode)
        panet = PANet(opt)
        panet.test()