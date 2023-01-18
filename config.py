import argparse

def make_config(mode):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/voc2012', help='root path of dataset')
    parser.add_argument('--device', type=str, default='0', help='gpu number to use. -1 is cpu')
    
    parser.add_argument('--n_ways', type=int, default=10, help='the number of ways')
    parser.add_argument('--n_shots', type=int, default=2, help='the number of shots')
    parser.add_argument('--n_query', type=int, default=1, help='the number of queries')
    parser.add_argument('--scaler', type=float, default=20, help='scaler for cosine similarity')
    
    if mode == 'train':
        parser.add_argument('--save_path', type=str, default='./train', help='path for saving')
        parser.add_argument('--half', type=bool, default=False, help='set True to use mixed precision')
        parser.add_argument('--num_workers', type=int, default=4, help='the number of workers to use in DataLoader class')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='initial learning rate')
        parser.add_argument('--batchsize', type=int, default=2, help='batch size')
        parser.add_argument('--align_loss_scaler', type=float, default=1, help='scaler for align loss')
    
    if mode == 'test':
        parser.add_argument('--save_path', type=str, default='./test', help='path for saving')
        parser.add_argument('--weights', type=str, default='/content/few_shot_segmentation_pytorch/train/ckpoints1/best_loss_10iter.pth')
        parser.add_argument('--batchsize', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.mode = mode
    return opt