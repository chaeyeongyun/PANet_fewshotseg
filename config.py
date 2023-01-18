from easydict import EasyDict

def make_config(mode):
    config = EasyDict({'mode':mode,
                        'data_path': '../data/voc2012', # root path of dataset
                        'device': '0', # gpu number to use. -1 is cpu
                        'n_ways': 20, # the number of ways
                        'n_shots': 2, # the number of shots
                        'n_query': 1, # the number of queries
                        'scaler': 20 # scaler for cosine similarity
                    })
    
    if mode == 'train':
        config.save_path = './train' # path to save ckpoints
        config.half = False # set True to use mixed precision
        config.num_workers = 4 # the number of workers to use in DataLoader class
        config.init_lr = 1e-4 # inital learning rate
        config.batchsize = 2 # batch size
        config.align_loss_scaler = 1. # scaler for align loss
    
    if mode == 'test':
        config.save_path = './test' # path to save results
        config.weights = '/content/few_shot_segmentation_pytorch/train/ckpoints1/best_loss_10iter.pth' # weights path to load
        config.batchsize = 1 # batch size
    return config