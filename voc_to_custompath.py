from tqdm import tqdm
import os
import shutil

def main(voc_path, save_path):
    img_path = os.path.join(voc_path, 'JPEGImages')
    mask_path = os.path.join(voc_path, 'SegmentationClass')
    txt_path = os.path.join(voc_path, 'ImageSets', 'Segmentation')
    modes = ['train', 'val', 'test']
    for mode in modes:
        print(f'copying {mode} files...')
        img_save = os.path.join(save_path, mode, 'imgs')
        os.makedirs(img_save, exist_ok=True)
        mask_save = os.path.join(save_path, mode, 'masks')
        os.makedirs(mask_save, exist_ok=True)
        with open(os.path.join(txt_path, mode+'.txt'), 'r') as f:
            filenames = f.readlines()
        for file in tqdm(filenames):
            file = file.strip()
            shutil.copy(os.path.join(img_path, file+'.jpg'), os.path.join(img_save, file+'.jpg'))
            if mode != 'test':
                shutil.copy(os.path.join(mask_path, file+'.png'), os.path.join(mask_save, file+'.png'))
    
    
if __name__ == '__main__':
    voc_path = '/content/downloads/VOC2012'
    save_path = '/content/data/voc2012'
    main(voc_path, save_path)