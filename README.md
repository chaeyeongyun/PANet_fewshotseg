# few-shot segmentation Pytorch  
Pytorch implementation of few-shot segmentation methods  
## model list  
- PANet | [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.pdf)  
---  
## How to use (example for PANet)  
You can modify options for train/test in config.py  
### Train
```python
python PANet.py --mode train
```

### Test
```python
python PANet.py --mode test
```