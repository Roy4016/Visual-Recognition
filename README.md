# HW2 

This homework implements an end-to-end object detection pipeline using PyTorch and the Faster R-CNN model with a ResNet-50 FPN backbone. It handles dataset loading, training, evaluation, and inference on a COCO-style dataset.

## Input Structure
To run this code, please convert the input into the following format.

```
.
├── data/
│   ├── train/            # Training images (1.png~)
│   ├── valid/            # Validation images (1.png~)
│   ├── test/             # Test images (1.png~)
│   ├── train.json        
│   ├── valid.json       

```

## Requirements

- Python 3.8+
- PyTorch >= 1.10
- torchvision
- pandas
- tqdm
- PIL

Install with pip:
```bash
pip install torch torchvision pandas tqdm pillow
```
or
```bash
pip install -r requirements.txt
```
## Training

Run the training:
```bash
python HW2.py --mode train
```
Run the Inference:
```bash
python HW2.py --mode infer
```



## Performance
![alt text](image.png)







