# Leveraging Text Localization for Scene Text Removal via Text-aware Masked Image Modeling 

This is a pytorch implementation for paper [TMIM](https://arxiv.org/abs/2409.13431)

## Installation

### 1.Requirements

- Python==3.8.12
- Pytorch==1.11.0
- CUDA==11.3

```bash
conda create -n tmim python==3.8.12
conda activate tmim
pip install --upgrade pip
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html 
pip install -r requirements.txt
```

### 2.Datasets
- Create a "data" folder. Download text removal dataset ([SCUT-Enstext](https://github.com/HCIILAB/SCUT-EnsText)) and text detection datasets([TextOCR](https://textvqa.org/textocr/dataset/), [Total-Text](https://github.com/cs-chan/Total-Text-Dataset), [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=downloads), [COCO-Text](https://bgshih.github.io/cocotext/#h2-download), [MLT19](https://rrc.cvc.uab.es/?ch=15&com=downloads), [ArT](https://rrc.cvc.uab.es/?ch=14&com=downloads), [lsvt](https://rrc.cvc.uab.es/?ch=16&com=downloads)(fullly annotated), [ReCTS](https://rrc.cvc.uab.es/?ch=12&com=downloads)). 
- Create the coco-style annotations for text detection datasets with the code in utils/prepare_dataset/ (or download them from [here(data.zip)](https://drive.google.com/drive/folders/1g3kLDe-WuSKmag978XhRRoc9Wg9riy7r?usp=sharing). 
- The structure of the data folder is shown below.

  ```bash
  data
  ├── text_det
  │   ├── art
  │   │   ├── train_images
  │   │   └── annotation.json
  │   ├── cocotext
  │   │   ├── train2014
  │   │   └── cocotext.v2.json
  │   ├── ic15
  │   │   ├── train_images
  │   │   └── annotation.json 
  │   ├── lsvt
  │   │   ├── train_images
  │   │   └── annotation.json 
  │   ├── mlt19
  │   │   ├── train_images
  │   │   └── annotation.json 
  │   ├── rects
  │   │   ├── img
  │   │   └── annotation.json 
  │   ├── textocr
  │   │   ├── train_images
  │   │   ├── TextOCR_0.1_train.json 
  │   │   └── TextOCR_0.1_val.json 
  │   └── totaltext
  │       ├── train_images
  │       └── annotation.json
  └── text_rmv
      └── SCUT-EnsText
          ├── train
          │   ├── all_images
          │   ├── all_labels
          │   └── mask
          └── test
              ├── all_images
              ├── all_labels
              └── mask

  ```

## Models

|        Model     |    Method   	|    PSNR 	  |    MSSIM 	  |     MSE 	  |     AGE 	  |     Download 	  |
|:---------------: |:------------:|:-----------:|:-----------:|:-----------:|:-----------:|:---------------:|
|    Uformer-B     |  Pretrained 	|    36.66   	|     97.66  	|    0.0637   |    1.70     |[uformer_b_tmim.pth](https://drive.google.com/file/d/1GQW6LRIXboQ2ECynjDEg3qu7W4DCrcAS/view?usp=drive_link)|    
|    Uformer-B     |  Fintuned   	|    37.42   	|     97.70   |    0.0459   |    1.52     |[uformer_b_tmim_str.pth](https://drive.google.com/file/d/1sMrVMnzplfCwGnyuw-GITypUMc2F5Icr/view?usp=drive_link)|
|      PERT        |  Pretrained 	|    34.51   	|     96.63  	|    0.1231   |    2.11     |[pert_tmim.pth](https://drive.google.com/file/d/1HB_Q5s0AwXa7ma-NuG3e6kKO6fO-UWbc/view?usp=drive_link)|    
|      PERT        |  Fintuned   	|    35.66   	|     97.18   |    0.0729   |    1.76     |[pert_tmim_str.pth](https://drive.google.com/file/d/1QVzbv4vcpEd5UOm0ubPJgtbjEzJdoJQh/view?usp=drive_link)|
|    EraseNet      |  Pretrained 	|    34.25   	|     97.03  	|    0.1141   |    2.23     |[erasenet_tmim.pth](https://drive.google.com/file/d/1utU-5b0kO795MJwGe8NujxHS1VX19HXH/view?usp=drive_link)|    
|    EraseNet      |  Fintuned   	|    35.47   	|     97.30   |    0.0765   |    1.95     |[erasenet_tmim_str.pth](https://drive.google.com/file/d/1ny9zQmbGqJKNn2ADFUGkQKBHJf3HgD6s/view?usp=sharing)|   


## Inference
- Download the pretrained models and run the following command for inference.
```bash
python -m torch.distributed.launch --master_port 29501 --nproc_per_node=1 demo.py --cfg configs/uformer_b_str.py --resume path/to/uformer_b_tmim_str.pth --test-dir path/to/image/folder --visualize-dir path/to/result/folder
```

## Training and Testing
- Set the "snapshot_dir"(The location for saving the checkpoints) and "dataroot"(The location of the datasets) in configs/*.py
- Erasenet and Pert require 4 1080ti GPUs. Uformer requires 8 1080ti GPUs

### 1.Pretraining
 - Run the following command to pretrain the model on text detection datasets.
```bash
python -m torch.distributed.launch --master_port 29501 --nproc_per_node=8 train.py --cfg configs/uformer_b_tmim.py --ckpt-name uformer_b_tmim --save-log 
```
 - Run the following command to test the performance of the pretrained model.
 ```bash
python test.py --cfg configs/uformer_b_tmim.py --ckpt-name uformer_b_tmim/latest.pth --save-log --visualize
```

### 2.Finetuning
 - Run the following command to finetune the model on text removal datasets.
```bash
python -m torch.distributed.launch --master_port 29501} --nproc_per_node=8 train.py --cfg configs/uformer_b_str.py --ckpt-name uformer_b_tmim_str --save-log --resume 'ckpt/uformer_b_tmim/latest.pth'
```
 - Run the following command to test the performance of the finetuned model.
 ```bash
python test.py --cfg configs/uformer_b_str.py --ckpt-name uformer_b_tmim_str/latest.pth --save-log --visualize
```

