# DGHDGH
Source code for ICLR 2026 paper “Deep Global-sense Hard-negative Discriminative Generation Hashing for Cross-modal Retrieval”.

## Training

### Processing dataset
Refer to [DSPH](https://github.com/QinLab-WFU/DSPH)

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start
> python main.py
