---
pipeline_tag: zero-shot-image-classification
tags:
- open_clip
- clip
- vision
- image-text-retrieval
- laion400m
---

# ViT-B-32 OpenCLIP Model on LAION-400M

This is a ViT-B-32 model trained using [OpenCLIP](https://github.com/mlfoundations/open_clip) on the LAION-400M dataset. 

## Training Details

The model was trained with the following configuration:
- **Model Architecture**: ViT-B-32
- **Dataset**: LAION-400M 
- **Number of Samples**: 400M (~ 268,836,185 filtered samples used)
- **Hardware**: 2 Nodes, each with 4 H200 141GB GPUs (Total 8 GPUs)
- **Batch Size (per GPU)**: 4096
- **Precision**: `amp_bfloat16`
- **Total Epochs**: 32
- **Warmup Steps**: 2000

Additional specific performance-enhancing flags enabled during training: `--torchcompile`, `--local-loss`, and `--gather-with-grad`.

## Evaluation

- **Eval Epoch**: 0
- **imagenet-zeroshot-val-top1**: 0.6086
- **imagenet-zeroshot-val-top5**: 0.8632

## Usage

```python
import torch
import open_clip

# Load the model directly from huggingface
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='hf-hub:lingkai/open-clip')

tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Example inference
image = preprocess(Image.open("astronaut.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```
