# Image Captioning with CLIP-Guided Training

## Overview

This project implements an image captioning system that generates descriptive captions for images using a combination of vision and language models. The system is trained using CLIP-guided optimization to ensure semantic alignment between generated captions and image content.

## Key Components

### Model Architecture

- **Image Encoder**: MobileNetV2 for efficient image feature extraction
- **Text Decoder**: DistilGPT2 for caption generation
- **CLIP Integration**: Uses CLIP model for semantic alignment between images and captions

### Training Strategy

The model employs a multi-objective training approach with three loss components:

1. Cosine Similarity Loss: Ensures alignment between image and caption embeddings
2. Contrastive Loss: Improves discrimination between different image-caption pairs
3. Diversity Loss: Encourages varied caption generation

## Project Structure

- `train.py`: Main training script with training and validation loops
- `model.py`: Model architecture definitions including ImageEncoder, ImagePrefixMapper, and ImageCaptioningModel
- `evaluate.py`: Evaluation scripts for computing alignment scores
- `data_loader.py`: Data loading utilities using FiftyOne dataset
- `utils.py`: Utility functions for loss calculations and image processing
- `early_stopping.py`: Implementation of early stopping mechanism

## Dataset

The project uses the Open Images V7 dataset, specifically focusing on "Home appliance" category images:

- Training set: 10,000 images
- Validation set: 100 images

## Requirements

- PyTorch
- Transformers (Hugging Face)
- FiftyOne
- NLTK
- CLIP
- tqdm

## Usage

### Training

```python
python train.py
```

### Evaluation

```python
python evaluate.py
```

## Model Checkpoints

Checkpoints are saved every 10 epochs in the `config_3_checkpoints` directory, containing:

- Model state
- Optimizer state
- Training and validation losses

## Performance Metrics

The model's performance is evaluated using:

- CLIP-based alignment scores
- Baseline comparison with generic captions
- Loss metrics for similarity, contrastiveness, and diversity

## Future Improvements

- Experiment with different backbone architectures
- Implement beam search for caption generation
- Add support for more diverse datasets
- Optimize hyperparameters for better performance

## Acknowledgments

- CLIP model from OpenAI
- Open Images Dataset
- FiftyOne Dataset Zoo
