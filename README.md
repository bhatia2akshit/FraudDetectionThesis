## Introduction

This project explores image authenticity classification, focusing on generating and distinguishing between real and AI-generated images. Developed as part of a master’s thesis, the codebase consists largely of self-written implementations.

### Project Overview

The project includes two main components:
1. **Image Collection & Generation**: Real images are sourced from the COCO and LSUN datasets, and fake images are generated based on real-image captions using Stable Diffusion and GAUGAN. Vision transformers are used to generate captions, with image processing, feature extraction, and caption generation using a ViT image processor and a GPT-2 based tokenizer.
   
2. **Classification of Real vs. Fake Images**: Neural networks are trained to classify images as real or fake. Two datasets are created: 
   - **Dataset1**: Includes general-themed images from the COCO dataset in the train folder and LSUN bedroom images in the test folder.
   - **Dataset2**: Contains LSUN Bedroom images in the train folder and LSUN Diningroom images in the test folder.

### Classification Algorithms

Four neural network-based algorithms are implemented for classification:
- **CLIPBasedSigmoid**: Uses OpenAI’s CLIP model to extract image features, applying a sigmoid layer for classification.
- **ResNet50**: A CNN-based approach for classification.
- **DCT-based CNN**: Utilizes Discrete Cosine Transform features for classification.
- **DIRE**: Implements a DDIM-based diffusion algorithm to reconstruct and compare images, classifying similar images as fake.

### Results

The study found that classification algorithms struggle to accurately classify images when trained on unknown-generation methods, especially with differing train and test image themes. This highlights limitations in current classification algorithms, challenging claims of generalizability in AI-driven image classification.
