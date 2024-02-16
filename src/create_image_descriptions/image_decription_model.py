import os
from PIL import Image
import pandas as pd

import torch
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

# load a fine-tuned image captioning model and corresponding tokenizer and image processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Function to preprocess images and generate captions
def process_images(images):
    generated_texts = []
    index=0
    for image in images:
        index+=1
        if index%300==0:
            print(f'{index} images have been processed. The last description is: {generated_texts[-1]}')
        opened_image = Image.open(image_folder_path+image)
        pixel_values = image_processor(opened_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_texts.append([image, generated_text])  # appends the path along with generated text
    return generated_texts

# Split image URLs into batches
image_folder_path = '/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/data/real/lsun/bedroom/'
images = os.listdir(image_folder_path)
batch_size = 32
image_batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]

# Process images in batches
all_generated_texts = []
for batch in image_batches:
    generated_texts_batch = process_images(batch)
    all_generated_texts.extend(generated_texts_batch)

print('images are processed')
df = pd.DataFrame(all_generated_texts)
df.to_csv('/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/data/captions/lsun/bedroom/captions.csv')

# Print generated captions
for text in all_generated_texts[:3]:
    print(text)
