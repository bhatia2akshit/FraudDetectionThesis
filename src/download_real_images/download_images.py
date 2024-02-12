from datasets import load_dataset
import os

# Load LSUN Bedrooms dataset
dataset = load_dataset("pcuenq/stable_diffusion-bedrooms")

# Specify the directory where you want to save the images
output_directory = "./data/real/stable_diffusion/bedroom/"


def write_images(threshold=3000):
# Iterate through the dataset and save images
    for idx, example in enumerate(dataset['train']):
        if idx >= threshold:
            return

        image_filename = f"{idx + 1}.jpg"  # You can use a more descriptive filename if needed
        image_path = os.path.join(output_directory, image_filename)
        example['image'].save(image_path)

    print("Saving images complete.")

write_images()