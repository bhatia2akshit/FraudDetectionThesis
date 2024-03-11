import os
import shutil


def rename_images(path):
    # Get the list of images present in this folder
    images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # Rename each file in this list in the numbered format starting from 1.jpg
    for i, image in enumerate(images, start=1):
        # Extract the file extension
        _, ext = os.path.splitext(image)
        # Construct the new file name
        new_name = f"{i}{ext}"
        # Rename the file
        os.rename(os.path.join(path, image), os.path.join(path, new_name))


def add_images_to_folder(source_folder, destination_folder):
    # Get list of images from source folder
    source_images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Get number of images already existing in the destination folder
    destination_images = [f for f in os.listdir(destination_folder) if os.path.isfile(os.path.join(destination_folder, f))]
    num_existing_images = len(destination_images)

    # Add images from source folder to destination folder, renaming each image's name
    for i, image in enumerate(source_images, start=num_existing_images + 1):
        # Construct the new file name
        new_name = f"{i}.jpg"
        # Copy the file from source to destination with the new name
        shutil.copyfile(os.path.join(source_folder, image), os.path.join(destination_folder, new_name))
    print(f"Copied")

# Example usage:
# Path to the folder containing images
source_folder = "/data/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/Dataset1/train/gaugan"
destination_folder="/data/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/Dataset1/train/fake"
folder_path = '/data/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/Dataset1/train/real'
# Call the function to rename images
rename_images(folder_path)
# add_images_to_folder(source_folder, destination_folder)