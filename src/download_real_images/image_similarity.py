from transformers import CLIPModel, CLIPProcessor
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import DBSCAN


# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

main_folder_path = '/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/'
image_folder_path = 'data/real/lsun/bedroom/'


def read_images():
    complete_path = main_folder_path + image_folder_path
    image_paths = [complete_path + image_path for image_path in os.listdir(complete_path)]
    return image_paths


def encode_images(image_paths):
    images = [Image.open(image_path) for image_path in image_paths]
    image_tensors = [processor(images=image, return_tensors="pt")["pixel_values"] for image in images]
    image_embeddings = torch.stack(image_tensors).numpy()
    return image_embeddings


def compute_similarity_matrix(image_embeddings):
    flattened_embeddings = image_embeddings.reshape(image_embeddings.shape[0], -1)
    similarity_matrix = cosine_similarity(flattened_embeddings, flattened_embeddings)

    return similarity_matrix


def save_heatmap(similarity_matrix, image_paths):
    heatmap = sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=image_paths, yticklabels=image_paths)
    heatmap.figure.savefig(main_folder_path + 'heatmap.png', format='png')
    plt.close()


def count_clusters(similarity_matrix, threshold):
    # Convert similarity matrix to distances (DBSCAN uses distances)
    distances = 1 - similarity_matrix

    # Apply DBSCAN
    dbscan = DBSCAN(eps=threshold, min_samples=2, metric='precomputed')
    labels = dbscan.fit_predict(distances)

    # Count the number of clusters (excluding noise, labeled as -1)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return num_clusters

image_paths = read_images()
print('start building embeddings')
image_embeddings = encode_images(image_paths)
print('start computing similarity')
similarity_matrix = compute_similarity_matrix(image_embeddings)

threshold1 = 0.85  # Set your desired similarity threshold
threshold2 = 0.9
threshold3 = 0.7
# Count the number of elements exceeding the threshold
# import numpy as np

# Exclude diagonal elements
non_diagonal_similarity_matrix = np.copy(similarity_matrix)
np.fill_diagonal(non_diagonal_similarity_matrix, 0)
matrix = np.triu(non_diagonal_similarity_matrix)


threshold1 = 0.85
num_clusters1 = count_clusters(matrix, threshold1)
print(f'Number of similar image clusters (> {threshold1}): {num_clusters1}')

threshold2 = 0.90
num_clusters2 = count_clusters(matrix, threshold2)
print(f'Number of similar image clusters (> {threshold2}): {num_clusters2}')

threshold3 = 0.70
num_clusters3 = count_clusters(matrix, threshold3)
print(f'Number of similar image clusters (> {threshold3}): {num_clusters3}')