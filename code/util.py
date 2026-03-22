import os
import random
import shutil
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from scenes import Scene


def normalize(array: np.ndarray, as_matrix: bool = False) -> np.ndarray:
    vector = array.flatten() # Compress to single dimension

    # Make vector zero-mean
    vector -= np.mean(vector)

    # Make vector unit-length
    norm = np.linalg.norm(vector)
    if norm != 0:
        vector /= norm

    if as_matrix:
        # Return as a matrix
        return vector.reshape(array.shape)
    else:
        return vector

def get_training_images(training_path: str, scene: Scene = None):
    # Get all training images
    output = []
    if scene is None:
        for scene_dir in [os.path.join(training_path, d) for d in os.listdir(training_path) if
                          os.path.isdir(os.path.join(training_path, d))]:
            files = glob(os.path.join(scene_dir, "*.jpg"))
            output.extend(files)
    else:
        path = os.path.join(training_path, scene.directory)
        files = glob(os.path.join(path, "*.jpg"))
        output.extend(files)
    return output

def get_testing_images(testing_path: str):
    return sorted(glob(os.path.join(testing_path, "*.jpg")), key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))


def show_visual_words(kmeans, patches, num_words: int = 5):
    num_of_labels = len(set(kmeans.labels_))

    # Sample the visual words
    words = [random.randint(0, num_of_labels - 1) for _ in range(num_words)]
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Find the patches that belong to each visual word sampled
    patch_indices = [np.where(labels == w)[0] for w in words]
    fig, axes = plt.subplots(num_words, 2 + 2 * num_words, figsize=(15, 15))

    for idx, ax in enumerate(axes.flat):
        # The current visual word
        word_class = words[idx // (2 * num_words + 2)]
        mean_img = np.zeros((1,))
        if idx % (2 * num_words + 2) == 0:  # If the left-most column
            # Plot the representative vector (the cluster center)
            mean_img = cluster_centers[word_class].reshape((8, 8))
            ax.imshow(mean_img, cmap='gray')
            ax.set_title(f"#{word_class}")
            ax.axis('off')
            continue

        elif idx % (2 * num_words + 2) == 1:  # If the second left-most column
            # Make a gap to separate mean image from samples
            ax.axis('off')
            continue
        else:
            patch_index = (idx % (2 * num_words + 2)) - 1
            current_img = patches[patch_indices[idx // (2 * num_words + 2)][patch_index]].reshape((8, 8))
            ax.imshow(current_img, cmap='gray')
            sim = cosine_similarity(mean_img, current_img)
            ax.set_title(f"{sim * 100:.2f}%")
            ax.axis('off')

    fig.suptitle("Visual words | % similarity to mean image (Cosine Similarity)")
    plt.tight_layout()
    plt.show()

def show_word(image, label=None, title=None):
    """
    Display a single visual word.

    :param image: The visual word to show.
    :param label: The label of the visual word. (optional)
    :param title: The title of the diagram. (optional)
                  If a title is provided, it will override any
                  provided label.
    """
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")
    if title is not None:
        plt.title(title)
    elif label is not None:
        plt.title(f"Visual Word #{label}")
    plt.show()


def cosine_similarity(image1, image2):
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    # Compute the norms
    norm1 = np.linalg.norm(image1_flat)
    norm2 = np.linalg.norm(image2_flat)

    # One of the visual words (#9 with the default random seed)
    # will match patches that are pure black. The norm of these patches are 0.
    # This handles that case, and computes how similar image1 is to a pure black image.
    if norm2 == 0:
        # Similarity to a black image is proportional to the "closeness" of image1 to zero
        return 1.0 - (norm1 / np.linalg.norm(np.ones_like(image1_flat)))  # Normalize to [0, 1]

    # Compute cosine similarity for non-black images
    return np.dot(image1_flat, image2_flat) / (norm1 * norm2)

def add_gaussian_noise(img):
    """Add Gaussian noise to an image."""
    noise_factor = 0.1  # Adjust the noise factor to control noise level
    mean = 0.0  # Mean of the Gaussian noise
    std_dev = 0.1  # Standard deviation of the noise
    noise = np.random.normal(mean, std_dev, img.shape)  # Create Gaussian noise
    noisy_img = img + noise_factor * noise  # Add the noise to the image
    noisy_img = np.clip(noisy_img, 0., 1.)  # Ensure pixel values are in [0, 1]
    return noisy_img

def move_to_subfolder(path:str):
    # Check if the path exists and is a directory
    if not os.path.isdir(path):
        raise ValueError(f"The path {path} is not a valid directory.")

    # Define the path of the subfolder
    subfolder_path = os.path.join(path, 'unlabelled')

    # Create the subfolder if it doesn't exist
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Iterate through the files in the provided directory
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Move the file to the subfolder
            shutil.move(file_path, os.path.join(subfolder_path, filename))


def move_from_subfolder(path:str):
    # Check if the path exists and is a directory
    if not os.path.isdir(path):
        raise ValueError(f"The path {path} is not a valid directory.")

    # Define the path of the subfolder
    subfolder_path = os.path.join(path, 'unlabelled')

    # Check if the subfolder exists
    if not os.path.exists(subfolder_path):
        raise FileNotFoundError(f"The subfolder {subfolder_path} does not exist.")

    # Iterate through the files in the subfolder
    for filename in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, filename)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Move the file to the parent directory
            shutil.move(file_path, os.path.join(path, filename))

    # Delete the subfolder after moving the files
    os.rmdir(subfolder_path)

def submit_results(output, predictions, testing_path):
    images = get_testing_images(testing_path)

    # Create output file
    with open(output, 'w') as f:
        for v1, v2 in zip(images, predictions):
            f.write(f"{v1.split(os.sep)[-1]} {v2}\n")

def count_non_empty_lines(file_path):
    with open(file_path, 'r') as file:
        # Count lines that are not empty or only whitespace
        non_empty_line_count = sum(1 for line in file if line.strip())  # line.strip() removes leading/trailing whitespace
    return non_empty_line_count


def compare_predictions(pred1, pred2, output_path = "differences.csv"):
    mismatched_lines = [f"image,{pred1},{pred2}"]
    if count_non_empty_lines(pred1) != count_non_empty_lines(pred2):
        raise ValueError("Files do not have the same number of predictions.")
    total = count_non_empty_lines(pred1)
    with open(pred1, 'r') as file1, open(pred2, 'r') as file2:
        # Enumerate over both files line by line
        for line_num, (line1, line2) in enumerate(zip(file1, file2), start=1):
            # Compare lines
            if line1.strip() != line2.strip():
                class1 = line1.strip().split(" ")[-1]
                class2 = line2.strip().split(" ")[-1]
                mismatched_lines.append(f"{line_num - 1},{class1},{class2}")

    print(f"{pred1} and {pred2} agree on {total - len(mismatched_lines)}/{total} ({((total - len(mismatched_lines)) * 100 / total):.2f}%)")
    with open(output_path, "w+") as out:
        out.write("\n".join(mismatched_lines))

def show_histogram(histogram, img_path):

    # Flatten the array to 1D if needed
    data = histogram.flatten()

    # Generate indices for the bars
    indices = np.arange(len(data))

    top_5_indices = np.argsort(data)[-5:][::-1]
    top_5_values = data[top_5_indices]
    print(f"{img_path}\nTop 5 indices: {top_5_indices}\nTop 5 values: {top_5_values}\n\n")

    # Plot the bar chart
    plt.bar(indices, data, color='tab:blue', alpha=0.7)
    plt.title(f"Visual Words in {img_path}")
    plt.xlabel("Visual Word")
    plt.ylabel("Frequency")
    plt.show()