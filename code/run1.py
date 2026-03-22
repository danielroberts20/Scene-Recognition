import os
import time
from glob import glob

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from scenes import Scene
from util import normalize, get_training_images, submit_results

# Size of the "tiny image" fixed resolution
global dim
global model

def crop_center_square(image_path: str) -> Image:
    """
    Crop image to square about the centre

    Parameters:
        image_path (str): Path to image

    Returns:
        img (Image): Cropped image
    """
    # Open the image
    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size

    # Calculate the size of the square
    new_size = min(width, height)

    # Calculate the cropping coordinates
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2

    # Crop the image
    img_cropped = img.crop((left, top, right, bottom))

    return img_cropped


def make_tiny_image(image_path: str) -> np.ndarray:
    """
    Extract "tiny image" feature from image

    Parameters:
        image_path (str): Path to image

    Returns:
        img (np.ndarray): Zero-mean, unit-length array of image
    """

    # Crop the image in a square about the center
    img = crop_center_square(image_path)
    img = img.resize((dim, dim))

    return normalize(np.asarray(img).astype(np.float32))


def create_dataset(training_path: str):
    """
    Create the matrix of training images and labels

    Parameters:
        training_path (str): Path to folder containing training images

    Returns:
        X (np.ndarray): Array of training images. Shape (num_of_training_images, tiny_image_dim ** 2)
        y (np.ndarray): Array of training labels. Shape (num_of_training_images)
    """

    all_training_images = get_training_images(training_path)

    # Create the feature matrix (X) and target vector (y)
    X = np.ndarray((len(all_training_images), dim ** 2))
    y = np.ndarray((len(all_training_images)))

    # Populate the matrix and target vector
    for idx, file in enumerate(all_training_images):
        # Find the Scene that corresponds to the current image by using it's filepath
        # e.g. training/livingroom/87.jpg will have scene Scene.LIVING_ROOM
        scene = Scene(file.split(os.sep)[-2])
        X[idx] = make_tiny_image(file)
        y[idx] = scene.index
    return X, y


def predict(testing_path: str, output: str = "run1.txt") -> None:
    """
    Predict the classes of the testing images

    Parameters:
    testing_path (str): Path to folder containing testing images
    output (str): Path to output file for class predictions.
                  If no output is provided, the name of this file is used.
    """

    # Get all the testing images
    images = sorted(glob(os.path.join(testing_path, "*.jpg")), key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))

    # Create a matrix for all the test images
    test_vector = np.ndarray((len(images), dim ** 2))

    # Populate the matrix with the "tiny_image" of each test image
    for idx, image_path in enumerate(images):
        test_vector[idx] = make_tiny_image(image_path)

    # Predict the scene of each test image
    global model
    predictions = model.predict(test_vector)
    # Convert the numerical scene to the corresponding Scene
    predictions = [Scene.from_index(int(p)).out for p in predictions]

    submit_results(output, predictions, testing_path)

def train(training_path: str, tiny_image_dim: int = 16, folds: int = 5, random: int = 42) -> None:
    """
    Train the kNN model on the dataset provided

    Parameters:
    training_path (str): Path to folder containing training images
    tiny_image_dim (int): Dimension of the tiny image feature (d x d) (optional, default 16)
    random (int): Random state to use for train/test splitting (optional, default 42)
    """
    global dim
    dim = tiny_image_dim
    X, y = create_dataset(training_path)
    kfold = KFold(n_splits=folds, shuffle=True, random_state=random)

    global model
    k_values = range(1, 21)
    accuracies = {key: 0 for key in k_values}

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        for k in k_values:
            test_model = KNeighborsClassifier(n_neighbors=k)
            test_model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, test_model.predict(X_test))
            if (sum(accuracies.values()) == 0) or (accuracy > max(accuracies.values())) :
                model = test_model
            accuracies[k] += accuracy
    accuracies = [a / folds for a in accuracies.values()]

    plt.plot(k_values, accuracies)
    plt.xticks(k_values)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.title('Model performance on various K')
    plt.savefig("run1_performance.jpg")
    plt.show()

    best_k = k_values[accuracies.index(max(accuracies))]
    print(f"Best Accuracy at {best_k}: {max(accuracies) * 100}%")
    print('Model is using k =', model.n_neighbors)

def main():
    t = time.time()
    train("training")
    predict("testing")
    print(f"  ⊢ Total running time: {time.time() - t:.1f} seconds")

if __name__ == "__main__":
    main()


