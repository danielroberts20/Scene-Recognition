import math
import os
import random
import time

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from scenes import Scene
from util import get_training_images, normalize, show_visual_words, get_testing_images, cosine_similarity, \
    submit_results


class Run2:

    num_visual_words = 0
    patch_size = ()
    stride = 0
    folds = 0
    training_patches = None
    model = None
    classifiers = None
    verbose = False
    show_vw = False

    def __init__(self, num_visual_words: int = 500, patch_size: tuple = (8, 8), stride: int = 4, seed: int = None, folds:int = 5, verbose: bool = False):
        self.num_visual_words = num_visual_words
        self.patch_size = patch_size
        self.stride = stride
        self.verbose = verbose
        self.folds = folds

        if seed is None:
            self.seed = random.randint(1, 100)
        else:
            self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def set_verbose(self, verbose: bool):
        self.verbose = verbose
    
    def show_visual_words(self):
        self.show_vw = True

    def _log(self, message: str):
        """
        Used for logging.
        :param message: The message to show.
        :type message: str
        """
        if self.verbose:
            print(message)

    def _extract_features(self, img: str) -> np.ndarray:
        """
        Extract the visual words from an image.

        :param img: Path to the image
        :type img: str

        :return: Histogram of visual words.
        """
        if not check_is_fitted(self.model,
                               msg="The KMeans classifier for vector quantisation must be fitted before extracting features."):
            pass

        # Extract the patches in the image
        patches = self._extract_patches(np.asarray(Image.open(img)))

        # Extract the visual words from the patches
        return list_to_histogram(self.model.predict(patches), len(set(self.model.labels_)))

    def _extract_patches(self, image: np.ndarray, normalise: bool = True) -> np.ndarray:
        """
        Extract patches from an image.
        :param image: The numpy array of the image of shape (*i_height*, *i_width*)
        :type image: np.ndarray

        :param normalise: Normalise each patch. Default is True.
        :type normalise: bool

        :return: Numpy array of patches of shape (*num_patches*, *p_height* * *p_width*).
        """

        # Get dimensions
        patch_height, patch_width = self.patch_size
        image_height, image_width = image.shape

        patches = []

        for y in range(0, image_height - patch_height + 1, self.stride):
            for x in range(0, image_width - patch_width + 1, self.stride):
                # Extract the patch
                patch = image[y:y + patch_height, x:x + patch_width]
                # Flatten and normalise
                if normalise:
                    flat_patch = normalize(patch.astype(np.float32))
                else:
                    flat_patch = patch.flatten()
                # Store
                patches.append(flat_patch)

        # Return array of patches
        output = np.array(patches)
        return output

    def _create_bovw_model(self, training_path: str) -> KMeans:
        """
        Create and train the KMeans model for vector quantisation of patches.
        A sample size of (words/100)% is used to learn the visual words.

        :param training_path: Path to directory containing all scene subfolders.
        :type training_path: str

        :return: A trained KMeans clustering model.
        """

        # Train the KMeans clustering to learn the bag-of-visual-words
        self._log("Training bag-of-visual-words features...")
        total_num_training = len(get_training_images(training_path))
        patches = []

        # Get some random images of each type of scene
        for scene in Scene:
            scene_images = get_training_images(training_path, scene)
            num = math.floor(total_num_training * (self.num_visual_words / 10000) / len(Scene))

            selected = random.sample(scene_images, num)
            patches.extend([self._extract_patches(np.asarray(Image.open(img))) for img in selected])

        self.training_patches = np.vstack(patches)

        # Train the model
        kmeans = KMeans(n_clusters=self.num_visual_words, random_state=self.seed)
        kmeans.fit(self.training_patches)

        return kmeans

    def _create_dataset(self, training_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Create the dataset for training.

        :param training_path: The path to the training images
        :type training_path: str

        :return: The dataset, the labels
        """

        path = os.path.join("data", "run2.npz")

        # Attempt to load the pre-computed dataset
        try:
            with np.load(path) as f:
                self._log(f"Loading dataset from {path}...")
                return f["X"], f["y"]
        except FileNotFoundError:
            all_training_images = get_training_images(training_path)

            # Create the output datasets
            X = np.ndarray((len(all_training_images), len(set(self.model.labels_))))
            y = np.ndarray((len(all_training_images),))

            # Loop through all training images
            self._log("Extracting visual words from training images...")
            for idx, img in enumerate(all_training_images):
                # Get the scene of the current image
                scene = Scene.from_path(img)

                # Extract visual words
                visual_words = self._extract_features(img)

                # Assign the visual words to the dataset, and the corresponding label
                X[idx] = visual_words
                y[idx] = scene.index

            self._log(f"Saving dataset to {path}...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, X=X, y=y)
            return X, y

    def predict(self, testing_path: str, output: str = "run2.txt"):
        """
        Predict the scenes of all testing images and output to a file.

        :param testing_path: The path to the testing images.
        :type testing_path: str

        :param output: The path to the desired output file.
        :type output: str
        """

        # Get all the testing images
        images = get_testing_images(testing_path)
        path = os.path.join("data", "test_vector.npy")

        # Load the test vector if already created
        try:
            self._log(f"Loading test vector from {path}...")
            test_vector = np.load(path)
        except FileNotFoundError:

            # Create test vector
            test_vector = np.ndarray((len(images), len(set(self.model.labels_))))
            progress_indicator = int(len(images) * 0.05)
            for idx, img in enumerate(images):
                if idx % progress_indicator == 0:
                    self._log(f"Populating test vector ({idx / len(images) * 100:.1f}%)")
                test_vector[idx] = self._extract_features(img)

            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, test_vector)

        # Predict the scene of each image
        self._log(f"Predicting confidence for classes...")
        confusion_matrix = np.ndarray((len(images), len(Scene)))
        for idx, classifier in enumerate(self.classifiers):
            probs = classifier.predict_proba(test_vector)[:,
                    1]  # Predict the probabilities of each image belonging to this class
            confusion_matrix[:, idx] = probs
        max_values = np.max(confusion_matrix, axis=1)  # Find the highest probability (confidence)
        print(f"  ⊢ Average prediction confidence: {np.sum(max_values) / len(images) * 100:.2f}%")

        # Find the column (class - 1) where the largest confidence is located
        predictions = (np.argmax(confusion_matrix, axis=1) + 1).astype(int)
        # Convert to scenes
        predictions = [Scene.from_index(int(p)).out for p in predictions]

        submit_results(output, predictions, testing_path)

    def fit(self, training_path: str):
        """
        Create training dataset, train bag-of-visual-words quantisation model and linear classifiers.

        :param training_path: The path to the training images.
        :type training_path: str
        """

        # Train the bag-of-visual-words vocab
        self.model = self._create_bovw_model(training_path)

        self._log("Creating dataset...")
        X, y = self._create_dataset(training_path)

        self._log("Creating KFold cross-validation...")
        kfold = KFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        accuracies = []

        for i, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            self._log(f"Fold {i+1}/{self.folds}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self._log(f"Training linear classifiers...")
            self.classifiers = []
            for c in range(1, 16):
                classifier = LogisticRegression(solver="liblinear", max_iter=self.num_visual_words,
                                                random_state=self.seed)
                transformed_y = (y_train == c).astype(int)  # Turn the labels into a one-vs-all binary dataset
                classifier.fit(X_train, transformed_y)
                self.classifiers.append(classifier)

            # Draw the visual words to see how similar the clustering is.
            if self.show_vw: show_visual_words(self.model, self.training_patches, num_words=6)

            # Evaluate classifiers on the testing dataset
            probability_matrix = np.ndarray((len(X_test), len(Scene)))
            for idx, classifier in enumerate(self.classifiers):
                probs = classifier.predict_proba(X_test)[:, 1]
                probability_matrix[:, idx] = probs
            predictions = (np.argmax(probability_matrix, axis=1) + 1).astype(int)
            accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
            accuracies.append(accuracy)

        print(f"Run 2 Average Accuracy ({self.folds} Folds): {(sum(accuracies)/len(accuracies)) * 100:.2f}%")

        similarity, inertia = eval_visual_words(self.model, self.training_patches, self.patch_size)
        print(f"  ⊢ Average visual word similarity: {similarity * 100:.2f}%")
        print(f"  ⊢ Inertia of visual word KMeans model: {inertia:.2f}")

def list_to_histogram(lst: list[int], range_size: int = None) -> np.ndarray:
    """
    Convert a list of ints into a numpy array of frequencies (a histogram),
    where the number at index *i* represents the number of times *i* appears in the
    original list.
    :param lst: The list of ints.
    :param range_size: The desired range of frequencies.
                       Default is max(lst).
    :return: A numpy array of frequencies.
    """

    if range_size is None:
        range_size = max(lst)

    # Create a frequency array (initialize to zeros)
    freq_array = np.zeros(range_size, dtype=int)

    # Use bincount to count occurrences of each number in the list
    np.add.at(freq_array, lst, 1)
    return np.array(freq_array)

def eval_visual_words(model, patches, patch_size):
    """
    Evaluate the performance of the visual word clustering.

    :param model: The KMeans visual word clustering model.
    :param patches: The patches used for training the model.
    :param patch_size: The size of the patches in the shape (width, height).
    :return: The average cosine similarity of all patches to their respective representative vector,
             the inertia of the model.
    """
    centres = model.cluster_centers_
    labels = model.labels_
    unique_labels = list(set(labels))
    scores = []
    for label in unique_labels:
        # Get the indices of all the patches that belong to the current visual word
        vw_idx = np.where(labels == label)[0]
        sim = sum([cosine_similarity(centres[label].reshape(patch_size), patches[i].reshape(patch_size)) for i in
                   vw_idx]) / len(vw_idx)
        scores.append(sim)
    score = sum(scores) / len(scores)
    return score, model.inertia_

def main():
    t = time.time()
    run2 = Run2(num_visual_words=750, patch_size=(8, 8), stride=4, seed=5)
    run2.set_verbose(True)
    run2.fit("training")
    run2.predict("testing")
    print(f"  ⊢ Total running time: {time.time() - t:.1f} seconds")


if __name__ == '__main__':
    main()
