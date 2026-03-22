import os

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

from dataset import Run3DatasetGenerator, Run3TestDataset
from scenes import Scene
from util import submit_results


class Run3:

    train_loader = None
    test_loader = None
    model = None

    def __init__(self,
                 directory: str,
                 batch_size: int = 64,
                 learning_rate: float = 0.00001,
                 epochs: int = 100):

        self.directory = directory
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using {self.device}.")

        # The transform to use for all training images.
        # Resize, flatten, normalise AND randomly flip (to artificially expand dataset).
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # The transform to use for all training images.
        # Resize, flatten, normalise BUT DO NOT randomly flip.
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._create_datasets(directory)
        self._create_model()

    def _create_datasets(self, directory):
        """
        Create the training and validation datasets.
        :param directory: The training image directory
        """
        gen = Run3DatasetGenerator(directory, validation_size=0.2)
        train_dataset = gen.train(self.transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_dataset = gen.validate(self.transform)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def _create_model(self):
        """
        Create the ResNet50 model and optimizers.
        """

        # Use Resnet50 as backbone model
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Modify the output layer to predict one of the scenes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(Scene))

        # Send model to the respective device (CPU or GPU)
        self.model = self.model.to(self.device)

        # Create the optimizers to train/learn based on
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self):
        """
        Learn the training images and adapt based on the validation set performance.
        """

        # Current best accuracy
        accuracy = 0

        # Record of accuracies
        accuracy_record = []

        # Loop through all the epochs
        for epoch in range(self.epochs):
            self.model.train() # Learn on the dataset
            total_loss = 0 # Current loss is 0

            # Dynamically updating progress bar
            for images, labels in tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Optimize the model
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward() # Back-propagation
                self.optimizer.step()

                total_loss += loss.item()

            current_accuracy = float(self.evaluate(self.test_loader, self.model))
            accuracy_record.append(current_accuracy)

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(self.train_loader)}")

            # If the current model is better than any model we've seen before
            if current_accuracy > accuracy:
                accuracy = current_accuracy
                if not os.path.exists("models"):
                    os.makedirs("models")
                torch.save(self.model.state_dict(), "models/run3_model_resnet50.pth") # Save the current model
                print(f"Model saved with accuracy: {current_accuracy}!")

    def predict(self, directory, model_path = "models/run3_model_resnet50.pth", batch_size:int = 32):
        """
        Predict the classes of all test images.
        :param directory: Direction to the test images
        :param model_path: Path to a pre-trained model (optional)
        :param batch_size: Size of batch to process test images (optional, default 32)
        """

        try:
            model = models.resnet50()
            model.fc = torch.nn.Linear(model.fc.in_features, len(Scene))
            model = model.to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except:
            if self.model is not None:
                model = self.model
            else:
                raise TypeError("Model has not been trained and no pre-trained path was provided.")
        finally:
            pass

        # Create the dataset of test images
        predict_dataset = Run3TestDataset(image_dir=directory, transform=self.test_transform)
        # Important: we don't shuffle the test dataset; order is crucial
        predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

        # Set the model to evaluation mode
        model.eval()
        # Make predictions and write them to the file
        predictions = []
        for images, img_names in tqdm(predict_loader, total=len(predict_loader), desc="Predicting"):
            images = images.to(self.device)
            outputs = model(images)
            prediction_indices = torch.argmax(outputs, dim=1).tolist()
            predictions.extend([Scene.from_index(s+1).out for s in prediction_indices])

        submit_results("run3.txt", predictions, directory)

    def evaluate(self, validate_loader, model, generate_confusion_matrix=False):
        """
        Evaluate a model's performance.
        :param validate_loader: The loader of the validation images
        :param model: The model to evaluate
        :param generate_confusion_matrix: Generate a confusion matrix image
        """
        model.eval()
        total, correct = 0, 0
        confusion_matrix = torch.zeros(len(Scene), len(Scene))
        with torch.no_grad():
            for images, labels in validate_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                labels = torch.argmax(labels, dim=1)

                outputs = model(images)
                prediction_indices = torch.argmax(outputs, dim=1)

                if generate_confusion_matrix:
                    for (predict, label) in zip(prediction_indices, labels):
                        confusion_matrix[predict, label] += 1

                total += labels.size(0)
                correct += (prediction_indices == labels).sum().item()

        accuracy = correct / total

        # Create the confusion matrix figure
        if generate_confusion_matrix:
            plt.figure(figsize=(12, 12))
            sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues", xticklabels=[s.title for s in Scene], yticklabels=[s.title for s in Scene],
                        cbar=False)
            plt.xlabel("Actual", fontsize=14)
            plt.xticks(fontsize=14)
            plt.ylabel("Predicted", fontsize=14)
            plt.yticks(fontsize=14)
            plt.savefig("confusion_matrix.png", bbox_inches='tight', pad_inches=0.1)

        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

def main():
    run3 = Run3("training")
    run3.fit()
    run3.predict("testing")

if __name__ == "__main__":
    main()
