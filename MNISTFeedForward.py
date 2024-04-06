import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')


class MNISTDataset:
    def __init__(self, root: str = "./data", train: bool = True) -> None:
        self.dataset = datasets.MNIST(
            root=root, train=train, transform=transforms.ToTensor(), download=True)

    def get_loader(self, batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.5) -> None:
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class Trainer:
    def __init__(self, model: NeuralNetwork, criterion: nn.Module, optimizer: torch.optim.Optimizer, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, cuda: bool = False) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cuda = cuda

        if self.cuda:
            self.model = self.model.cuda()

    def train(self, epochs: int, early_stopping: bool = False, patience: int = 5) -> None:
        best_val_loss = float('inf')
        counter = 0
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=1)
        for epoch in range(epochs):
            correct_train = 0
            running_loss = 0
            self.model.train()
            for i, (images, labels) in enumerate(self.train_loader):
                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                images = images.view(-1, 28 * 28)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                _, predicted = torch.max(outputs.data, 1)
                correct_train += (predicted == labels).sum().item()
                running_loss += loss.item()

            train_accuracy = 100 * correct_train / \
                len(self.train_loader.dataset)
            print('epoch [{}/{}], training loss: {:.12f}, training accuracy: {:.5f}%'.format(
                epoch+1, epochs, running_loss/len(self.train_loader), train_accuracy))

            if early_stopping:
                val_loss = self.evaluate()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("early stopping...")
                        break

    def evaluate(self) -> float:
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                images = images.view(-1, 28 * 28)
                outputs = self.model(images)
                val_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        val_loss /= len(self.test_loader)
        return val_loss


class Tester:
    def __init__(self, model: NeuralNetwork, test_loader: torch.utils.data.DataLoader, cuda: bool = False) -> None:
        self.model = model
        self.test_loader = test_loader
        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()

    def test(self) -> None:
        correct = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                images = images.view(-1, 28 * 28)
                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / len(self.test_loader.dataset)
        print('Test accuracy: {:.3f}%'.format(test_accuracy))


class ClassificationReporter:
    def __init__(self, model: NeuralNetwork, test_loader: torch.utils.data.DataLoader, cuda: bool = False):
        self.model = model
        self.test_loader = test_loader
        self.cuda = cuda

        if self.cuda:
            self.model = self.model.cuda()

    def generate_report(self) -> str:
        self.model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                images = images.view(-1, 28 * 28)
                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        report = classification_report(all_labels, all_predictions)
        return report


class ImageDisplay:
    def __init__(self, model):
        self.model = model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])

    def display_image(self, directory_path, image_names):
        for image_name in image_names:
            image_path = os.path.join(directory_path, image_name)

            try:
                if not os.path.exists(image_path):
                    print(f"'{image_path}' does not exist.")
                    return

                image = Image.open(image_path).convert("L")

                if image is None:
                    print(f"unable to open or read the file {image_path}")

                if len(image.size) == 0:
                    raise ValueError("Image has no size")

                aspect_ratio = 28 / max(image.size)

                resized_image = image.resize(
                    (int(image.width * aspect_ratio), int(image.height * aspect_ratio)))

                pad_width = (28 - resized_image.width) // 2
                pad_height = (28 - resized_image.height) // 2
                padded_image = Image.new('L', (28, 28), color=0)
                padded_image.paste(resized_image, (pad_width, pad_height))

                tensor_image = self.transform(
                    padded_image).view(1, -1).to(self.device)

                with torch.no_grad():

                    if self.model.cuda:
                        tensor_image = tensor_image.cuda()

                    outputs = self.model(tensor_image)

                    _, predicted = torch.max(outputs.data, 1)
                    prediction = predicted.item()

                if int(image_name[0]) == prediction:
                    predicted_text = f"predicted digit: {prediction} (Correct)"
                else:
                    predicted_text = f"predicted digit: {prediction} (Incorrect)"
                print(predicted_text)

            except ValueError as e:
                print(f"error processing image: {e}")

            except Exception as e:
                print(f"unexpected error occured: {e}")


class ModelManager:
    def __init__(self):
        pass

    def save_model(self, model, model_path):
        torch.save(model.state_dict(), model_path)
        print(f"model is successfully saved to {model_path}")

    def load_model(self, model, model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"model loaded successfully from {model_path}")

        except FileNotFoundError:
            print(f"model file is not found at {model_path}")

        except Exception as e:
            print(f"error loading the model: {e}")


class CAMVisualizer:
    def __init__(self, model, target_layer, output_size):
        self.model = model
        self.target_layer = target_layer
        print(f"the target layer is : {self.target_layer}")
        self.model.eval()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.output_size = output_size
        self.feature_maps = None
        self.hook_handle = self.target_layer.register_forward_hook(
            self._hook_fn)

    def _hook_fn(self, module, input, output):
        try:

            if isinstance(output, torch.Tensor):
                self.feature_maps = output

            elif isinstance(output, tuple):
                self.feature_maps = output[0]

            else:
                logging.error(
                    f"Unexpected output type in _hook_fn: {type(output)}")

        except Exception as e:
            logging.error(f"Error occurred in _hook_fn method: {e}")

        finally:
            logging.debug("Exiting _hook_fn method.")

    def _get_cam(self, feature_maps, weights, class_idx, size):
        try:
            if not isinstance(size, tuple) or len(size) < 2:
                size = (28, 28)

            if weights[class_idx].dim() == 0:
                weight = weights[class_idx]
                cam = F.relu(weight * feature_maps[class_idx])
            else:
                cam = F.relu(torch.matmul(weights[class_idx], feature_maps.T))

            assert cam.dim() == 1, "CAM does not have the correct shape"
            print(f"Cam shape: {cam.shape}")
            return cam
        except Exception as e:
            print(f"Error occurred on _get_cam method: {e}")
            return None

    def visualize(self, image_path, image_tensor, class_idx=None):
        try:
            image = Image.open(image_path).convert("L")

            aspect_ratio = 28 / max(image.size)

            resized_image = image.resize(
                (int(image.width * aspect_ratio), int(image.height * aspect_ratio)))

            pad_width = (28 - resized_image.width) // 2
            pad_height = (28 - resized_image.height) // 2

            padded_image = Image.new('L', (28, 28), color=0)
            padded_image.paste(resized_image, (pad_width, pad_height))

            flatten_size = image_tensor.size(2) * image_tensor.size(3)

            image_tensor = image_tensor.to(self.device)
            image_tensor = image_tensor.view(-1, flatten_size)

            self.model.zero_grad()
            logits = self.model(image_tensor)
            if class_idx is None:
                class_idx = logits.argmax(dim=1)

            logits[:, class_idx].backward()

            gradients = self.target_layer.weight.grad

            if self.feature_maps is None or self.feature_maps.shape[1] != gradients.shape[0]:
                return None

            weights = torch.mean(gradients, dim=1)

            cam = self._get_cam(self.feature_maps, weights,
                                class_idx, image_tensor)

            if cam is None:
                return None

            cam = cam.detach().squeeze().cpu().numpy()

            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + (1e-8))

            return cam

        except Exception as e:
            print(f"error occurred in visualize method: {e}")
            return None

    def plot_cam_on_image(self, image_path, cam):
        try:
            image = Image.open(image_path).convert("L")

            aspect_ratio = 28 / max(image.size)
            resized_image = image.resize(
                (int(image.width * aspect_ratio), int(image.height * aspect_ratio)))

            pad_width = (28 - resized_image.width) // 2
            pad_height = (28 - resized_image.height) // 2
            padded_image = Image.new('L', (28, 28), color=0)
            padded_image.paste(resized_image, (pad_width, pad_height))

            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + (1e-8))
            cam = (cam * 255).astype(np.uint8)

            heatmap = plt.get_cmap("jet")(cam)

            plt.imshow(padded_image, cmap="gray")
            plt.imshow(heatmap, alpha=0.5, extent=(0, 28, 28, 0))
            plt.axis("off")
            plt.show()

            # heatmap = Image.fromarray(cam, mode='L').resize(
            #     padded_image.size, Image.Resampling.BILINEAR)

            # heatmap = np.array(heatmap)

            # heatmap = np.uint8(255 * heatmap)
            # heatmap = Image.fromarray(heatmap, 'L')

            # heatmap = heatmap.convert('RGBA')
            # heatmap.putalpha(128)

            # final_image = Image.new('RGBA', heatmap.size)
            # final_image.paste(padded_image, (0, 0), padded_image)
            # final_image.paste(heatmap, (0, 0), heatmap)

        except Exception as e:
            print(f"an error occurred in plot cam on image method: {e}")

    def remove_hook(self):
        self.hook_handle.remove()


class MainExecutor:
    def __init__(self) -> None:
        self.input_size = 784
        self.hidden_size = 400
        self.out_size = 10
        self.epochs = 10
        self.batch_size = 100
        self.learning_rate = 0.001

    def run(self) -> None:
        train_dataset = MNISTDataset(root="./data", train=True)
        test_dataset = MNISTDataset(root="./data", train=False)
        train_loader = train_dataset.get_loader(
            batch_size=self.batch_size, shuffle=True)
        test_loader = test_dataset.get_loader(
            batch_size=self.batch_size, shuffle=False)

        model = NeuralNetwork(self.input_size, self.hidden_size, self.out_size)

        report = ClassificationReporter(
            model, test_loader, cuda=torch.cuda.is_available())

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        trainer = Trainer(model, criterion, optimizer,
                          train_loader, test_loader, cuda=torch.cuda.is_available())

        trainer.train(self.epochs)

        tester = Tester(model, test_loader, cuda=torch.cuda.is_available())
        tester.test()

        classification_report = report.generate_report()
        print(classification_report)

        model.eval()

        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                output_size = module.out_features
                cam_visualizer = CAMVisualizer(
                    model, module, output_size=output_size)
                print(f"Visualizing CAM for layer {name}...")

                # I have 5mnist.png in here so I wanted it to be predicted.

                image_name = "5mnist.png"
                image_directory = "[image_path]"

                image_path = os.path.join(image_directory, image_name)
                image = Image.open(image_path).convert("L")

                resize_transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                ])
                image_tensor = resize_transform(image).unsqueeze(0)

                cam_visualizer.model.eval()
                out_size = module.out_features
                class_idx_range = 1

                if cam_visualizer.feature_maps is not None:

                    class_idx_range = min(
                        out_size, cam_visualizer.feature_maps.shape[1])

                for class_idx in range(class_idx_range):
                    class_idx = min(class_idx, out_size - 1)

                    cam = cam_visualizer.visualize(
                        image_path, image_tensor, class_idx=class_idx)
                    if cam is not None:
                        cam_visualizer.plot_cam_on_image(
                            image_path, cam)

                cam_visualizer.remove_hook()


if __name__ == "__main__":
    executor = MainExecutor()
    executor.run()

    model = NeuralNetwork(executor.input_size,
                          executor.hidden_size, executor.out_size)

    model_manager = ModelManager()
    model_path = "[model_path]"

    model_manager.save_model(model, model_path)

    loaded_model = NeuralNetwork(executor.input_size,
                                 executor.hidden_size, executor.out_size)
    model_manager.load_model(loaded_model, model_path)

    model = model.to(torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
