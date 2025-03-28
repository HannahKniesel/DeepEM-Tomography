from abc import ABC, abstractmethod
from typing import Any, List
import os
from pathlib import Path
import datetime
import torch

from deepEM.Utils import print_error, find_model_file


class AbstractInference(ABC):
    """
    Abstract base class for performing model inference. 

    Subclasses must implement all abstract methods to define how models are loaded, 
    how predictions are made, and how results are saved.
    """

    def __init__(self, model_path: str, data_path: str, batch_size: int) -> None:
        """
        Initializes the inference pipeline with model and data paths.

        Args:
            model_path (str): Path to the model checkpoint file or directory containing it.
            data_path (str): Path to the input data (single file or directory).
            batch_size (int): Number of samples to process in a single batch during inference.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = find_model_file(model_path)
        self.data_path = data_path
        self.batch_size = batch_size

        if self.model_path:
            self.metadata = self.load_metadata()
            self.model = self.setup_model()
            self.load_checkpoint()

            # Create a directory for storing inference results
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_root = Path(self.model_path).parent.parent.parent.stem
            results_dir = (
                os.path.join(self.data_path, f"results-{model_root}", timestamp)
                if os.path.isdir(self.data_path)
                else os.path.join(Path(self.data_path).parent, f"results-{model_root}", timestamp)
            )
            self.save_to = results_dir
            os.makedirs(self.save_to, exist_ok=True)

            # Log model and data paths
            with open(os.path.join(self.save_to, "model-and-data.txt"), "w") as file:
                file.write(f"Model path: {os.path.abspath(self.model_path)}\n")
                file.write(f"Data path: {os.path.abspath(self.data_path)}\n")


    def load_metadata(self) -> dict:
        """
        Loads metadata from the model checkpoint.

        Returns:
            dict: Metadata extracted from the checkpoint.
        """
        checkpoint = torch.load(self.model_path)
        return checkpoint["metadata"]

    def load_checkpoint(self) -> None:
        """
        Loads model weights and sets the model to evaluation mode.
        """
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)
        self.metadata = checkpoint["metadata"]

    @abstractmethod
    def setup_model(self) -> torch.nn.Module:
        """
        Defines and initializes the model architecture for inference.

        Returns:
            torch.nn.Module: The model ready for inference.
        """
        raise NotImplementedError("The 'setup_model' method must be implemented by the DL specialist.")

    def get_image_files(self, folder_path: str, ext:List[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif")) -> List[str]:
        """
        Retrieves all image files from a directory.

        Args:
            folder_path (str): Path to the directory containing images.

        Returns:
            List[str]: List of file paths to images in the directory.
        """
        return [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.lower().endswith(ext)
        ]

    def inference(self) -> None:
        """
        Runs inference on the input data.

        Depending on whether `data_path` is a file or directory, 
        calls `predict_single()` for a single image or `predict_batch()` for multiple images.
        """
        with torch.no_grad():
            if self.model_path:
                if os.path.isdir(self.data_path):
                    self.predict_batch()
                elif os.path.isfile(self.data_path):
                    self.predict_single()
                else:
                    print_error(f"Invalid data path: {self.data_path} is neither a file nor a directory.")

    @abstractmethod
    def predict_single(self) -> Any:
        """
        Performs inference on a single image.

        Implementations should call `save_prediction(prediction, save_file)` 
        to store the prediction result.

        Returns:
            Any: The model's prediction for the given input.
        """
        raise NotImplementedError("The 'predict_single' method must be implemented by the DL specialist.")

    @abstractmethod
    def predict_batch(self) -> List[Any]:
        """
        Performs inference on a batch of images.

        Implementations should call `save_prediction(prediction, save_file)` 
        for each predicted output.

        Returns:
            List[Any]: List of predictions for the batch.
        """
        raise NotImplementedError("The 'predict_batch' method must be implemented by the DL specialist.")

    @abstractmethod
    def save_prediction(self, prediction: Any, save_file: str) -> None:
        """
        Saves a model prediction to a file.

        Args:
            prediction (Any): The prediction result to be saved.
            save_file (str): Path to the file where the prediction should be stored.
        """
        raise NotImplementedError("The 'save_prediction' method must be implemented by the DL specialist.")