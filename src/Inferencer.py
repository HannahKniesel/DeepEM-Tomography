import tifffile
from PIL import Image
from typing import Any, List
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import datetime
import os
from pathlib import Path
import json
import time
from tqdm import tqdm
import tifffile as tiff


from deepEM.Inferencer import AbstractInference
from deepEM.Utils import print_error,print_info,print_warning,format_time, find_model_file
from src.Model import Model 
from src.Dataset import TiltSeries_Dataset, Reconstruction_Dataset, min_max_norm_np
from src.STEM import uniform_samples, density_based_samples, accumulate_beams


        
    
class Inference(AbstractInference):
    """
    Class for model inference. Implements all abstract methods
    to handle loading models, making predictions, and saving results.
    """
    def __init__(self, model_path: str, data_path: str, batch_size: int) -> None:
        """
        Initializes the inference pipeline with model and data paths.

        Args:
            data_path (str): Path to the input data (single file or directory).
            batch_size (int): Number of samples to process in a single batch during inference.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = find_model_file(model_path)
        self.batch_size = batch_size

        self.metadata = self.load_metadata()
        self.data_path = self.metadata["data_path"]
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
        print_info(f"Will save results to {self.save_to}.")

        # Log model and data paths
        with open(os.path.join(self.save_to, "model-and-data.txt"), "w") as file:
            file.write(f"Model path: {os.path.abspath(self.model_path)}\n")
            file.write(f"Data path: {os.path.abspath(self.data_path)}\n")
        self.resize = self.metadata["resize"]

              
    def min_max_norm(self, value):
        return (value-value.min())/(value.max()-value.min())
        
    def setup_model(self) -> None:
        """
        sets up the model class for inference.

        Returns: 
            torch.nn.Module: the model
        """
        self.model_small =  Model(n_posenc = self.metadata["pos_enc"]//2, n_features = 128, n_layers = 3, skip_layer = 0).cuda()
        mlp =  Model(n_posenc = self.metadata["pos_enc"], n_features = 256, n_layers = 6, skip_layer = 0).cuda()
        return mlp
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)
        self.model_small.load_state_dict(checkpoint["small_model_state_dict"])
        self.model_small.eval()
        self.model_small.to(self.device)
        self.metadata = checkpoint["metadata"]
        return
    
    def load_single_image(self, path):
        return
    
    def inference(self):        
        with torch.no_grad():            
            # load metadata
            with open(os.path.join(self.data_path,"metadata.json"), 'r') as file:
                metadata = json.load(file)
            pixelsize = metadata["pixelsize_nmperpixel"]
            slice_thickness_nm = metadata["slice_thickness_nm"]
            original_px_resolution = metadata["original_px_resolution"]
            
            
            test_dataset = Reconstruction_Dataset(self.resize, slice_thickness_nm, pixelsize, original_px_resolution)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last = False)
            with torch.no_grad(): 
                reconstruction = []
                for samples in tqdm(test_dataloader, desc="Generate Tomogram"): 
                    densities = self.model(samples.cuda())
                    reconstruction.extend(densities.cpu().numpy().astype(np.float16))
                reconstruction = min_max_norm_np(np.array(reconstruction).reshape(test_dataset.x_dim, test_dataset.y_dim, test_dataset.z_dim))
                reconstruction = (reconstruction*255).astype(np.uint8).transpose(2,0,1) # (depth, height, width)
                tiff.imwrite(os.path.join(self.save_to,"tomogram.tif"), reconstruction)

        print_info(f"Tomogram was saved to {os.path.join(self.save_to,'tomogram.tif')}")
        
        return 
    

     
    
    def predict_single(self) -> Any:
        """
        Perform inference on a single image.

        Returns:
            Any: The prediction result for the image.
        """        
        return 
        
    
    def predict_batch(self) -> List[Any]:
        """
        Perform inference on a batch of images.
        """
        return 
        
    def save_prediction(self, prediction, save_file: str) -> None:
        """
        Save predictions to a file.

        Args:
            input (Any): single input to save.
            prediction (Any): Prediction of the input to save. (Return of the self.predict_single method)
            save_file (str): Filename and Path to save the predictions. You need to set the format.
        """
        return

    
    