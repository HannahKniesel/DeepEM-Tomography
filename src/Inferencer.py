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
import tqdm
import tifffile as tiff


from deepEM.Inferencer import AbstractInference
from deepEM.Utils import print_error,print_info,print_warning,format_time
from src.Model import Model 
from src.Dataset import TiltSeries_Dataset, Reconstruction_Dataset, min_max_norm_np
from src.STEM import uniform_samples, density_based_samples, accumulate_beams


        
    
class Inference(AbstractInference):
    """
    Class for model inference. Implements all abstract methods
    to handle loading models, making predictions, and saving results.
    """
    def __init__(self, model_path: str, data_path: str, batch_size: int, resize: int) -> None:
        """
        Initializes the inference pipeline with model and data paths.

        Args:
            data_path (str): Path to the input data (single file or directory).
            batch_size (int): Number of samples to process in a single batch during inference.
        """
        super().__init__(model_path, data_path, batch_size)
        self.resize = resize
        if(self.resize != self.metadata["resize"]):
            print_warning("The resizing parameter used during inference differs from the one used during model development. As a result, the previously selected hyperparameters may no longer be optimal. To ensure the best performance, you can rerun the model development process to find the ideal hyperparameters for the desired resize setting.")
                
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
    
    def inference(self):
        dataset = TiltSeries_Dataset(self.data_path, resize = self.resize)
        dataloader = DataLoader(dataset, batch_size=self.metadata["batch_size"], shuffle=True)
        mse_loss = torch.nn.MSELoss()
        optim_small = torch.optim.Adam(self.model_small.parameters(), lr = self.metadata["lr"]) 
        optim = torch.optim.Adam(self.model.parameters(), lr = self.metadata["lr"])
        best_train_loss = np.inf

        print_info("Start Model Training...")
        for epoch in range(self.metadata["epochs"]):
            start_time = time.time()
            
            avg_loss = 0
            avg_loss_small = 0
            for batch_idx, batch in enumerate(dataloader):
                beam_origins, beam_directions, beam_ends, beam_detections = batch 
                # get predictions of small MLP 
                samples, distances = uniform_samples(beam_origins, beam_directions, beam_ends, self.metadata["beam_samples"])
                samples = samples.reshape(-1,3)
                densities = self.model_small(samples.cuda())
                predicted_detections = accumulate_beams(densities, samples, self.metadata["beam_samples"])
                loss_small = mse_loss(predicted_detections, beam_detections.cuda())
                loss_small = loss_small / self.metadata["accum_gradients"]         # normalize loss to account for batch accumulation
                avg_loss_small += loss_small.item() 
                loss_small.backward()
                
                # get predictions of large MLP 
                samples = density_based_samples(densities, distances, beam_origins, beam_directions, beam_ends, self.metadata["beam_samples"])
                samples = samples.reshape(-1,3)
                densities = self.model(samples.cuda())
                predicted_detections = accumulate_beams(densities, samples, self.metadata["beam_samples"]*2)
                loss = mse_loss(predicted_detections, beam_detections.cuda())
                loss = loss / self.metadata["accum_gradients"]         # normalize loss to account for batch accumulation
                avg_loss += loss.item() 
                loss.backward()
                

                if((batch_idx + 1) % self.metadata["accum_gradients"] == 0) or (batch_idx + 1 == len(dataloader)):
                    optim_small.step()
                    optim_small.zero_grad()
                    optim.step()
                    optim.zero_grad()
            
            if(avg_loss < best_train_loss):
                torch.save(self.model_small.state_dict(), os.path.join(self.save_to, "mlp_small.pth"))    
                torch.save(self.model.state_dict(), os.path.join(self.save_to, "mlp.pth")) 
                with open(self.save_to+"/config.json", "w") as outfile: 
                    json.dump(self.metadata, outfile)
                print_info(f"Saved currently best model states at {self.save_to} with training loss = {avg_loss/len(dataloader)}")  
                best_train_loss = avg_loss 
            end_time = time.time()
            elapsed_time = end_time - start_time
            accum_time += elapsed_time
            remaining_time = (self.metadata["epochs"] - (epoch+1))*(accum_time/(epoch+1))
            print_info(f"Avg time single epoch: {format_time(accum_time/(epoch+1))} | Remaining time training: {format_time(remaining_time)}")
        print_info("Finished model training. Tomogramm is being computed...\n")
        
        with torch.no_grad():
            self.model.load_state_dict(torch.load(os.path.join(self.save_to, "mlp.pth"), weights_only=True))
            self.model_small.load_state_dict(torch.load(os.path.join(self.save_to, "mlp_small.pth"), weights_only=True))
            
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
                # image_3d = np.random.randint(0, 256, (50, 100, 100), dtype=np.uint8)  
                tiff.imwrite(os.path.join(self.save_to,"tomogram.tif"), reconstruction)

        print_info(f"Tomogram was saved to {os.path.join(self.save_to,'tomogram.tif')}")
        
        return 
    
    def load_checkpoint(self):
        # no checkpoint loading required, as we need to retrain for inference.
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

    
    