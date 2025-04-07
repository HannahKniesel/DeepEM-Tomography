
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import copy
import numpy as np
import tifffile as tiff
from tqdm import tqdm

from deepEM.ModelTrainer import AbstractModelTrainer
from src.Model import Model 
from src.Dataset import TiltSeries_Dataset, Reconstruction_Dataset, min_max_norm_np
from src.STEM import uniform_samples, density_based_samples, accumulate_beams
from deepEM.Utils import find_file

criterion = torch.nn.MSELoss()



def min_max_norm(value):
        return (value - value.min())/(value.max()-value.min())

class ModelTrainer(AbstractModelTrainer):
    def __init__(self, data_path, logger, resume_from_checkpoint = None):
        """
        Initializes the trainer class for training, validating, and testing models.

        Args:
            model (torch.nn.Module): The model to train.
            logger (Logger): Logger instance for logging events.
            config (dict): Contains all nessecary hyperparameters for training. Must at least contain: `epochs`, `early_stopping_patience`, `validation_interval`, `scheduler_step_by`.
            resume_from_checkpoint (str): Path to resume checkpoint
            train_subset (float, optional): Use subset of training data. This can be used for quick hyperparamtertuning. Defaults to `None`. 
            reduce_epochs (float, optional): Use subset of epochs. This can be used for quick hyperparamtertuning. Defaults to `None`. 
        """
        super().__init__(data_path, logger, resume_from_checkpoint )
        
    def setup_model(self):
        """
        Setup and return the model for training, validation, and testing.

        This method must be implemented by the DL expert.

        Returns:
            model (lib.Model.AbstractModel): The dataloader for the training dataset.
        """
        self.model_small =  Model(n_posenc = self.parameter["pos_enc"]//2, n_features = 128, n_layers = 3, skip_layer = 0).to(self.device)
        mlp =  Model(n_posenc = self.parameter["pos_enc"], n_features = 256, n_layers = 6, skip_layer = 0).to(self.device)

        return mlp

    
    def inference_metadata(self):
        """
        Returns possible metadata needed for inference (such as class names) as dictonary.
        This metadata will be saved along with model weights to the training checkpoints. 
        
        
        Returns:
            dict: dictonary with metadata
            
        """
        metadata_path = find_file(self.data_path, "metadata.json")
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        pixelsize = metadata["pixelsize_nmperpixel"]
        slice_thickness_nm = metadata["slice_thickness_nm"]
        original_px_resolution = metadata["original_px_resolution"]
    
        metadata = {"epochs": self.num_epochs, 
                    "lr": self.parameter["learning_rate"],
                    "batch_size": self.parameter["batch_size"], 
                    "pos_enc": self.parameter["pos_enc"], 
                    "beam_samples": self.parameter["beam_samples"],
                    "accum_gradients": self.parameter["accum_gradients"], 
                    "resize": self.parameter["resize"], 
                    "data_path": self.data_path, 
                    "pixelsize_nmperpixel": pixelsize,
                    "slice_thickness_nm": slice_thickness_nm, 
                    "original_px_resolution": original_px_resolution}
        return metadata
        
            
    def setup_datasets(self):
        """
        Setup and return the dataloaders for training, validation, and testing.

        This method must be implemented by the DL expert.
        
        The data_path provided by the EM specialist can b accessed via self.data_path

        Returns:
            train_dataset (torch.utils.data.Dataset): The dataset for the training dataset.
            val_dataset (torch.utils.data.Dataset): The dataset for the validation dataset.
            test_dataset (torch.utils.data.Dataset): The dataset for the test dataset.
        """
        if(not os.path.isdir(os.path.join(self.data_path, "noisy-projections"))):
            self.logger.log_error(f"Data path {self.data_path} does not contain a folder 'noisy-projections'. Please provide a data path which contains a folder 'noisy-projections'.")
            return None, None, None
        
        train_dataset = TiltSeries_Dataset(os.path.join(self.data_path, "noisy-projections"), resize = self.parameter["resize"])
        
        if(os.path.isdir(os.path.join(self.data_path, "clean-projections"))):
            val_dataset = TiltSeries_Dataset(os.path.join(self.data_path, "clean-projections"), percentage=0.2, resize = self.parameter["resize"])
            test_dataset = TiltSeries_Dataset(os.path.join(self.data_path, "clean-projections"), percentage=1.0, resize = self.parameter["resize"])
        else: 
            val_dataset = TiltSeries_Dataset(os.path.join(self.data_path, "noisy-projections"), percentage=0.2, resize = self.parameter["resize"])
            test_dataset = TiltSeries_Dataset(os.path.join(self.data_path, "noisy-projections"), percentage=1.0, resize = self.parameter["resize"])


        return train_dataset, val_dataset, test_dataset
    
    def setup_visualization_dataloaders(self, val_dataset, test_dataset):
        """
        Setup and return the dataloaders for visualization during validation, and testing.
        This method will subsample the val_dataset and test_dataset to contain self.parameter["images_to_visualize"] datapoints
        This method should be overidden for imbalanced data, to pick the most interesting data samples.
                        
        Inputs:
            valset (torch.utils.data.Dataset): The validation dataset.
            testset (torch.utils.data.Dataset): The test dataset.

        Returns:
            val_vis_loader (torch.utils.data.DataLoader): The dataloader for visualizing a subset of the validation dataset.
            test_vis_loader (torch.utils.data.DataLoader): The dataloader for visualizing a subset of the test dataset.
        """
        # if(os.path.isdir(os.path.join(self.data_path, "clean-projections"))):
        #     vis_val_subset = TiltSeries_Dataset(os.path.join(self.data_path, "clean-projections"), percentage=1.0, resize = self.parameter["resize"])
        # elif(os.path.isdir(os.path.join(self.data_path, "noisy-projections"))): 
        #     vis_val_subset = TiltSeries_Dataset(os.path.join(self.data_path, "noisy-projections"), percentage=1.0, resize = self.parameter["resize"])
        # else:
        #     self.logger.log_error(f"Data path {self.data_path} does not contain a folder 'noisy-projections'. Please provide a data path which contains a folder 'noisy-projections' containing the tilt series.")
        #     return None, None
        val_vis_loader = self.test_loader # DataLoader(vis_val_subset, batch_size=self.parameter["batch_size"], shuffle=False)
        
        test_vis_loader = val_vis_loader
        return val_vis_loader, test_vis_loader
          
        

    def setup_optimizer(self):
        """
        Setup and return the optimizer and learning rate scheduler.

        This method must be implemented by the DL expert.

        Returns:
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.parameter["learning_rate"])
        self.optimizer_small = torch.optim.Adam(self.model_small.parameters(), lr = self.parameter["learning_rate"])
        
        lr_scheduler = None
        return optimizer, lr_scheduler

    
    def compute_loss(self, outputs, targets):
        """
        Compute the loss for a batch.
        
        Args:
            outputs (torch.Tensor): Model outputs.
            targets (torch.Tensor): Ground truth labels.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        loss = criterion(outputs, targets)
        loss = loss / self.parameter["accum_gradients"]  
        
        return criterion(outputs, targets)
        

    def train_step(self, batch_idx, batch):
        """
        Perform one training step.

        Args:
            batch (tuple): A batch of data (inputs, targets).
        
        Returns:
            torch.Tensor: The loss for this batch.
        """
        
        beam_origins, beam_directions, beam_ends, beam_detections = batch
        samples, distances = uniform_samples(beam_origins, beam_directions, beam_ends, self.parameter["beam_samples"])
        samples = samples.reshape(-1,3)
        densities = self.model_small(samples.cuda())
        predicted_detections = accumulate_beams(densities, samples, self.parameter["beam_samples"])
        loss_small = self.compute_loss(predicted_detections, beam_detections.cuda())
        loss_small.backward()


        # get predictions of large MLP 
        samples = density_based_samples(densities, distances, beam_origins, beam_directions, beam_ends, self.parameter["beam_samples"])
        samples = samples.reshape(-1,3)
        densities = self.model(samples.cuda())
        predicted_detections = accumulate_beams(densities, samples, self.parameter["beam_samples"]*2)
        loss = self.compute_loss(predicted_detections, beam_detections.cuda())
        loss.backward()
        
        if((batch_idx + 1) % self.parameter["accum_gradients"] == 0) or (batch_idx + 1 == len(self.train_loader)):
            self.optimizer_small.step()
            self.optimizer_small.zero_grad()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()
    
    def qualify(self, dataloader, file_name: str):
        """
        Saves visualizations of model predictions for a given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader containing the data for visualization.
            file_name (str): Prefix for the saved image files.
        """
        with torch.no_grad():
            # predicted_micrographs_small = []
            predicted_micrographs = []
            micrographs = []
            
            max_number_rays = self.parameter["images_to_visualize"]*self.parameter["resize"]*self.parameter["resize"]
            for beam_origins, beam_directions, beam_ends, beam_detections in dataloader:
                samples, distances = uniform_samples(beam_origins, beam_directions, beam_ends, self.parameter["beam_samples"])
                samples = samples.reshape(-1,3)
                densities = self.model_small(samples.cuda())   
                predicted_detections = accumulate_beams(densities, samples, self.parameter["beam_samples"])
                # predicted_micrographs_small.extend(predicted_detections.cpu().numpy())
                
                samples = density_based_samples(densities, distances, beam_origins, beam_directions, beam_ends, self.parameter["beam_samples"])
                samples = samples.reshape(-1,3)
                densities = self.model(samples.cuda())
                predicted_detections = accumulate_beams(densities, samples, self.parameter["beam_samples"]*2)
                predicted_micrographs.extend(predicted_detections.cpu().numpy())
                micrographs.extend(beam_detections.cpu().numpy())
                
                if(len(micrographs)>= max_number_rays):
                    break
            
            # predicted_micrographs_small = np.array(predicted_micrographs_small)
            predicted_micrographs = np.array(predicted_micrographs)
            micrographs = np.array(micrographs)

            # generate visualizations of the projections
            for val_idx in range(self.parameter["images_to_visualize"]): 
                start_idx = val_idx * self.parameter["resize"]**2
                end_idx = start_idx + (self.parameter["resize"]**2)
                # predicted_micrograph_small = predicted_micrographs_small[start_idx:end_idx].reshape(self.parameter["resize"], self.parameter["resize"])
                predicted_micrograph = predicted_micrographs[start_idx:end_idx].reshape(self.parameter["resize"], self.parameter["resize"])
                micrograph = micrographs[start_idx:end_idx].reshape(self.parameter["resize"], self.parameter["resize"])
                fig,axs = plt.subplots(1,2)
                plt.suptitle(f"Validation Image {val_idx}/{dataloader.dataset.num_images}\nProjection = {dataloader.dataset.val_projection_angles[val_idx]}°")
                # axs[0].imshow(predicted_micrograph_small, cmap="gray")
                # axs[0].set_title("Predicted Projection \nSmall MLP")
                axs[0].imshow(predicted_micrograph, cmap="gray")
                axs[0].set_title("Predicted Micrgraph")
                axs[1].imshow(micrograph, cmap="gray")
                axs[1].set_title("Ground Truth Micrograph")
                for a in axs: 
                    a.set_axis_off()
                plt.tight_layout()
                
                plt.savefig(os.path.join(self.logger.samples_dir, f"{file_name}_{val_idx}.png"))
                plt.close()
                
            
       
    
    
    def visualize(self, batch):
        """
        Visualizes the models input and output of a single batch and returns them as PIL.Image.

        Args:
            batch: A batch of data defined by your Dataset implementation.
        
        Returns:
            List[PIL.Image]: List of visualizations for the batch data.
            
        """
        return
    
    def compute_metrics(self, outputs, targets):
        """
        Computes a metric for evaluation based on the models outputs and targets

        Args:
            outputs[torch.Tensor]: A batch of model outputs
            targets[torch.Tensor]: A batch of targets
            
        
        Returns:
            dict: dictonary of the computed metrics
            
        """
        outputs = outputs.cpu()
        targets = targets.cpu()
        
        metrics = {"MSE": criterion(outputs, targets)}
        
        return metrics
        

    def val_step(self, batch):
        """
        Perform one validation step.

        Args:
            batch (tuple): A batch of data (inputs, targets).
        
        Returns:
            torch.Tensor: The loss for this batch.
            dict: Dictionary of metrics for this batch (e.g., accuracy, F1 score, etc.).
            
        """
                
        self.model.eval()
        beam_origins, beam_directions, beam_ends, beam_detections = batch
        samples, distances = uniform_samples(beam_origins, beam_directions, beam_ends, self.parameter["beam_samples"])
        samples = samples.reshape(-1,3)
        densities = self.model_small(samples.cuda())   
        predicted_detections = accumulate_beams(densities, samples, self.parameter["beam_samples"])        
        samples = density_based_samples(densities, distances, beam_origins, beam_directions, beam_ends, self.parameter["beam_samples"])
        samples = samples.reshape(-1,3)
        densities = self.model(samples.cuda())
        predicted_detections = accumulate_beams(densities, samples, self.parameter["beam_samples"]*2)
        loss = criterion(predicted_detections, beam_detections.cuda())
        metrics = self.compute_metrics(predicted_detections, beam_detections)
        return loss.item(), metrics
        

    def test_step(self, batch):
        """
        Perform one test step.

        Args:
            batch (tuple): A batch of data (inputs, targets).
        
        Returns:
            torch.Tensor: The loss for this batch.
            dict: Dictionary of metrics for this batch (e.g., accuracy, F1 score, etc.).
            
        """
        # Implementation could look like this:
        return self.val_step(batch)
    
    def test(self, evaluate_on_full):
        with torch.no_grad():
            super().test(evaluate_on_full)
            # load metadata
            # TODO retrieve from checkpoint
            with open(os.path.join(os.path.join(self.data_path, "noisy-projections"),"metadata.json"), 'r') as file:
                metadata = json.load(file)
            pixelsize = metadata["pixelsize_nmperpixel"]
            slice_thickness_nm = metadata["slice_thickness_nm"]
            original_px_resolution = metadata["original_px_resolution"]
            
            if(os.path.isfile(os.path.join(self.data_path, "phantom-volume", "volume.raw"))):
                test_dataset = Reconstruction_Dataset(self.parameter["resize"], slice_thickness_nm, pixelsize, original_px_resolution, os.path.join(self.data_path,"phantom-volume", "volume.raw"))
                test_dataloader = DataLoader(test_dataset, batch_size=self.parameter["batch_size"], shuffle=False, drop_last = False)
                reconstruction = []
                for samples in tqdm(test_dataloader, desc="Generate Tomogram"): 
                    densities = self.model(samples.cuda())
                    reconstruction.extend(densities.cpu().numpy().astype(np.float16))
                reconstruction = min_max_norm_np(np.array(reconstruction).reshape(test_dataset.x_dim, test_dataset.y_dim, test_dataset.z_dim))
                if(not(test_dataset.volume is None)):
                    mse = criterion(torch.from_numpy(reconstruction), torch.from_numpy(test_dataset.volume))
                    # Calculate average test loss
                    self.logger.log_info(f"MSE phantom: {mse:.4f}") 
                    metric = {"MSE phantom": mse}        
                    self.logger.append_test_results(metric)      
                
                reconstruction = 255-(reconstruction*255).astype(np.uint8).transpose(2,0,1) # (depth, height, width)              
                tiff.imwrite(os.path.join(self.logger.samples_dir, f"tomogram.tif"), reconstruction)

                self.logger.log_info(f"Evaluation Tomogram was saved to {os.path.join(self.logger.samples_dir, f'tomogram.tif')}")




    def save_checkpoint(self, epoch, val_loss):
        """
        Save the current model checkpoint. If the validation loss improves, save the best model.

        Args:
            epoch (int): The current training epoch.
            val_loss (float): The validation loss at the current epoch.
        
        Saves:
            - A checkpoint containing the model state, optimizer state, scheduler state (if applicable), and metadata.
            - The best model based on validation loss.
            - A TorchScript version of the best model for inference.
        """
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'small_model_state_dict': self.model_small.state_dict(),
            'small_optimizer_state_dict': self.optimizer_small.state_dict(),
            'val_loss': val_loss, 
            'parameter': self.parameter,
            'metadata':self.inference_metadata(),
        }
            
        # Save latest model for possible resuming of training
        checkpoint_path = os.path.join(self.logger.checkpoints_dir, f"latest_model.pth")
        torch.save(checkpoint, checkpoint_path)
        
        
        # Save best model for later use    
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0  # Reset the counter since we have improvement
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            
            checkpoint_path = os.path.join(self.logger.checkpoints_dir, f"best_model.pth")
        
            # Save model, optimizer, and scheduler state
            torch.save(checkpoint, checkpoint_path)

            
        else:
            self.patience_counter += 1

    def load_checkpoint(self, checkpoint_path, finetuning = False):
        """
        Load the model, optimizer, and scheduler state from a checkpoint to resume training.

        Args:
            checkpoint_path (str): The file path to the checkpoint.

        Restores:
            - Model weights.
            - Optimizer and scheduler states (if applicable).
            - Best validation loss.
        
        Logs:
            - Information about the resumed training session.
        """
        checkpoint = torch.load(checkpoint_path)
        self.parameter = checkpoint['parameter']
        self.prepare(set_parameters=False, num_epochs=self.num_epochs)
        
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_small.load_state_dict(checkpoint['small_model_state_dict'])
        import traceback
        try:

            print(f"Try to load metadata from: {os.path.join(self.data_path, 'noisy-projections','metadata.json')}")
            

            with open(os.path.join(self.data_path, "noisy-projections","metadata.json"), 'r') as file:
                metadata = json.load(file)
            pixelsize = metadata["pixelsize_nmperpixel"]
            slice_thickness_nm = metadata["slice_thickness_nm"]
            original_px_resolution = metadata["original_px_resolution"]
            if(checkpoint['metadata']['pixelsize_nmperpixel'] != pixelsize):
                self.logger.log_warning(f"The pixel size of the dataset at {self.data_path} ({pixelsize}) differs from the pixelsize of the loaded model ({checkpoint['metadata']['pixelsize_nmperpixel']}). Predictions are likely to be incorrect. Make sure that the model you load for evaluation was trained on the data specified on top of this notebook.")
            if(checkpoint['metadata']['slice_thickness_nm'] != slice_thickness_nm):
                self.logger.log_warning(f"The slice thickness in nm of the dataset at {self.data_path} ({slice_thickness_nm}) differs from the slice thickness in nm of the loaded model ({checkpoint['metadata']['slice_thickness_nm']}). Predictions are likely to be incorrect. Make sure that the model you load for evaluation was trained on the data specified on top of this notebook.")
            if(checkpoint['metadata']['original_px_resolution'] != original_px_resolution):
                self.logger.log_warning(f"The original pixel resolution of the dataset at {self.data_path} ({original_px_resolution}) differs from the original pixel resolution of the loaded model ({checkpoint['metadata']['original_px_resolution']}). Predictions are likely to be incorrect. Make sure that the model you load for evaluation was trained on the data specified on top of this notebook.")
            print(f"Pixelsize: {pixelsize} vs. {checkpoint['metadata']['pixelsize_nmperpixel']}")
            print(f"Pixelsize: {original_px_resolution} vs. {checkpoint['metadata']['original_px_resolution']}")
            print(f"Slice Thickness: {slice_thickness_nm} vs. {checkpoint['metadata']['slice_thickness_nm']}")

        except Exception as e:
            traceback.print_exc()  # This prints the full traceback to stderr
            

        if(not finetuning):

            try:
                # should only save when training was done. Does not save model checkpoint to evaluate. 
                self.save_checkpoint(checkpoint['epoch'], checkpoint['val_loss'])
            except: 
                pass

            self.start_epoch = checkpoint['epoch']
            remaining_epochs = self.num_epochs - self.start_epoch
            if(remaining_epochs <= 0):
                self.logger.log_warning(f"Current number of training epochs ({self.num_epochs}) is smaller or equal than last epoch of the loaded model ({self.start_epoch}). Will train the model for {self.num_epochs} epochs.")
                self.start_epoch = 0
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.optimizer_small.load_state_dict(checkpoint['small_optimizer_state_dict'])
            
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.best_val_loss = checkpoint['val_loss']
            self.patience_counter = 0  # Reset patience counter
            self.logger.log_info(f"Resumed training from checkpoint: {checkpoint_path} (Validation Loss: {self.best_val_loss:.4f}) | Remaining epochs: {self.num_epochs - self.start_epoch}")

        else: 
            self.start_epoch = 0
            self.logger.log_info(f"Loaded model checkpoint for finetuning from: {checkpoint_path} (Validation Loss: {self.best_val_loss:.4f})")
            
        self.model.to(self.device)

    
