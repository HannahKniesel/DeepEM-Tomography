import torch
import os
import copy
import math
import time
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader, Subset

from deepEM.Utils import load_json, extract_defaults, get_fixed_parameters, format_time


config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
        
class AbstractModelTrainer(ABC):
    """
    Abstract base class for training, validating, and testing deep learning models.

    Subclasses must implement methods for setting up the model, datasets, optimizer, and scheduler.

    This class manages the entire training pipeline, including training, validation, and testing loops, 
    as well as early stopping and model checkpointing based on validation performance.

    Args:
        data_path (str): Path to the dataset used for training, validation, and testing.
        logger (Logger): Logger instance for logging events and training progress.
        resume_from_checkpoint (str, optional): Path to a checkpoint to resume training from. Defaults to `None`.
        
    Attributes:
        device (str): Device on which the model will be trained ('cuda' if GPU is available, else 'cpu').
        model (torch.nn.Module, optional): The model to be trained. Initialized during the preparation step.
        parameter (dict): Hyperparameters for the model training, including epochs, early stopping patience, etc.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        val_vis_loader (DataLoader): DataLoader for visualizing validation data.
        test_vis_loader (DataLoader): DataLoader for visualizing test data.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler for the optimizer.
        best_val_loss (float): Best validation loss observed for early stopping and checkpointing.
        patience_counter (int): Counter for early stopping based on validation loss.
        best_model_wts (dict): Best model weights for checkpointing.
    """
    
    def __init__(self, data_path, logger, resume_from_checkpoint=None):
        """
        Initializes the trainer class for training, validating, and testing models.

        Args:
            data_path (str): Path to the dataset used for training, validation, and testing.
            logger (Logger): Logger instance for logging events and training progress.
            resume_from_checkpoint (str, optional): Path to a checkpoint to resume training from. Defaults to `None`.
        """
        self.data_path = data_path
        self.logger = logger
        self.resume_from_checkpoint = resume_from_checkpoint
        self.parameter = get_fixed_parameters(os.path.join(config_dir, "parameters.json"))
        
        # Load and update hyperparameters
        hyperparameters = load_json(os.path.join(config_dir, "parameters.json"))
        self.parameter.update(extract_defaults(hyperparameters))
                
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def prepare(self, config=None, train_subset=None, reduce_epochs=None, set_parameters=True):
        """
        Prepares the training pipeline by setting up the model, datasets, dataloaders, 
        optimizer, scheduler, and other configurations.

        Args:
            config (dict, optional): Dictionary of hyperparameters to override the defaults. Defaults to `None`.
            train_subset (float, optional): Fraction of the training dataset to use for quick hyperparameter tuning. Defaults to `None`.
            reduce_epochs (float, optional): Fraction of epochs to use for quick hyperparameter tuning. Defaults to `None`.
            set_parameters (bool, optional): Whether to set the hyperparameters from the provided config. Defaults to `True`.
        """
        if set_parameters:
            if not config:
                hyperparameters = load_json(os.path.join(config_dir, "parameters.json"))
                self.parameter.update(extract_defaults(hyperparameters))
                self.logger.log_warning(f"Could not find a config based on a hyperparameter search. Will use default parameters:\n{self.parameter}\n")
            else:
                self.parameter.update(config)
                
            self.logger.log_hyperparameters(config)
            
        self.train_subset = train_subset
        self.reduce_epochs = reduce_epochs
        
        if not self.model:
            self.model = self.setup_model()
            self.logger.log_info("Model was setup.")
        
        # Setup datasets and dataloaders
        trainset, valset, testset = self.setup_datasets()
        trainset = self.subsample_trainingdata(trainset)
        self.train_loader, self.val_loader, self.test_loader = self.setup_dataloaders(trainset, valset, testset)
        
        self.val_vis_loader, self.test_vis_loader = self.setup_visualization_dataloaders(valset, testset)
        
        # Set epochs and optimizer
        self.set_epochs()
        self.optimizer, self.scheduler = self.setup_optimizer()
        
        # Initialize variables for early stopping and checkpointing
        self.best_val_loss = math.inf
        self.patience_counter = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def set_epochs(self):
        """
        Sets the number of epochs and validation interval based on the configuration.

        The number of epochs and validation interval may be reduced if the `reduce_epochs` 
        parameter is provided.
        """
        self.num_epochs = self.parameter["epochs"]
        self.validation_interval = self.parameter['validation_interval']
        if self.reduce_epochs:
            self.num_epochs = np.max((1, int(self.reduce_epochs * self.num_epochs)))
            self.validation_interval = np.max((1, int(self.reduce_epochs * self.validation_interval)))

    def subsample_trainingdata(self, dataset):
        """
        Subsamples the training dataset if a subset fraction is provided.

        Args:
            dataset (torch.utils.data.Dataset): The training dataset.

        Returns:
            torch.utils.data.Subset: A subset of the dataset if `train_subset` is specified, else the full dataset.
        """
        if self.train_subset:
            num_samples = int(len(dataset) * self.train_subset)
            torch.manual_seed(42)
            shuffled_indices = torch.randperm(len(dataset))[:num_samples]
            subset = torch.utils.data.Subset(dataset, shuffled_indices)
            return subset
        else:
            return dataset

    @abstractmethod
    def setup_model(self):
        """
        Setup and return the model for training, validation, and testing.

        This method must be implemented by the DL expert.

        Returns:
            deepEM.Model.AbstractModel: The initialized model ready for training, validation, and testing.
        """
        raise NotImplementedError("The 'setup_model' method must be implemented by the DL specialist.")

    @abstractmethod
    def setup_datasets(self):
        """
        Setup and return the datasets for training, validation, and testing.

        This method must be implemented by the DL expert.
        
        The data path provided by the EM specialist can be accessed via `self.data_path`.

        Returns:
            tuple: A tuple containing:
                - trainset (torch.utils.data.Dataset): The training dataset.
                - valset (torch.utils.data.Dataset): The validation dataset.
                - testset (torch.utils.data.Dataset): The test dataset.
        """
        raise NotImplementedError("The 'setup_datasets' method must be implemented by the DL specialist.")

    def setup_visualization_dataloaders(self, val_dataset, test_dataset):
        """
        Sets up and returns dataloaders for visualizing a subset of validation and test datasets.

        This method subsamples the `val_dataset` and `test_dataset` to contain `self.parameter["images_to_visualize"]` 
        datapoints. It should be overridden for imbalanced data to select the most interesting samples.

        Args:
            val_dataset (torch.utils.data.Dataset): The validation dataset.
            test_dataset (torch.utils.data.Dataset): The test dataset.

        Returns:
            tuple: A tuple containing:
                - val_vis_loader (torch.utils.data.DataLoader): Dataloader for visualizing a subset of the validation dataset.
                - test_vis_loader (torch.utils.data.DataLoader): Dataloader for visualizing a subset of the test dataset.
        """
        torch.manual_seed(42)
        val_indices = torch.randperm(len(val_dataset))[:self.parameter["images_to_visualize"]]
        vis_val_subset = Subset(val_dataset, val_indices)
        val_vis_loader = DataLoader(vis_val_subset, batch_size=self.parameter["batch_size"], shuffle=False)
        
        test_indices = torch.randperm(len(test_dataset))[:self.parameter["images_to_visualize"]]
        vis_test_subset = Subset(test_dataset, test_indices)
        test_vis_loader = DataLoader(vis_test_subset, batch_size=self.parameter["batch_size"], shuffle=False)

        return val_vis_loader, test_vis_loader

    def setup_dataloaders(self, train_dataset, val_dataset, test_dataset):
        """
        Setup and return the dataloaders for training, validation, and testing.

        This method must be implemented by the DL expert.
        
        The data_path provided by the EM specialist can b accessed via self.data_path
        
        Inputs:
            trainset (torch.utils.data.Dataset): The training dataset.
            valset (torch.utils.data.Dataset): The validation dataset.
            testset (torch.utils.data.Dataset): The test dataset.

        Returns:
            train_loader (torch.utils.data.DataLoader): The dataloader for the training dataset.
            valset_loader (torch.utils.data.DataLoader): The dataloader for the validation dataset.
            testset_loader (torch.utils.data.DataLoader): The dataloader for the test dataset.
        """
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.parameter["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.parameter["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.parameter["batch_size"], shuffle=False)
    

        return train_loader, val_loader, test_loader

    @abstractmethod
    def setup_optimizer(self):
        """
        Setup and return the optimizer and learning rate scheduler.

        This method must be implemented by the DL expert.

        Returns:
            tuple:
                - optimizer (torch.optim.Optimizer): The optimizer for the model.
                - lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        """
        raise NotImplementedError("The 'setup_optimizer' method must be implemented by the DL specialist.")

    @abstractmethod
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for a batch.
        
        Args:
            outputs (torch.Tensor): Model outputs.
            targets (torch.Tensor): Ground truth labels.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        raise NotImplementedError("The 'compute_loss' method must be implemented by the DL specialist.")

    @abstractmethod
    def train_step(self, batch_idx: int, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Perform one training step.

        Args:
            batch_idx (int): The current batch index during training.
            batch (tuple): A batch of data (i.e. inputs, targets).
            
        
        Returns:
            torch.Tensor: The loss for this batch.
        """
        raise NotImplementedError("The 'train_step' method must be implemented by the DL specialist.")

    def qualify(self, dataloader, file_name: str):
        """
        Saves visualizations of model predictions for a given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader containing the data for visualization.
            file_name (str): Prefix for the saved image files.
        """
        idx = 0
        for batch in dataloader:
            images = self.visualize(batch)
            for img in images:
                idx += 1
                img = img.convert("RGBA")
                img.save(os.path.join(self.logger.samples_dir, f"{file_name}_{idx}.png"))
        self.logger.log_info(f"Saved visualizations to {os.path.join(self.logger.samples_dir, f'{file_name}_*')}")

    @abstractmethod
    def inference_metadata(self) -> Dict:
        """
        Returns possible metadata needed for inference (such as class names) as a dictionary.

        This metadata will be saved along with model weights in the training checkpoints.
        
        Returns:
            dict: Dictionary containing metadata.
        """
        raise NotImplementedError("The 'inference_metadata' method must be implemented by the DL specialist.")
    
    @abstractmethod
    def visualize(self, batch) -> List[Image.Image]:
        """
        Visualizes the model's input and output of a single batch and returns them as PIL images.

        Args:
            batch: A batch of data defined by the dataset implementation.
        
        Returns:
            List[PIL.Image]: List of visualizations for the batch data.
        """
        raise NotImplementedError("The 'visualize' method must be implemented by the DL specialist.")

    @abstractmethod
    def val_step(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one validation step.

        Args:
            batch: A batch of data defined by the dataset implementation.
        
        Returns:
            tuple:
                - torch.Tensor: The loss for this batch.
                - dict: Dictionary of metrics for this batch (e.g., accuracy, F1 score, etc.).
        """
        raise NotImplementedError("The 'val_step' method must be implemented by the DL specialist.")

    @abstractmethod
    def test_step(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one test step.

        Args:
            batch: A batch of data defined by the dataset implementation.
        
        Returns:
            tuple:
                - torch.Tensor: The loss for this batch.
                - dict: Dictionary of metrics for this batch (e.g., accuracy, F1 score, etc.).
        """
        raise NotImplementedError("The 'test_step' method must be implemented by the DL specialist.")
        

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
            'val_loss': val_loss, 
            'parameter': self.parameter,
            'metadata':self.inference_metadata(),
        }
            
        # Save latest model for possible resuming of training
        checkpoint_path = os.path.join(self.logger.checkpoints_dir, f"latest_model.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.log_info(f"Current model checkpoint saved to {checkpoint_path}")
        
        
        # Save best model for later use    
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0  # Reset the counter since we have improvement
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            
            checkpoint_path = os.path.join(self.logger.checkpoints_dir, f"best_model.pth")
        
            # Save model, optimizer, and scheduler state
            torch.save(checkpoint, checkpoint_path)

            self.logger.log_info(f"Best model checkpoint saved to {checkpoint_path} (Validation Loss: {val_loss:.4f})")
            
        else:
            self.patience_counter += 1
            self.logger.log_info(f"No improvement in validation loss. Patience counter: {self.patience_counter}/{self.parameter['early_stopping_patience']}")

    def load_checkpoint(self, checkpoint_path):
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
        self.prepare(set_parameters=False)
        
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['val_loss']
        self.patience_counter = 0  # Reset patience counter
        
        
        self.logger.log_info(f"Resumed training from checkpoint: {checkpoint_path} (Validation Loss: {self.best_val_loss:.4f})")

    def train_epoch(self, epoch):
        """
        Perform one full epoch of training and validation.

        Args:
            epoch (int): The current training epoch.

        Returns:
            tuple:
                - (float) Average training loss over the epoch.
                - (float) Average validation loss over the validation set.
                - (bool) Whether early stopping was triggered.
        
        Notes:
            - Uses a learning rate scheduler if available.
            - Saves the best model checkpoint.
            - Applies early stopping if necessary.
        """
        train_loss = 0.0
        val_loss = float("nan")
        metrics_sum = {}  # To store the sum of metrics across batches
        num_batches = len(self.val_loader)

        # Training loop
        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch_idx, batch)
            train_loss += loss
            
            # Step the learning rate scheduler if available
            if self.scheduler and (self.parameter['scheduler_step_by'] == "iteration"):
                self.scheduler.step()
        
        train_loss /= len(self.train_loader)
        self.logger.log_info(f"Epoch {epoch} - Training loss: {train_loss:.4f}")

        # Validation loop
        
        with torch.no_grad():    
            if((epoch % self.validation_interval) == 0):
                val_loss = 0.0
                self.model.eval()
                self.qualify(self.val_vis_loader, f"validation_epoch-{epoch}")
                
                
                for batch in self.val_loader:
                    loss, batch_metrics = self.val_step(batch)
                    val_loss += loss

                    # Accumulate metrics from each batch
                    for metric, value in batch_metrics.items():
                        if metric not in metrics_sum:
                            metrics_sum[metric] = 0.0
                        metrics_sum[metric] += value

                # Calculate average validation loss
                val_loss /= num_batches

                # Calculate average for each metric
                metrics_avg = {metric: value / num_batches for metric, value in metrics_sum.items()}

                # Format metrics into a single log string
                metrics_str = ", ".join(f"{metric}: {avg_value:.4f}" for metric, avg_value in metrics_avg.items())

                # Log validation loss and metrics
                self.logger.log_info(f"Epoch {epoch} - Validation loss: {val_loss:.4f}, {metrics_str}")

                # Save best model checkpoint and apply early stopping
                self.save_checkpoint(epoch, val_loss)

                # Early stopping logic
                if self.patience_counter >= self.parameter['early_stopping_patience']:
                    self.logger.log_info(f"Early stopping triggered at epoch {epoch}.")
                    return train_loss, val_loss, True  # Return True to indicate early stopping

        return train_loss, val_loss, False

    def test(self):
        """
        Evaluate the trained model on the test dataset.

        Returns:
            float: The average test loss.

        Logs:
            - The test loss.
            - Computed metrics averaged over the test set.
        
        Notes:
            - Computes various evaluation metrics and logs them.
            - Uses a visualization function for qualitative assessment.
        """
        with torch.no_grad():
            self.logger.init(f"Evaluate")
            self.logger.init_directories()
            
            self.model.to(self.device)

            self.qualify(self.test_vis_loader, f"test")
            
            test_loss = 0.0
            metrics_sum = {}  # To store the sum of metrics across batches
            num_batches = len(self.test_loader)

            for batch in self.test_loader:
                loss, batch_metrics = self.test_step(batch)
                test_loss += loss

                # Accumulate metrics from each batch
                for metric, value in batch_metrics.items():
                    if metric not in metrics_sum:
                        # metrics_sum[metric] = 0.0
                        metrics_sum[metric] = []
                        
                    # metrics_sum[metric] += value
                    metrics_sum[metric].append(value)
                    

            # Calculate average test loss
            test_loss /= num_batches
            self.logger.log_info(f"Test loss: {test_loss:.4f}")

            # Calculate and log the average of each metric
            # metrics_avg = {metric: value / num_batches for metric, value in metrics_sum.items()}
            metrics_avg = {metric: np.mean(value) for metric, value in metrics_sum.items()}
            
            for metric, avg_value in metrics_avg.items():
                self.logger.log_info(f"{metric}: {avg_value:.4f}")

            self.logger.log_test_results(test_loss, metrics_avg)
            return test_loss

    def fit(self):
        """
        Train the model for the specified number of epochs.

        Returns:
            float: The lowest validation loss achieved during training.
        
        Logs:
            - Training progress, including losses and estimated remaining time.
            - Training and validation loss curves.
        
        Notes:
            - Supports resuming training from a checkpoint.
            - Implements early stopping based on validation loss.
            - Uses a scheduler for learning rate adjustment if available.
        """
        self.logger.init_directories()
        self.logger.log_info(f"Start Training | Epoch: {self.num_epochs} | Dataset size: {len(self.train_loader.dataset)} | Parameters: {self.parameter} ")
        # Resume training from checkpoint if specified, otherwise reset parameters before training
        if self.resume_from_checkpoint:
            self.load_checkpoint(self.resume_from_checkpoint)
        else: 
            self.model.reset_model_parameters_recursive()
            
        self.model.to(self.device)
        
        train_loss_history = []
        val_loss_history = [] 
        val_epoch = []
        train_epoch = []
        accum_time = 0
        for epoch in range(self.num_epochs):
            start_time = time.time()
            train_loss, val_loss, early_stop = self.train_epoch(epoch)
            end_time = time.time()
            elapsed_time = end_time - start_time
            accum_time += elapsed_time
            remaining_time = (self.num_epochs - (epoch+1))*(accum_time/(epoch+1))
            self.logger.log_info(f"Avg time single epoch: {format_time(accum_time/(epoch+1))} | Remaining time training: {format_time(remaining_time)}")
            
            train_loss_history.append(train_loss)
            train_epoch.append(epoch)
            
            if(not math.isnan(val_loss)):
                val_loss_history.append(val_loss)
                val_epoch.append(epoch)
                
            
            self.logger.plot_training_curves(train_loss_history, train_epoch, val_loss_history, val_epoch)

            if early_stop:
                break
            
            # Step the learning rate scheduler if available
            if self.scheduler and (self.parameter['scheduler_step_by'] == "epoch"):
                self.scheduler.step()

        return np.min(val_loss_history)


                