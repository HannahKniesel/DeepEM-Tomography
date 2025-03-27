import os
import datetime
import json
import matplotlib.pyplot as plt
import torch
import psutil
import logging
from pathlib import Path
import re
import numpy as np

from deepEM.Utils import load_json

class Logger:
    def __init__(self, data_path):
        """
        Initializes the Logger, creating a timestamped directory for logs.

        Args:
            data_path (str): The base directory where logs will be stored.
        """
        self.data_path = data_path
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"{Path(data_path).stem}_{timestamp}"
        self.root_dir = os.path.join("logs", name)
        self.init()

    def init(self, file_name=None):
        """
        Initializes logging directories and sets up the logging system.

        Args:
            file_name (str, optional): If provided, creates a subdirectory within the log directory.
        """
        self.log_dir = os.path.join(self.root_dir, file_name) if file_name else self.root_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Create subdirectories for specific logs
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        self.plots_dir = os.path.join(self.log_dir, "plots")
        self.samples_dir = os.path.join(self.log_dir, "samples")

        print(f"Logger initialized. Logs will be saved to: {self.log_dir}")

        # Set up logging to file and console
        self.logger = logging.getLogger("Logger")
        self.logger.setLevel(logging.DEBUG)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        log_file_path = os.path.join(self.log_dir, "log.txt")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def init_directories(self):
        """
        Creates necessary directories for storing checkpoints, plots, and samples.
        """
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        return True

    def log_info(self, message):
        """
        Logs an informational message.

        Args:
            message (str): The message to log.
        """
        self.logger.info(message)

    def log_warning(self, message):
        """
        Logs a warning message.

        Args:
            message (str): The message to log.
        """
        self.logger.warning(message)

    def log_error(self, message):
        """
        Logs an error message.

        Args:
            message (str): The message to log.
        """
        self.logger.error(message)

    def log_best_sweepparameters(self, best_params):
        """
        Saves the best hyperparameters found during a sweep to a JSON file.

        Args:
            best_params (dict): Dictionary containing hyperparameters and validation loss.
        """
        hyperparams_path = os.path.join(self.data_path, "Sweep_Parameters", "best_sweep_parameters.json")
        try:
            curr_val = load_json(hyperparams_path)['val_loss']
        except:
            curr_val = np.inf
            os.makedirs(os.path.join(self.data_path, "Sweep_Parameters"), exist_ok=True)

        if best_params["val_loss"] < curr_val:
            with open(hyperparams_path, "w") as f:
                json.dump(best_params, f, indent=4)
            self.log_info(f"Updated best sweep parameters saved to {hyperparams_path}")

    def load_best_sweep(self):
        """
        Loads the best hyperparameters found during a sweep.

        Returns:
            dict or None: The best hyperparameters if available, otherwise None.
        """
        hyperparams_path = os.path.join(self.data_path, "Sweep_Parameters", "best_sweep_parameters.json")
        try:
            return load_json(hyperparams_path)
        except:
            return None

    def log_hyperparameters(self, hyperparams):
        """
        Saves training hyperparameters to a JSON file.

        Args:
            hyperparams (dict): Dictionary of hyperparameters.
        """
        hyperparams_path = os.path.join(self.log_dir, "hyperparameters.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=4)
        self.log_info(f"Hyperparameters saved to {hyperparams_path}")


    def plot_training_curves(self, train_loss, train_epoch, val_loss, val_epoch):
        """
        Plots and saves training and validation loss curves.

        Args:
            train_loss (list): List of training loss values.
            train_epoch (list): List of training epochs.
            val_loss (list): List of validation loss values.
            val_epoch (list): List of validation epochs.
        """
        plt.figure()
        plt.plot(train_epoch, train_loss, label="Train Loss")
        plt.plot(val_epoch, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")

        plot_path = os.path.join(self.plots_dir, "training_curves.png")
        plt.savefig(plot_path)
        plt.close()
        self.log_info(f"Training curves saved to {plot_path}")


    def get_resource_usage(self):
        """
        Retrieves current system and GPU resource usage.

        Returns:
            tuple: (dict containing resource usage, formatted string representation).
        """
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        ram_usage = memory_info.used / (1024 ** 3)  # GB
        total_ram = memory_info.total / (1024 ** 3)

        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            gpu_memory_allocated = gpu_memory_total = 0

        resources = {
            "CPU": cpu_percent,
            "RAM": ram_usage,
            "Total RAM": total_ram,
            "GPU": gpu_memory_allocated,
            "Total GPU": gpu_memory_total
        }

        resources_str = (
            f"CPU: {cpu_percent}%, RAM: {ram_usage:.2f}GB/{total_ram:.2f}GB, "
            f"GPU: {gpu_memory_allocated:.2f}GB/{gpu_memory_total:.2f}GB\n"
        )
        return resources, resources_str

    def log_resource_usage(self):
        """
        Logs system resource usage to a file.
        """
        resources, resources_str = self.get_resource_usage()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        resource_log_path = os.path.join(self.log_dir, "resource_usage.log")

        with open(resource_log_path, "a") as f:
            f.write(f"{timestamp}: {resources_str}")

    def log_test_results(self, test_loss, metrics):
        save_to = os.path.join(self.log_dir, "test_results.txt")
        result_str = f"Test loss: {test_loss:.4f}\n"
        for metric in metrics.keys(): 
            result_str += f"{metric}: {metrics[metric]:.4f}"   
            
        with open(save_to, "w") as f:
            f.write(result_str)
        return
               

    def get_most_recent_logs(self):
        """
        Retrieves the most recent log directories for each dataname in the folder.
    
        Returns:
            dict: A dictionary where keys are datanames and values are paths to the most recent log directories.
        """
        # Dictionary to store the most recent log directory for each dataname
        most_recent_logs = {}
    
        # Path to the logs folder
        folder_path = "./logs/"
    
        # Regular expression to match the directory structure
        pattern = re.compile(r"(?P<dataname>.+)_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
    
        for dir_name in os.listdir(folder_path):
            match = pattern.match(dir_name)
            if match:
                dataname = match.group("dataname")
                timestamp_str = match.group("timestamp")
                timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
    
                # Update the most recent log for the dataname
                if dataname not in most_recent_logs or timestamp > most_recent_logs[dataname]["timestamp"]:
                    most_recent_logs[dataname] = {
                        "timestamp": timestamp,
                        "path": os.path.join(folder_path, dir_name),
                    }
    
        # Extract only the paths from the dictionary
        return {dataname: info["path"] for dataname, info in most_recent_logs.items()}
