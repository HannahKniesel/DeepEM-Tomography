# DeepEM Playground Template  

This template provides guidelines for contributing your work to the **DeepEM Playground**. 



## Overview

### Jupyter Notebooks  

Each use case consists of two main notebooks:  

- **`1_Development.ipynb`** – Used for model development and hyperparameter tuning.  
- **`2_Inference.ipynb`** – Used for running inference on trained models.  

These notebooks serve as an **interface between deep learning (DL) experts and electron microscopy (EM) experts**. To ensure consistency and simplify the learning process for EM researchers, the notebooks should follow a standardized structure.  

Please update the **markdown cells** in the notebooks to describe your specific use case.  

To assist you:  
- Markdown text requiring your input is **highlighted in** <span style="color:red">**red**</span>.  
- Example markdown descriptions that should be modified for your use case are **marked in** <span style="color:green">**green**</span>.  

Before submitting your use case, **please remove all color formatting**.  

### Project Structure  

The **`deepEM/`** folder contains a lightweight library for implementing your use case.  
- Only modify this library if absolutely necessary.  
- Otherwise, use the provided modules via appropriate imports.  

Your **custom code implementation** should be placed in the `src/` folder.  
- All costum implementations need to implement the `Inferencer.py`, `Model.py` and `ModelTrainer.py` based on their corresponding modules of the `deepEM library`.
- A simple example is provided by implementing [this tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).  
- Adapt this code to implement your use case.  

For library documentation and available functions, refer to [this guide](https://viscom-ulm.github.io/DeepEM/documentation.html).  

### Model Configuration  

The **DeepEM** library manages model parameters through a configuration file:  

- **`configs/parameters.json`** – Defines all **hyperparameters** for training.  
- EM experts can use the **API in `1_Development.ipynb`** to fine-tune hyperparameters via **grid search**.  

### Tunable vs. Non-Tunable Parameters  
- **Tunable hyperparameters** should be clearly documented for adjustment.  
- **Non-tunable hyperparameters** are not accessible to EM experts but should still be included **with explanations** to improve their understanding of the underlying method.  

All parameters—both **tunable and non-tunable**—must be **well-documented**.  

For detailed documentation, see **`configs/README.md`**.  

## Setup

### Lightning AI
<a target="_blank" href="https://lightning.ai/hannah-kniesel/studios/deepem-template">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg"
    alt="Open in Lightning AI Studios" />
</a>

Start immediately using the Lightning AI Studio template by clicking the button above—no additional setup required.

### Local Setup 
For a quick setup, we offer the use of `conda`, `pip` or `docker`. This will provide all needed libraries as well as common libraries used for deep learning (for more details you can check `requirements.txt`). Of course you are free to install additional dependencies, if needed.  

#### Conda (LightningAI)
On your machine, run:
```bash
conda env create -f environment.yml
conda activate deepEM
```

If you are working on [LightingAI](https://lightning.ai/) Studios, there will be a base environment, which you can update with the needed dependencies, by running: 
```bash
conda env update --file environment.yml --prune
```

#### Pip
When working with `pip`, please make sure you have a compatible python version installed. The `deepEM` library was tested on `python == 3.12.5`/`3.11.9` with `cuda==12.1`/`11.8` and `cudnn9`.
Next, you can run
```bash
pip install -r requirements.txt
```

#### Docker
Build your own image with: 
```bash 
docker build -t deepem .
```
This will generate a docker image called `deepem`. 

or use the existing docker image from `hannahkniesel/deepem`. 

Start the docker with this command: 
```bash
docker run --gpus all -it -p 8888:8888 --rm --ipc=host -v /local_dir/:/workspace/ --name <container-name> <image-name> bash
```

Inside the container start `jupyter notebook`
```bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

