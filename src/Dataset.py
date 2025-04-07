
import numpy as np
from glob import glob
import cv2
from torch.utils.data import Dataset
import torch
from scipy.ndimage import zoom
import os
import json


from src.STEM import image_to_beams

# min max normalization over the given values
def min_max_norm_np(values):
    maximum = np.max(values)
    minimum = np.min(values)
    # add 1e-10 for numerical stability
    return (values - minimum)/(maximum - minimum +1e-10) 

# load .rawtlt file from provided path
def load_angles(path):
    try:
        file = open(path, mode='r')
        print(f"INFO:: tilt angles were loaded from {path}")
    except: 
        print(f"ERROR:: .rawtlt file was not found at {path}")
    content = file.read()
    arr = np.array(content.split('\n'))
    arr = arr[arr != '']
    arr = arr.astype(float)
    file.close()
    return arr


# Define dataset for training and validation using the STEM simulator
class TiltSeries_Dataset(Dataset):
    def __init__(self, noisy_dir, percentage = -1, resize = 100, image_format = ".tif"):
        # load metadata
        with open(os.path.join(noisy_dir,"metadata.json"), 'r') as file:
            metadata = json.load(file)
        pixelsize = metadata["pixelsize_nmperpixel"]
        slice_thickness_nm = metadata["slice_thickness_nm"]
        # gather and load micrographs
        image_files = os.path.join(noisy_dir,"*"+image_format)
        noisy_projections = sorted(glob(image_files))
        print(f"INFO::Found {len(noisy_projections)} images within tilt series at {image_files}.")    
        # pick subset for validation
        if(percentage > 0):
            indices = np.random.choice(np.arange(len(noisy_projections)), size=(int(percentage*len(noisy_projections),)), replace=False)
            noisy_projections = [projection for i,projection in enumerate(noisy_projections) if(i in indices)]
            print(f"INFO::Use {int(percentage*100)}% for validation, resulting in {len(noisy_projections)} projection images.")

        noisy_projections = [cv2.imread(noisy_projection, cv2.IMREAD_GRAYSCALE) for noisy_projection in noisy_projections]
        
        

        # compute pixel resolution and slice resolution
        pixel_resolution = noisy_projections[0].shape[0]
        image_resolution_nm = pixelsize*pixel_resolution
        slice_thickness = slice_thickness_nm / image_resolution_nm
        
        self.resize = resize
        # Resizing of the images 
        if(resize):
            init_resolution = noisy_projections[0].shape
            noisy_projections = [cv2.resize(noisy_projection, (resize,resize), interpolation= cv2.INTER_LINEAR) for noisy_projection in noisy_projections]
            print(f"INFO::Resized image resolution from {init_resolution} to {noisy_projections[0].shape}. Note that strong downscaling can lead to loss of information.")

        # Image Normalization: min-max normalization to make micrograph intensities to a fixed range of [0,1]
        noisy_projections = np.array(noisy_projections)
        noisy_projections = min_max_norm_np(noisy_projections)
        
        # load tilt angles
        angles_degree = load_angles(glob(os.path.join(noisy_dir,"*.rawtlt"))[0])
        if(percentage > 0):
            angles_degree = [angle for i,angle in enumerate(angles_degree) if(i in indices)]


        
        self.num_images = len(noisy_projections)

        # Retrieve beam data for STEM simulator from images
        self.val_projection_angles = angles_degree
        sample_from = []
        self.beam_origins, self.beam_directions, self.beam_ends, self.beam_detections = [], [], [], []
        for angle, noisy in zip(angles_degree, noisy_projections):
            beam_origin, beam_direction, beam_end, beam_detection = image_to_beams(noisy, angle, slice_thickness)
            self.beam_origins.extend(beam_origin)
            self.beam_directions.extend(beam_direction)
            self.beam_ends.extend(beam_end)

            # cull beams which are not showing the area of interest (the reconstruction space)
            bool_arr = (beam_end[:,2]>-1*slice_thickness) | (beam_origin[:,2]<slice_thickness) # TODO something seems to be off here
            sample_from.extend(bool_arr)
            self.beam_detections.extend(beam_detection)

        self.indices = np.arange(len(self.beam_detections))[sample_from]

    def __len__(self):
        return len(self.beam_origins)

    def __getitem__(self, idx):
        return self.beam_origins[idx], self.beam_directions[idx], self.beam_ends[idx], self.beam_detections[idx].astype(np.float32)

# Define center slice dataset for validation purposes. 
class CenterSlice_Dataset(Dataset):
    def __init__(self, size = 100):
        xs = torch.linspace(-1, 1, steps=size)
        ys = torch.linspace(-1, 1, steps=size)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        z = torch.zeros_like(x)
        self.coordinates = torch.stack([x,y,z]).permute(1,2,0).reshape(-1,3)

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        return self.coordinates[idx]
    
def open_raw(path, voxel_type='uint8'):
    try:
        data = np.fromfile(path[0], dtype=voxel_type)
    except: 
        try:
            data = np.fromfile(path, dtype=voxel_type)
        except:
            path = glob.glob(path+"/*.raw")
            data = np.fromfile(path[0], dtype=voxel_type)
            

    shape = int(round(data.shape[0]**(1/3),0))
    data = data.reshape((shape,shape,shape))[:,::-1,:]
    data = np.rot90(data, k = 1, axes = (0,1))
    print("Max loaded data: "+str(np.max(data)))
    return data

class Reconstruction_Dataset(Dataset):
    def __init__(self, size, slice_thickness_nm, pixelsize, original_pxres, volume_file = ""):
        
        image_resolution_nm = pixelsize*original_pxres
        slice_thickness = slice_thickness_nm / image_resolution_nm
        
        xs = torch.linspace(-1, 1, steps=size) #width
        ys = torch.linspace(-1, 1, steps=size) #height
        zs = torch.linspace(-slice_thickness, slice_thickness, steps=int(size*slice_thickness)) #depth 
        
        x, y, z = torch.meshgrid(xs, ys, zs)
        self.coordinates = torch.stack([x,y,z]).permute(1,2,3,0)
        self.x_dim, self.y_dim, self.z_dim, _ =  self.coordinates.shape
        self.coordinates = self.coordinates.reshape(-1,3)

        self.volume = None
        if(volume_file!= ""):
            self.volume = open_raw(volume_file)
            init_size = self.volume.shape[0]
            factor = size / init_size
            self.volume = zoom(self.volume, (factor,factor,factor), order=1)  # order=1 for bilinear interpolation
            if(factor != 1):
                print(f"INFO::Original shape of volume ({init_size}, {init_size}, {init_size}) was resized to {self.volume.shape}")
            a = (size-self.z_dim)/2
            b = a + self.z_dim
            self.volume = min_max_norm_np(self.volume).astype(np.float16)[:,:,int(a):int(b)]

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        return self.coordinates[idx]
    




