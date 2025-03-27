import numpy as np
import torch

# rotates batch of points counterclockwise by angle 
def rotate_y(pos, degree_angle):
    angle = np.deg2rad(degree_angle)
    oy, oz = 0, 0
    px, py, pz = pos.T
    qy = oy + np.cos(angle) * (py - oy) - np.sin(angle) * (pz - oz) 
    qz = oz + np.sin(angle) * (py - oy) + np.cos(angle) * (pz - oz)
    new_point = np.array([px, qy, qz])
    return new_point.T

# function for computing direction d and origin o (Equation 1) of the beam given an input image and its tilt angle.  
# We skip empty space by using the approximated sample slice 
def image_to_beams(image, angle, slice_thickness, remain_shape = False):
    w,h = image.shape
    init_pos = np.zeros((w,h,3))

    # generate 3d positions of 0° tilt micrograph,
    # we define the positions within the 3D reconstruction as a cube in [-1,1]
    x_coords = np.linspace(start = -1, stop = 1, num = w)
    y_coords = np.linspace(start = -1, stop = 1, num = h)
    x_coords, y_coords = np.meshgrid(x_coords,y_coords)
    init_pos[:,:,0] = x_coords.T
    init_pos[:,:,1] = y_coords.T
    init_pos[:,:,2] = 0

    #flatten init_pos to shape [bs, 3]
    init_pos = init_pos.reshape(-1,3)

    # init beam direction for 0° tilt micrograph with shape [bs,3]
    init_dir = np.array([[0,0,-1]])

    # rotate the beam attributes by the given projection angle
    beam_origins = rotate_y(init_pos, angle) # shape [w*h,3]
    beam_directions = rotate_y(init_dir, angle) # shape [1,3]
    beam_directions = np.tile(np.squeeze(beam_directions), (w, h, 1)).reshape(w*h, -1) #shape [w*h, 3]

    sample_origin = np.zeros_like(beam_origins)
    f = (slice_thickness - beam_origins[:,2])/beam_directions[:,2]
    sample_origin[:,0] = beam_directions[:,0]*f + beam_origins[:,0]
    sample_origin[:,1] = beam_directions[:,1]*f + beam_origins[:,1]
    sample_origin[:,2] = beam_directions[:,2]*f + beam_origins[:,2]

    sample_end = np.zeros_like(beam_origins)
    f = (-1*slice_thickness - beam_origins[:,2])/beam_directions[:,2]
    sample_end[:,0] = beam_directions[:,0]*f + beam_origins[:,0]
    sample_end[:,1] = beam_directions[:,1]*f + beam_origins[:,1]
    sample_end[:,2] = beam_directions[:,2]*f + beam_origins[:,2]


    if(remain_shape): 
        beam_origins = beam_origins.reshape(w,h,-1)
        beam_directions = beam_directions.reshape(w,h,-1)
        beam_detection = image
    else: 
        beam_detection = image.reshape(-1)



    return sample_origin, beam_directions, sample_end, beam_detection

# computes the samples along a beam for computing Equation 3
def uniform_samples(beam_origins, beam_directions, beam_ends, num_samples):
    batch_size = beam_origins.shape[0]
    max_distance = torch.sqrt(torch.sum(((beam_origins - beam_ends)**2), dim = 1))
    distances,_ = torch.sort(torch.FloatTensor(batch_size, num_samples).uniform_(0, float(max_distance[0])), dim = -1)
    samples = beam_origins[:,None,:] + beam_directions[:,None,:]*distances[:,:,None]
    return samples.float(), distances # [bs, samples, 3] 

def density_based_samples(densities, distances, beam_origins, beam_directions, beam_ends, beam_samples): 
    densities = densities + torch.abs(torch.min(densities))
    if(torch.sum(densities) == 0):
        return uniform_samples(beam_origins, beam_directions, beam_ends, beam_samples*2)[0]
    batch_size = beam_origins.shape[0]
    densities = densities.reshape((batch_size, beam_samples))
    max_distance = torch.sqrt(torch.sum(((beam_origins - beam_ends)**2), dim = 1))
    bin_size = max_distance[0] / beam_samples 
    offset = torch.cuda.FloatTensor(batch_size, beam_samples*2).uniform_(float(-bin_size/2), float(bin_size/2))

    # compute samples based on densities
    distances_idx = torch.multinomial(densities.reshape(-1).cpu(), batch_size*beam_samples, replacement=True)
    distances = distances.reshape(-1)[distances_idx].reshape((batch_size, beam_samples)).cuda()
    # also add samples with a re uniformly distributed.
    max_distance = torch.sqrt(torch.sum(((beam_origins - beam_ends)**2), dim = 1))
    unif_distances = torch.FloatTensor(batch_size, beam_samples).uniform_(0, float(max_distance[0]))
    distances = torch.cat([unif_distances.cuda(), distances], dim=1)
    distances,_ = torch.sort(distances, dim = -1)
    distances = distances + offset
    samples = beam_origins[:,None,:].cuda() + beam_directions[:,None,:].cuda()*distances[:,:,None]
    return samples.float() 

# computes the euklid distance between two 3d positions. This is used for computing c in Equation (3)
def euklid(x, y):
        dist = (x-y)**2
        dist = torch.sum(dist, dim = 2)
        dist = torch.sqrt(dist)
        return dist

# implementation of Equation (3) on a batch of beam densities 
def accumulate_beams(densities, samples, beam_samples): 
    samples = samples.reshape((-1,beam_samples,3))
    # get distance between two samples
    distances = torch.ones((samples.shape[0], beam_samples), device="cuda") # [bs, samples]
    distances[:,1:] = euklid(samples[:,:-1], samples[:,1:])
    samples = samples.reshape(-1,3)

    # set densities outside reconstruction volume = 0
    bool_array = torch.sum(((samples > 1) | (samples < -1)), dim = 1) > 0
    densities = torch.where(bool_array.cuda(), torch.zeros(bool_array.shape).cuda(), torch.squeeze(densities))
    # reshape densities to single beams: shape = [bs, samples]
    densities = densities.reshape(-1, beam_samples) 
    return torch.exp(-1*torch.sum(densities*distances, dim = 1))