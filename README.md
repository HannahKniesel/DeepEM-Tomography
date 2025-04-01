# Deep-EM Playground: Bringing Deep Learning to Electron Microscopy Labs

This codebase is part of the Deep-EM Playground. For more details, please see our [webpage](https://viscom-ulm.github.io/DeepEM/).


## 2D to 3D  

### Primary Focus: Tomographic Reconstruction   
### Application: Tomographic Reconstruction of STEM tilt series

#### Challenge: Evaluation with missing ground truth    
#### Required Labels: None

TL;DR 🧬✨ We use deep learning for tomographic reconstruction of 2D STEM projections, following [1,2]. This approach enables 3D volume reconstruction, revealing detailed cellular structures and relationships not visible in 2D.

![Teaser](./images/Teaser.gif)

---

[1] Kniesel, Hannah, et al. "Clean implicit 3D structure from noisy 2D STEM images." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[2] Mildenhall, Ben, et al. "Nerf: Representing scenes as neural radiance fields for view synthesis." Communications of the ACM 65.1 (2021): 99-106.


## Lightning AI 
Get started via Lightning AI. Duplicate the studio by clicking the button below: 


## Docker 
Docker image is available at `hannahkniesel/deepem_tomography`

Start the docker like this: 
```bash
docker run --gpus all -it -p 8888:8888 --rm --ipc=host -v /local_dir/:/workspace/ --name <container-name> hannahkniesel/deepem_tomography bash
```
Inside the container start `jupyter notebook`
```bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
``` 

## Citation

If you find this code useful, please cite us: 

    @inproceedings{
    }
