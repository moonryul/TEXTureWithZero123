import time
from pathlib import Path
from typing import Any, Dict, Union, List

import cv2
import einops
import imageio
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_scatter import scatter_max

import torchvision
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.stable_diffusion_depth import StableDiffusion
from src.training.views_dataset import Zero123PlusDataset, ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy, combine_components_to_zero123plus_grid, split_zero123plus_grid_to_components

<<<<<<< HEAD
=======
from PIL import Image, ImageDraw
# JA: scale_latents, unscale_latents, scale_image, and unscale_image are from the Zero123++ pipeline code:
# https://huggingface.co/sudo-ai/zero123plus-pipeline/blob/main/pipeline.py
def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents

def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents

def scale_image(image):
    image = image * 0.5 / 0.8
    return image

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6

class TEXTure:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.paint_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)

        # Make view_dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.view_dirs = ['front', 'left', 'back', 'right', 'overhead', 'bottom'] # self.view_dirs[dir] when dir = [4] = [right]
        self.mesh_model = self.init_mesh_model()
        self.diffusion = self.init_diffusion()

        if self.cfg.guide.use_zero123plus:
            self.zero123plus = self.init_zero123plus()

        self.text_z, self.text_string = self.calc_text_embeddings()
        self.dataloaders = self.init_dataloaders()
        self.back_im = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0,
                                 1) / 255.0

        self.zero123_front_input = None
        
        
        # Set the camera poses:
        self.thetas = []
        self.phis = []
        self.radii = []
       
        for i, data in enumerate(self.dataloaders['train']):
            theta, phi, radius = data['theta'], data['phi'], data['radius']
            phi = phi - np.deg2rad(self.cfg.render.front_offset)
            phi = float(phi + 2 * np.pi if phi < 0 else phi)

            self.thetas.append(theta)
            self.phis.append(phi)
            self.radii.append(radius)
<<<<<<< HEAD
            
        #MJ: check if we use project_back_max_z_normals to learn the max_z_normals texture map   
        if self.cfg.optim.learn_max_z_normals:
           
             #MJ:  use project_back_max_z_normals to learn the max_z_normals texture map   
             self.project_back_max_z_normals()  
             #MJ: Render using the learned self.meta_texture_img             
             meta_outputs = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii,
                                                   background=torch.Tensor([0, 0, 0]).to(self.device),
                                                   use_meta_texture=True, render_cache=None)
             z_normals = meta_outputs["normals"][:,2:3,:,:].clamp(0, 1)
             max_z_normals = meta_outputs['image'][:,0:1,:,:].clamp(0, 1) 
             self.view_weights = self.compute_view_weights(z_normals, max_z_normals, alpha=-2 ) #MJ: try to vary alpha from -1 to -10
         
        
            
        else:
                
=======

        #MJ: check if we use project_back_max_z_normals to learn the max_z_normals texture map  
        # JA: It computes the self.meta_texture_img which contains the best z normals of the pixel
         # images
        #MJ: self.project_back_max_z_normals() 
        if not self.cfg.optim.learn_max_z_normals: #MJ: when you do not use learn_max_z_normals
          
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
            augmented_vertices = self.mesh_model.mesh.vertices

            batch_size = len(self.dataloaders['train'])
            # JA: We need to repeat several tensors to support the batch size.
            # For example, with a tensor of the shape, [1, 3, 1200, 1200], doing
            # repeat(batch_size, 1, 1, 1) results in [1 * batch_size, 3 * 1, 1200 * 1, 1200 * 1]
            mask, depth_map, normals_image,face_normals,face_idx = self.mesh_model.render_face_normals_face_idx(
                augmented_vertices[None].repeat(batch_size, 1, 1),
                self.mesh_model.mesh.faces, # JA: the faces tensor can be shared across the batch and does not require its own batch dimension.
                self.mesh_model.face_attributes.repeat(batch_size, 1, 1, 1),
<<<<<<< HEAD
                elev=torch.tensor(self.thetas).to(self.device),
                azim=torch.tensor(self.phis).to(self.device),
                radius=torch.tensor(self.radii).to(self.device),
=======
                
                elev=torch.tensor( self.thetas).to(self.device), #MJ: elev should a tensor
                azim=torch.tensor( self.phis).to(self.device),
                radius=torch.tensor( self.radii).to(self.device),
                
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
                look_at_height=self.mesh_model.dy,
                background_type='none'
            )
            
<<<<<<< HEAD
            self.mask = mask #MJ: object_mask of the mesh
            self.depth_map = depth_map
            self.normals_image = normals_image
            self.face_normals = face_normals
            self.face_idx = face_idx

            logger.info(f'Generating face view map')

            #MJ: get the binary masks for each view which indicates how much the image rendered from each view
            # should contribute to the texture atlas over the mesh which is the cause of the image
            face_view_map = self.create_face_view_map(face_idx)

            # logger.info(f'Creating weight masks for each view')
            weight_masks = self.compare_face_normals_between_views(face_view_map, face_normals, face_idx)
=======
            logger.info(f'Generating face view map')

            #MJ: get the binary masks for each view which indicates how much the image rendered from each view
            # should contribute to the texture atlas over the mesh which is the cause of the image
            face_view_map = self.create_face_view_map(face_idx)

            # logger.info(f'Creating weight masks for each view')
            weight_masks = self.compare_face_normals_between_views(face_view_map, face_normals, face_idx)

            self.view_weights  = weight_masks
            
            for i in range( len(self.thetas) ):
               
                self.log_train_image(
                    torch.cat((weight_masks[i][None], weight_masks[i][None], weight_masks[i][None]), dim=1),
                    f'debug:weight_masks_{i}' )
                
                
                self.log_train_image(
                    torch.cat(( normals_image[i][None,-1:], normals_image[i][None,-1:], normals_image[i][None,-1:]), dim=1),
                    f'debug:z_normals_{i}' )
                #MJ: face_normals[views, 2, face_ids] 
        
        else: #MJ: when you use learn_max_z_normals, you compute the max_z_normals after you project_back the front view iamge
              #by calling self.paint_viewpoint(), because it requires the initial meta_textuure_img,
              #but learning max_z_normals changes self.meta_texture_img      
              
              #MJ: But before doing it, you need to learn max_z_normals if you use learn_max_z_normals 
      # to define self.view_weights
      
        #if self.cfg.optim.learn_max_z_normals:    
            # # JA: It computes the self.meta_texture_img which contains the best z normals of the pixel
            # # images
            self.project_back_max_z_normals()
            #MJ: Render all views using self.meta_texture_img (max_z_normals) learned  by  self.project_back_max_z_normals()           
            meta_outputs = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii,
                                                    background=torch.Tensor([0, 0, 0]).to(self.device),
                                                    use_meta_texture=True, render_cache=None)
            z_normals = meta_outputs["normals"][:,2:3,:,:].clamp(0, 1)
            max_z_normals = meta_outputs['image'][:,0:1,:,:].clamp(0, 1) 
            self.view_weights = self.compute_view_weights(z_normals, max_z_normals, alpha=self.cfg.optim.alpha) #MJ: = -50 , -100, -10000
            #MJ: try to vary alpha from -1 to -10: If alpha approaches -10, 
            # the face_idx whose z_normals are greater have more weights
    
            #MJ: for debugging:
            max_z_normals_red =  meta_outputs['image'][:,:,:,:].clamp(0, 1)       
            for i in range( len(self.thetas) ):
               
                self.log_train_image(
                    torch.cat((z_normals[i][None], z_normals[i][None], z_normals[i][None]), dim=1),
                    f'z_normals_{i}'
                )
                self.log_train_image( torch.cat( (max_z_normals[i][None], max_z_normals[i][None],
                                                   max_z_normals[i][None]), dim=1), f'max_z_normals_gray_{i}' )
                
                self.log_train_image( max_z_normals_red[i][None], f'max_z_normals_red_{i}' )
                
                self.log_train_image( torch.cat( (self.view_weights[i][None], self.view_weights[i][None], self.view_weights[i][None]), dim=1),
                                      f'view_weights_{i}' )  #MJ: view_weights: (B,1,H,W)
          
        #End of self.cfg.optim.learn_max_z_normals
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6

            self.view_weights  = weight_masks

        
        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')
        
    def compute_view_weights(self, z_normals, max_z_normals, alpha=-10.0 ):        
        
        """
        Compute view weights where the weight increases exponentially as z_normals approach max_z_normals.
        
        Args:
            z_normals (torch.Tensor): The tensor containing the z_normals data.
            max_z_normals (torch.Tensor): The tensor containing the max_z_normals data.
            alpha (float): A scaling parameter that controls how sharply the weight increases (should be negative).
        
        Returns:
            torch.Tensor: The computed weights with the same shape as the input tensors (B, 1, H, W).
        """
        # Ensure inputs have the same shape
        assert z_normals.shape == max_z_normals.shape, "Input tensors must have the same shape"
        
        # Compute the difference between max_z_normals and z_normals
        delta = max_z_normals - z_normals

        # Calculate the weights using an exponential function, multiplying by negative alpha
        weights = torch.exp(alpha * delta)  #MJ: the max value of torch.exp(alpha * delta)   will be torch.exp(alpha * 0) = 1 
        
        # Normalize to have the desired shape (B, 1, H, W)
        #weights = weights.view(weights.size(0), 1, weights.size(1), weights.size(2))
        
        return weights

        
            
    #MJ:    Moved out of the loop; Currently, in every loop the def statement itself defined. The inner functions are also hard to read
    def scale_latents(self,latents): #MJ: move out of the loop
        latents = (latents - 0.22) * 0.75
        return latents

    def unscale_latents(self,latents):
        latents = latents / 0.75 + 0.22
        return latents

    def scale_image(self,image):
        image = image * 0.5 / 0.8
        return image

    def unscale_image(self,image):
        image = image / 0.5 * 0.8
        return image


      

    def create_face_view_map(self, face_idx):
        num_views, _, H, W = face_idx.shape  # Assume face_idx shape is (B, 1, H, W)

        # Flatten the face_idx tensor to make it easier to work with
        face_idx_flattened_2d = face_idx.view(num_views, -1)  # Shape becomes (num_views, H*W)

        # Get the indices of all elements
        # JA: From ChatGPT:
        # torch.meshgrid is used to create a grid of indices that corresponds to each dimension of the input tensor,
        # specifically in this context for the view indices and pixel indices. It allows us to pair each view index
        # with every pixel index, thereby creating a full coordinate system that can be mapped directly to the values
        # in the tensor face_idx.
        view_by_pixel_indices, pixel_by_view_indices = torch.meshgrid(
            torch.arange(num_views, device=face_idx.device),
            torch.arange(H * W, device=face_idx.device),
            indexing='ij'
        )

        # Flatten indices tensors
        view_by_pixel_indices_flattened = view_by_pixel_indices.flatten()
        pixel_by_view_indices_flattened = pixel_by_view_indices.flatten()

        faces_idx_view_pixel_flattened = face_idx_flattened_2d.flatten()

        # Convert pixel indices back to 2D indices (i, j)
        pixel_i_indices = pixel_by_view_indices_flattened // W
        pixel_j_indices = pixel_by_view_indices_flattened % W

        # JA: The original face view map is made of nested dictionaries, which is very inefficient. Face map information
        # is implemented as a single tensor which is efficient. Only tensors can be processed in GPU; dictionaries cannot
        # be processed in GPU.
        # The combined tensor represents, for each pixel (i, j), its view_idx 
        combined_tensor_for_face_view_map = torch.stack([
            faces_idx_view_pixel_flattened,
            view_by_pixel_indices_flattened,
            pixel_i_indices,
            pixel_j_indices
        ], dim=1)

        # Filter valid faces
        faces_idx_valid_mask = faces_idx_view_pixel_flattened >= 0

        # JA:
        # [[face_id_1, view_1, i_1, j_1]
        #  [face_id_1, view_1, i_2, j_2]
        #  [face_id_1, view_1, i_3, j_3]
        #  [face_id_1, view_2, i_4, j_4]
        #  [face_id_1, view_2, i_5, j_5]
        #  ...
        #  [face_id_2, view_1, i_k, j_l]
        #  [face_id_2, view_1, i_{k + 1}, j_{l + 1}]
        #  [face_id_2, view_2, i_{k + 2}, j_{l + 2}]]
        #  ...
        # The above example shows face_id_1 is projected, under view_1, to three pixels (i_1, j_1), (i_2, j_2), (i_3, j_3)
        # Shape is Nx4 where N is the number of pixels (no greater than H*W*num_views = 1200*1200*7) that projects the
        # valid face ID.
        return combined_tensor_for_face_view_map[faces_idx_valid_mask]

    def compare_face_normals_between_views(self,face_view_map, face_normals, face_idx):
        num_views, _, H, W = face_idx.shape
        weight_masks = torch.full((num_views, 1, H, W), True, dtype=torch.bool, device=face_idx.device)

        face_ids = face_view_map[:, 0] # JA: face_view_map.shape = (H*W*num_views, 4) = (1200*1200*7, 4) = (10080000, 4)
        views = face_view_map[:, 1]
        i_coords = face_view_map[:, 2]
        j_coords = face_view_map[:, 3]
        z_normals = face_normals[views, 2, face_ids] # JA: The shape of face_normals is (num_views, 3, num_faces)
                                                     # For example, face_normals can be (7, 3, 14232)
                                                     # z_normals is (N,)

        # Scatter z-normals into the tensor, ensuring each index only keeps the max value
        # JA: z_normals is the source/input tensor, and face_ids is the index tensor to scatter_max function.
        max_z_normals_over_views, _ = scatter_max(z_normals, face_ids, dim=0) # JA: N is a subset of length H*W*num_views
        # The shape of max_z_normals_over_N is the (num_faces,). The shape of the scatter_max output is equal to the
        # shape of the number of distinct indices in the index tensor face_ids.

        # Map the gathered max normals back to the respective face ID indices
        # JA: max_z_normals_over_views represents the max z normals over views for every face ID.
        # The shape of face_ids is (N,). Therefore the shape of max_z_normals_over_views_per_face is also (N,).
        max_z_normals_over_views_per_face = max_z_normals_over_views[face_ids]

        # Calculate the unworthy mask where current z-normals are less than the max per face ID
        unworthy_pixels_mask = z_normals < max_z_normals_over_views_per_face

        # JA: Update the weight masks. The shapes of face_view_map, whence views, i_coords, and j_coords were extracted
        # from, all have the shape of (N,), which represents the number of valid pixel entries. Therefore,
        # weight_masks[views, 0, i_coords, j_coords] will also have the shape of (N,) which allows the values in
        # weight_masks to be set in an elementwise manner.
        #
        # weight_masks[views[0], 0, i_coords[0], j_coords[0]] = ~(unworthy_pixels_mask[0])
        # The above variable represents whether the pixel (i_coords[0], j_coords[0]) under views[0] is worthy to
        # contribute to the texture atlas.
        weight_masks[views, 0, i_coords, j_coords] = ~(unworthy_pixels_mask)

        # weight_masks[views[0], 0, i_coords[0], j_coords[0]] = ~(unworthy_pixels_mask[0])
        # weight_masks[views[1], 0, i_coords[1], j_coords[1]] = ~(unworthy_pixels_mask[1])
        # weight_masks[views[2], 0, i_coords[2], j_coords[2]] = ~(unworthy_pixels_mask[2])
        # weight_masks[views[3], 0, i_coords[3], j_coords[2]] = ~(unworthy_pixels_mask[3])

        return weight_masks

    def init_mesh_model(self) -> nn.Module:
        # fovyangle = np.pi / 6 if self.cfg.guide.use_zero123plus else np.pi / 3
        fovyangle = np.pi / 3
        cache_path = Path('cache') / Path(self.cfg.guide.shape_path).stem
        cache_path.mkdir(parents=True, exist_ok=True)
        model = TexturedMeshModel(self.cfg.guide, device=self.device,
                                  render_grid_size=self.cfg.render.train_grid_size,
                                  cache_path=cache_path,
                                  texture_resolution=self.cfg.guide.texture_resolution,
                                  augmentations=False,
                                  fovyangle=fovyangle)

        model = model.to(self.device)
        logger.info(
            f'Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_diffusion(self) -> Any:
        # JA: The StableDiffusion class composes a pipeline by using individual components such as VAE encoder,
        # CLIP encoder, and UNet
        second_model_type = self.cfg.guide.second_model_type
        if self.cfg.guide.use_zero123plus:
            second_model_type = "zero123plus"

        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          concept_path=self.cfg.guide.concept_path,
                                          latent_mode=False,
                                          min_timestep=self.cfg.optim.min_timestep,
                                          max_timestep=self.cfg.optim.max_timestep,
                                          no_noise=self.cfg.optim.no_noise,
                                          use_inpaint=True,
                                          second_model_type=self.cfg.guide.second_model_type,
                                          guess_mode=self.cfg.guide.guess_mode)

        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model
    
    def init_zero123plus(self) -> DiffusionPipeline:
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16
        )

        pipeline.add_controlnet(ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
        ), conditioning_scale=2)

        pipeline.to(self.device)

        return pipeline

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
            text_string = ref_text
        else:
            text_z = []
            text_string = []
            for d in self.view_dirs:
                text = ref_text.format(d)
                if d != 'front':
                    text = "" # JA: For all non-frontal views, we wish to use a null string prompt
                text_string.append(text)
                logger.info(text)
                negative_prompt = None
                logger.info(negative_prompt)
                text_z.append(self.diffusion.get_text_embeds([text], negative_prompt=negative_prompt))
        return text_z, text_string # JA: text_z contains the embedded vectors of the six view prompts

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        if self.cfg.guide.use_zero123plus:
            init_train_dataloader = Zero123PlusDataset(self.cfg.render, device=self.device).dataloader()
        else:
            init_train_dataloader = MultiviewDataset(self.cfg.render, device=self.device).dataloader()

        val_loader = ViewsDataset(self.cfg.render, device=self.device,
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device,
                                        size=self.cfg.log.full_eval_size).dataloader()  #MJ: not random views
        dataloaders = {'train': init_train_dataloader, 'val': val_loader,
                       'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def paint(self):
        if self.cfg.guide.use_zero123plus:
            start_time = time.perf_counter()  # Record the start time
            
            self.paint_zero123plus()
            end_time = time.perf_counter()  # Record the end time
            total_elapsed_time = end_time - start_time  # Calculate elapsed time

            print(f"Total Elapsed time: {total_elapsed_time:.4f} seconds")

        else:
            self.paint_legacy()

    def paint_zero123plus(self):
        logger.info('Starting training ^_^')
<<<<<<< HEAD
        # Evaluate the initialization
=======
        
        zero123plus_prep_start_time = time.perf_counter()  # Record the end time
        
       
        # Evaluate the initialization: Currently self.texture_img and self.meta_texture_img are not learned at all;
        #MJ:  Because we are not learning the textures sequentially scanning over the viewpoints, this evaluation is meaningless
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
        #MJ: self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        
    # Create the batch of viewpoints:
        self.thetas = []
        self.phis = []
        self.radii = []
        self.viewdata =[]
        for viewpoint, data in enumerate(self.dataloaders['train']):       #MJ: This loop should be moved to the init method     
            #MJ:  self.thetas will contain all views including the first view, which is the front view
            self.viewdata.append(data)
            theta, phi, radius = data['theta'], data['phi'], data['radius']
            phi = phi - np.deg2rad(self.cfg.render.front_offset)
            phi = float(phi + 2 * np.pi if phi < 0 else phi)

            self.thetas.append(theta)  #MJ: Create self.thetas in the init method to save computing time
            self.phis.append(phi)
            self.radii.append(radius)
        #MJ: list self.thetas will be converted to torch.tensor(self.thetas).to(self.device), when it is used in render function.
          
        # JA: Set the background color (gray) to be the same one as is used by Zero123++.
        self.background = torch.Tensor([0.5, 0.5, 0.5]).to(self.device)
        
        #MJ: create the depth maps and the object masks of the mesh from each viewpoint
        self.front_viewdata = self.viewdata[0]  #MJ: get the front view data

        # JA: The first viewpoint should always be frontal. It creates the extended version of the cropped
        # front view image. #MJ: self.paint_viewpoint() is modified to return   
        # outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)

        #self.output_frontview  = self.paint_viewpoint(self.front_viewdata, should_project_back=False)
        self.output_frontview  = self.paint_viewpoint(self.front_viewdata, should_project_back=True)
        self.rgb_output_front = self.output_frontview['image'].detach() #MJ:  self.rgb_output_front.shape: torch.Size([1, 3, 1200, 1200])
        self.object_mask_front =  self.output_frontview['mask'].detach()
        #MJ self.depth_front = self.output_frontview['depth'] #MJ: convert the depth map from sd format to zero123++ format
                
        self.clean_front_image = self.rgb_output_front * self.object_mask_front \
                    + torch.ones_like(self.rgb_output_front, device=self.device) * (1 - self.object_mask_front)
        
        #MJ: self.clean_front_image which depends on the computational graph (rendering func)
        # will be used continuously during the blending operation (after the project_back backpropagation);
        # So, we detach it from the computational graph
        
                    
        #MJ: crop the front view image to the object mask size
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(self.object_mask_front[0, 0])
        
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        
        self.cropped_front_image = crop(self.clean_front_image)
                    
        #MJ: Render all views (including the front view):  render_cache=None by default   
        #MJ:   self.output_rendered_allviews will hold references to tensors that are part of the computational graph.contains the computational graph:
        # Using Saved Tensors: If you've saved intermediate tensors produced during the forward pass and subsequently use them 
        # in computations outside the training loop, you're still accessing the computational graph.
        # Even though you're not explicitly computing gradients,
        # these tensors are part of the graph structure and may have dependencies that affect memory management and graph retention.
        self.output_rendered_allviews = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii, background=self.background) #MJ: Why not providing the multiple viewponits as a batch
        # self.self.output_rendered_allviews=self.output_rendered_allviews.detach()
        #Detach the outputs of the rendering (processing the computational graph) from the computational graph

        # JA: In the depth controlled Zero123++ code example, the test depth map is found here:
        # https://d.skis.ltd/nrp/sample-data/0_depth.png
        # As it can be seen here, the foreground is closer to 0 (black) and background closer to 1 (white).
        # This is opposite of the SD 2.0 pipeline and the TEXTure internal renderer and must be inverted
        # (i.e. 1 minus the depth map, since the depth map is normalized to be between 0 and 1)
        self.depth_allviews = 1 - self.output_rendered_allviews['depth'].detach()
        
        #MJ We detach self.depth_allview for the same reason that we do for    self.clean_front_image:
        
        # MJ: when you do outputs = outputs.detach(), you are creating a detached version of the outputs tensor 
        # and assigning it back to the outputs variable. This means that outputs will now refer to the detached tensor, 
        # and any further operations involving outputs will not be part of the computational graph               
        #MJ: self.rgb_render_raw_allviews = self.output_rendered_allviews['image'] #MJ: self.rgb_render_raw_allviews.shape:torch.Size([7, 3, 1200, 1200])
        
        # JA: Get the Z component of the face normal vectors relative to the camera
        self.z_normals_allviews = self.output_rendered_allviews['normals'].detach()[:, -1:, :, :].clamp(0, 1)  #MJ: z_normals can be computed as self.z_normals in the init method
        #z_normals_cache = meta_output['image'].clamp(0, 1) =>  z_normals_cache is not used

<<<<<<< HEAD
        
        self.z_normals_cache = None
        self.object_mask_allviews = self.output_rendered_allviews['mask'].detach() # JA: mask has a shape of 1200x1200
    
        
        # JA: The generated depth only has one channel, but the Zero123++ pipeline requires an RGBA image.
        # The mask is the object mask, such that the background has value of 0 and the foreground a value of 1.
        self.depth_rgba_allviews = torch.cat((self.depth_allviews, self.depth_allviews, self.depth_allviews, self.object_mask_allviews), dim=1)
        #MJ: self.depth_rgba_allviews.shape: torch.Size([7, 4, 1200, 1200])
        self.depth_rgba_allviews = self.depth_rgba_allviews.detach()     

        #MJ: get the cropped depth images to the object mask sizes
        max_cropped_image_height, max_cropped_image_width = 0, 0
        cropped_depth_sizes = []
        cropped_depths_rgba = []
        for i in range( self.depth_rgba_allviews.shape[0] ):
         
            min_h, min_w, max_h, max_w = utils.get_nonzero_region( self.object_mask_allviews[i][0] )             
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        
            cropped_depth_rgba  = crop( self.depth_rgba_allviews[i][None] )
             
            max_cropped_image_height = max(max_cropped_image_height,  cropped_depth_rgba.shape[2] )
            max_cropped_image_width = max(max_cropped_image_width, cropped_depth_rgba.shape[3])

            cropped_depths_rgba.append(cropped_depth_rgba)
            
         #MJ: interplate the depth maps to the max size: Error:  but got input with spatial dimensions of [761] and output size of (1048, 1048). P
        for i in range(  len(cropped_depths_rgba) ):
            
            cropped_depths_rgba[i] = F.interpolate(
                         cropped_depths_rgba[i],
                        (max_cropped_image_height, max_cropped_image_width),
                        mode='bilinear',
                        align_corners=False
                    )
        
         # JA: cropped_depths_rgba is a list that arranges the rows of the depth map, row by row
        self.cropped_depth_grid = torch.cat((
            torch.cat((cropped_depths_rgba[1], cropped_depths_rgba[4]), dim=3),
            torch.cat((cropped_depths_rgba[2], cropped_depths_rgba[5]), dim=3),
            torch.cat((cropped_depths_rgba[3], cropped_depths_rgba[6]), dim=3),
=======
        # JA: This color is the same as the background color of the image grid generated by Zero123++.
        background_gray = torch.Tensor([0.5, 0.5, 0.5]).to(self.device)
        front_image = None
        
              
        frontview_data_iter = iter(self.dataloaders['train'])
        frontview_data = next(frontview_data_iter)  # Gets the first batch
                        
        front_view_start_time = time.perf_counter()  # Record the start time
        rgb_output_front, object_mask_front = self.paint_viewpoint(frontview_data, should_project_back=True)
        
        front_view_end_time = time.perf_counter()  # Record the end time
        front_view_elapsed_time = front_view_end_time - front_view_start_time  # Calculate elapsed time

        print(f"Elapsed time in Front view image generation (with possible project-back): {front_view_elapsed_time} seconds")

               
        # JA: The object mask is multiplied by the output to erase any generated part of the image that
        # "leaks" outside the boundary of the mesh from the front viewpoint. This operation turns the
        # background black, but we would like to use a white background, which is why we set the inverse 
        # of the mask to a ones tensor (the tensor is normalized to be between 0 and 1).
        front_image = rgb_output_front * object_mask_front \
            + torch.ones_like(rgb_output_front, device=self.device) * (1 - object_mask_front)
                     
             
        outputs_with_magenta_map = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii, background=background_gray)
        rgb_renders_with_magenta_map =  outputs_with_magenta_map['image']
        render_cache =  outputs_with_magenta_map['render_cache'] # 
        
        self.outputs = self.mesh_model.render(
                background=background_gray,
                render_cache=render_cache,
                use_median=True  #MJ: Rendering the mesh again using use_median = True updates the default color
                # region (magenta color) of texture map with the median color of the learned texture map from the front view
                
            )
        
        rgb_renders = self.outputs['image']
        self.render_cache =  self.outputs['render_cache']
        #MJ: prepare the front view image
        min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(object_mask_front[0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_front_image = crop(front_image)
        
        self.log_train_image(front_image, 'paint_zero123plus:front_image')
        self.log_train_image(cropped_front_image, 'paint_zero123plus:cropped_front_image')

        # masks = self.outputs['mask']
        # bounding_boxes = utils.get_nonzero_region_vectorized( masks )    
        # cropped_object_mask = utils.crop_mask_to_bounding_box( masks, bounding_boxes)
        
        
        # JA: In the depth controlled Zero123++ code example, the test depth map is found here:
        # https://d.skis.ltd/nrp/sample-data/0_depth.png
        # As it can be seen here, the foreground is closer to 0 (black) and background closer to 1 (white).
        # This is opposite of the SD 2.0 pipeline and the TEXTure internal renderer and must be inverted
        # (i.e. 1 minus the depth map, since the depth map is normalized to be between 0 and 1)
        depth = 1 - self.outputs['depth']
      

        # JA: The generated depth only has one channel, but the Zero123++ pipeline requires an RGBA image.
        # The mask is the object mask, such that the background has value of 0 and the foreground a value of 1.
        # MJ: the mask is used as the alpha; background alpha = 0 means that it is transparent
        self.object_mask = self.outputs['mask']
        depth_rgba = torch.cat((depth, depth, depth, self.object_mask), dim=1)

         
        #self.bounding_boxes = utils.get_nonzero_region_vectorized( self.object_mask )
        #MJ: depth_rgba.shape:depth_rgba.shape: torch.Size([7, 4, 1200, 1200]) => Do the following for debugging
        B, C, H, W =  self.object_mask.shape 
        self.bounding_boxes = torch.zeros((B, 4), dtype=torch.int32)  # Prepare output tensor

        for i in range( len(self.thetas) ):
            mask_i = self.object_mask[i,0] #MJ: self.object_mask: shape=(7,1,1200,1200); mask_i: shape = (1200,1200)
            bbox_i = utils.get_nonzero_region_tensor(mask_i)
            
            self.bounding_boxes[i,:] =  bbox_i
     
       #cropped_depth_rgba = utils.crop_img_to_bounding_box( depth_rgba, self.bounding_boxes)
        cropped_depths_rgba_list = []
        for i in range( len(self.thetas) ):
            mask_i = self.object_mask[i,0]
            min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(mask_i) #MJ: outputs["mask"][0, 0]: shape (1,1,H,W)
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
            cropped_depth_rgba = crop(depth_rgba[i][None])
            cropped_depths_rgba_list.append(cropped_depth_rgba)
        
        # max_cropped_image_height = max([box[2] - box[0] for box in self.bounding_boxes])
        # max_cropped_image_width = max([box[3] - box[1] for box in self.bounding_boxes])
        #MJ:  bounding_boxes[i, :] = torch.tensor([min_h, min_w, max_h, max_w])   
        
        # Compute the heights and widths
        heights = self.bounding_boxes[:, 2] - self.bounding_boxes[:, 0]
        widths = self.bounding_boxes[:, 3] - self.bounding_boxes[:, 1]

        # Find the maximum height and width
        max_cropped_image_height = torch.max(heights)
        max_cropped_image_width = torch.max(widths)    
        #cropped_depths_rgba = torch.cat( cropped_depths_rgba, dim=0) #MJ => concatenate is impossible because each element in the list has different shapes
        for i, cropped_depth_rgba in enumerate(cropped_depths_rgba_list):
           cropped_depths_rgba_list[i] = F.interpolate(
                    cropped_depth_rgba,
                    (max_cropped_image_height, max_cropped_image_width),
                    mode='bilinear',
                    align_corners=False
                )
            #MJ: cropped_depth_rgba.shape: torch.Size([7, 4, 937, 937])        
        #self.cropped_rgb_render =  utils.crop_img_to_bounding_box(self.rgb_render_full, self.bounding_boxes) 
      
        self.cropped_rgb_renders_with_magenta_map_list  = []
        for i in range( len(self.thetas) ):
            mask_i = self.object_mask[i,0]
            min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(mask_i) #MJ: outputs["mask"][0, 0]: shape (1,1,H,W)
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
            cropped_rgb_render  = crop(  rgb_renders_with_magenta_map[i][None])
            self.cropped_rgb_renders_with_magenta_map_list.append(cropped_rgb_render)
            
        self.cropped_rgb_renders_list  = []
        for i in range( len(self.thetas) ):
            mask_i = self.object_mask[i,0]
            min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(mask_i) #MJ: outputs["mask"][0, 0]: shape (1,1,H,W)
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
            cropped_rgb_render  = crop(  rgb_renders[i][None])
            self.cropped_rgb_renders_list.append(cropped_rgb_render)
        
       # cropped_rgb_renders = torch.cat( cropped_rgb_renders, dim=0) 
       # but got input with spatial dimensions of [3, 886, 886] and output size of (320, 320)               
        cropped_rgb_renders_small_list =[]   
        for i in range( len(self.thetas) ):    
            cropped_rgb_renders_small_list.append( F.interpolate(
                            self.cropped_rgb_renders_list[i], (320, 320),   
                            mode='bilinear',
                            align_corners=False
            ))

        # JA: the Zero123++ pipeline converts the latent space tensor z into pixel space
        # tensor x in the following manner:
        #   x = postprocess(unscale_image(vae_decode(unscale_latents(z) / scaling_factor)))
        # It implies that  to convert pixel space tensor x into latent space tensor z, 
        # the inverse operation must be applied in the following manner:
        #   z = scale_latents(vae_encode(scale_image(preprocess(x))) * scaling_factor)

        cropped_rgb_renders_small = torch.cat( cropped_rgb_renders_small_list, dim=0)

        preprocessed_rgb_renders_small = self.zero123plus.image_processor.preprocess(cropped_rgb_renders_small)
           
        scaled_rgb_renders_small =  scale_image(preprocessed_rgb_renders_small.half() )
        
        scaled_latent_renders_small =  self.zero123plus.vae.encode(
               scaled_rgb_renders_small,
               return_dict=False
            )[0].sample() * self.zero123plus.vae.config.scaling_factor 

        self.gt_latents = scale_latents(scaled_latent_renders_small)
                    
             
       


        #MJ: cropped_depths_rgba_list is a list of tensors 
    
        # # JA: cropped_depths_rgba is a list that arranges the rows of the depth map, row by row
        cropped_depth_grid = torch.cat((
            torch.cat((cropped_depths_rgba_list[1], cropped_depths_rgba_list[4]), dim=3),
            torch.cat((cropped_depths_rgba_list[2], cropped_depths_rgba_list[5]), dim=3),
            torch.cat((cropped_depths_rgba_list[3], cropped_depths_rgba_list[6]), dim=3),
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
        ), dim=2)
        
        # cropped_depth_grid = torch.cat((
        #     torch.cat((cropped_depths_rgba[1][None], cropped_depths_rgba[4][None]), dim=3),
        #     torch.cat((cropped_depths_rgba[2][None], cropped_depths_rgba[5][None]), dim=3),
        #     torch.cat((cropped_depths_rgba[3][None], cropped_depths_rgba[6][None]), dim=3),
        # ), dim=2)

<<<<<<< HEAD
       
        self.log_train_image(self.cropped_front_image, 'cropped_front_image')
        self.log_train_image(self.cropped_depth_grid[:, 0:3], 'cropped_depth_grid') #MJ: self.cropped_depth_grid[:, 0:3].shape: torch.Size([1, 3, 3144, 2096])

        # This is new code. cropped_front_image is the cropped version. In control zero123 pipeline, zero123_front_input is used without padding

        # # JA: depths_rgba_allviews is a tensor that arranges the rows of the depth map, row by row
        # # These depths are not cropped versions
        # tile_size = self.depth_rgba_allviews.shape[2]  #MJ: the height = 1200
        # self.depth_grid = combine_components_to_zero123plus_grid(self.depth_rgba_allviews, tile_size)

        #MJ: self.depth_rgba_allviews[1].shape: torch.Size([4, 1200, 1200])
=======
        #self.log_train_image(cropped_front_image, 'cropped_front_image')
        self.log_train_image(cropped_depth_grid[:, 0:3], 'cropped_depth_grid')
        for i in range( len( self.thetas) ):
           self.log_train_image(self.cropped_rgb_renders_list[i], f'v={i}: cropped_rgb_render')   
      
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
        # JA: From: https://pytorch.org/vision/main/generated/torchvision.transforms.ToPILImage.html
        # Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image
        # while adjusting the value range depending on the mode.
        # From: https://www.geeksforgeeks.org/python-pil-image-resize-method/
        # Parameters: 
        # size – The requested size in pixels, as a 2-tuple: (width, height).

<<<<<<< HEAD
        # JA: Zero123++ was trained with 320x320 images: https://github.com/SUDO-AI-3D/zero123plus/issues/70: 1200 => 
        self.cond_image = torchvision.transforms.functional.to_pil_image( self.cropped_front_image[0]).resize((320, 320))
        self.depth_grid_image = torchvision.transforms.functional.to_pil_image(self.cropped_depth_grid[0] ).resize((640, 960))

        #MJ: Transform the rendered gt images for each view into the latent space so that they can blended with the latent images generated by the
        # image gneration pipeline, the sd depth pipeline for the front view and the zero123++ (depth) pipeline for the other views.
        
         #MJ: When the generated latent tensor is decoded, the Zero123++ pipeline 
        # performs operations in the process of turning the latent space tensor z into pixel space
        # tensor x in the following manner:
        #   x = postprocess(unscale_image( vae_decode(unscale_latents(z)   /  self.vae.config.scaling_factor)))
        # In order to encode space tensor x into latent space tensor z, the inverse must be
        # applied in the following manner:
        #   z = scale_latents( vae_encode( scale_image(preprocess(x))) * self.vae.config.scaling_factor )
        
        #1) As the first step of the transformation, change the size of an image to that of the images assumed by the zero123++                    
        self.front_image_small_pixelspace  =  \
                           F.interpolate(self.clean_front_image, (320, 320), mode='bilinear', align_corners=False) 
                           #MJ: (1,3,1200,1200) => (1,3,320,320)
        
        #2) Process the image format etc:    supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor); in our case,                    
        self.preprocessed_front_image_small  = self.zero123plus.image_processor.preprocess(self.front_image_small_pixelspace) 
        #MJ:    preprocessed_front_image: Tensor in range (0,1): shape = (1,3,1200,1200): # expected range [0,1], normalize to [-1,1]#MJ: Important. do_normalize means: (0,1) => (-1,1) 
                    
       
        #3) Encode the image into the latent space:  x = self.mean + self.std * sample, sample is rand 
        self.clean_front_image_latent = self.scale_latents( 
                        #MJ: image in [-1,1] => latent encoded image                                   
                        self.zero123plus.vae.encode(  #MJ: self.preprocessed_front_image_small.: (B,3,H,W)=(B,3,320,320) => (B,4,H/8, W/8) => (1,4,40,40)
                        self.scale_image( self.preprocessed_front_image_small.half())
                        ).latent_dist.sample() * self.zero123plus.vae.config.scaling_factor
                )

     

        # MJ: Transform the other view images in the same way as the front view image is transformed
        #  These will be used for blending the latent images generated by zero123plus 
        # with the (latent) gt rendered image as the background
        self.rgb_render_raw_allviews = self.output_rendered_allviews['image'].detach() #MJ: self.rgb_render_raw_allviews.shape:torch.Size([7, 3, 1200, 1200])
        self.rgb_render_raw_allviews =    self.rgb_render_raw_allviews.detach()
        
        for i in range(  self.rgb_render_raw_allviews[1:].shape[0]):
                        self.log_train_image(  self.rgb_render_raw_allviews[i][None], 'rgb_render_raw_allviews')   
                     
        self.rgb_render_small_allviews = F.interpolate(self.rgb_render_raw_allviews, (320, 320), mode='bilinear', align_corners=False)

        preprocessed_rgb_render_small_allviews = self.zero123plus.image_processor.preprocess(self.rgb_render_small_allviews) #MJ: rgb_render_small=nan;  rgb_render_small  (320,320)  in range of [0,1] => [-1,1]; handles Pil, ndarray, Tensor image
        #MJ: ??
        ## confer (from TEXTure code)
        #   # JA: inputs is the "cropped_input" which represents the non-zero region of the rendered image of the current texture atlas
        #     pred_rgb_small = F.interpolate(inputs, (image_size, image_size), mode='bilinear',
        #                                  align_corners=False) # JA: Shape of inputs is (827, 827)

        #     # if self.second_model_type in ["zero123", "control_zero123"] and view_dir != "front":
        #     #     latents = None#torch.randn((64, 64), device=self.device)
        #     # else:
        #     latents = self.encode_imgs(pred_rgb_small) # JA: Convert the rgb_render_output to the latent space of shape 64x64



        self.gt_latents_rendered_allviews = self.scale_latents(
            self.zero123plus.vae.encode(  #MJ: encode the rendered gt image: (B,3,H,W) => (B,4,H/8, W/8)
                self.scale_image(preprocessed_rgb_render_small_allviews.half()),
                ).latent_dist.sample() * self.zero123plus.vae.config.scaling_factor
            
        ) #MJ: self.zero123plus.vae.config.scaling_factor = 1/sigma; z * * self.zero123plus.vae.config.scaling_factor  normalizes z
        #MJ: As shown in __call__ of Zero123PlusPipeline, latent condition (the crossattention condition) is NOT normalized during training.
                
                # cond_lat = self.encode_condition_image(image)
                # def encode_condition_image(self, image: torch.Tensor):
                # image = self.vae.encode(image).latent_dist.sample()
                # return image
    
    
        #MJ:   The latent representations of the encoded images: If `return_dict` is True, a
        #     [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        #Note:  
            # moments = self.quant_conv(h)
            # posterior = DiagonalGaussianDistribution(moments)

            # if not return_dict: #MJ: return_dict is True by default
            #     return (posterior,)

            # return AutoencoderKLOutput(latent_dist=posterior)

          
        #MJ: Create the mask for blending
                
        #MJ:  we use the object mask, i.e., outputs['mask'], as curr_mask:
        #MJ: Because cropped_rgb_render_raw is rendered using the unlearned default texture atlas, 
        # cropped_rgb_render_raw will be the same as the default texture color, and so diff will be zero.
        # So exact_generate_mask will be the same as the object_mask, outputs['mask'].
        
    #  diff = (cropped_rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
    #             self.device)).abs().sum(axis=1)
    #         #MJ:  self.default_color = [0.8, 0.1, 0.8] # JA: This is the magenta color, set to the texture atlas
    #         exact_generate_mask = (diff < 0.1).float().unsqueeze(0)
            
        exact_generate_mask = self.object_mask_allviews
        # Extend mask
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)
=======
        # JA: Zero123++ was trained with 320x320 images: https://github.com/SUDO-AI-3D/zero123plus/issues/70
        cond_image = torchvision.transforms.functional.to_pil_image(cropped_front_image[0]).resize((320, 320))
        depth_image = torchvision.transforms.functional.to_pil_image(cropped_depth_grid[0]).resize((640, 960))

        
        
        # # JA: From: https://pytorch.org/vision/main/generated/torchvision.transforms.ToPILImage.html
        # # Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image
        # # while adjusting the value range depending on the mode.
        # # From: https://www.geeksforgeeks.org/python-pil-image-resize-method/
        # # Parameters: 
        # # size – The requested size in pixels, as a 2-tuple: (width, height).

        
        # cropped_rgb_renders_tensor = \
        #         torch.cat((
        #             torch.cat(( cropped_rgb_renders_saved[0],cropped_rgb_renders_saved[3]), dim=3),
        #             torch.cat(( cropped_rgb_renders_saved[1],cropped_rgb_renders_saved[4]), dim=3),
        #             torch.cat(( cropped_rgb_renders_saved[2],cropped_rgb_renders_saved[5]), dim=3),
        #         ), dim=2)
        
       
        zero123plus_prep_end_time = time.perf_counter()  
        elapsed_time =  zero123plus_prep_end_time -  zero123plus_prep_start_time
        print(f'zero123plus_prep_time={elapsed_time:0.4f}')
        
            
        @torch.enable_grad
        def on_step_end(pipeline, iter, t, callback_kwargs):
            
            on_step_end_start_time = time.perf_counter()  # Record the start time
                          
                
            grid_latent = callback_kwargs["latents"]

            latents = split_zero123plus_grid(grid_latent, 320 // pipeline.vae_scale_factor)
            blended_latents = []
            
            
            #cropped_rgb_renders =[]
            
            for v in range( len(self.thetas) ): #MJ: v=0,1,...,6: 
                if v  == 0:
                    continue  #MJ: This loop processes only the results of the zero123plus latents; so skip the front viewpoint

                image_row_index = (v - 1) % 3
                image_col_index = (v - 1) // 3

                latent = latents[image_row_index][image_col_index]

                #MJ: In TEXTure, the latent image generated by the text-to-image pipeline from a given viewpoint
                # is blended/inpainted with the image rendered from the mesh from that viewpoint using the currently learned 
                # texture atlas. This blended latent is used as the input to the next step of denoising.
                # This enforces the generated image to align with the correct image rendered using the correct part of the 
                # texture atlas corresponding to the front viewpoint.  The texture atlas is partly learned from the front view image.
                # 
                # In a new viewpoint, when the image rendered using the partly learned texture atlas 
                # may contain the part which is rendered from the part of the texture atlas with the
                # initial magenta color. If so, it means that for this part, the image should be generated by the generation
                # pipeline. That is, if the newly rendered image is not different from the default texture color, 
                # it is taken to mean that the new region should be painted by the generated image.isidentifier()
                # But, the new region should be well-aligned with the part that is already valid; This part is
                # considered as the background in the context of inpainting.
                
                #MJ: Why use cropped_rgb_render_raw (the image rendered without median_color) to compute diff??
                # diff = ( self.cropped_rgb_renders_list[v].detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
                #      self.device) ).abs().sum(axis=1)
                diff = ( self.cropped_rgb_renders_with_magenta_map_list[v].detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
                      self.device) ).abs().sum(axis=1)
                
                diff_batch = diff.unsqueeze(0)
                self.log_train_image( torch.cat( [diff_batch,diff_batch,diff_batch], dim=1),  f'v=diff-{v}:')        
                exact_generate_mask = (diff < 0.1).float().unsqueeze(0)
                self.log_train_image( torch.cat( [exact_generate_mask,exact_generate_mask,exact_generate_mask], dim=1),  f'v=exact_generate_mask-{v}:')        
                # Dilate the bounary of the  mask
                generate_mask = torch.from_numpy(
                    cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
                    exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

                curr_mask = F.interpolate(
                    generate_mask,
                    (320 // pipeline.vae_scale_factor, 320 // pipeline.vae_scale_factor),
                    mode='nearest'
                )

                self.log_train_image( torch.cat( [curr_mask,curr_mask,curr_mask], dim=1),  f'curr_mask-v={v}') 
               
                # JA: the Zero123++ pipeline converts the latent space tensor z into pixel space
                # tensor x in the following manner:
                #   x = postprocess(unscale_image(vae_decode(unscale_latents(z) / scaling_factor)))
                # It implies that  to convert pixel space tensor x into latent space tensor z, 
                # the inverse operation must be applied in the following manner:
                #   z = scale_latents(vae_encode(scale_image(preprocess(x))) * scaling_factor)


                gt_latent = self.gt_latents[v] #MJ: v=1,2,..5,6
                
                noise = torch.randn_like(gt_latent)
                noised_truth = pipeline.scheduler.add_noise(gt_latent, noise, t[None])

                # 
                # This blending equation is originally from TEXTure
                blended_latent = latent * curr_mask + noised_truth * (1 - curr_mask) 
                #MJ: for debugging, skip the blending to test if the zero123++ generates images correctly            
                #blended_latent = latent

                blended_latents.append(blended_latent)
            #End  for viewpoint_index, data in enumerate(self.dataloaders['train'])
   
        
            callback_kwargs["latents"] = torch.cat((
                torch.cat((blended_latents[0], blended_latents[3]), dim=3),
                torch.cat((blended_latents[1], blended_latents[4]), dim=3),
                torch.cat((blended_latents[2], blended_latents[5]), dim=3),
            ), dim=2).half()
            
            on_step_end_end_time = time.perf_counter()  # Record the end time
            elapsed_time = on_step_end_end_time - on_step_end_start_time # Calculate elapsed time

            print(f"Elapsed time in one on_step_end={i}: {elapsed_time} seconds")

            #MJ: For debugging, save   grid_latent, the latent being denoised at the current step i
            # x = postprocess(unscale_image(vae_decode(unscale_latents(z) / scaling_factor)))
             
            unscaled_latent = unscale_latents(grid_latent)  #MJ: grid_latent = callback_kwargs["latents"]
            scaled_image =  pipeline.vae.decode(
                 unscaled_latent/pipeline.vae.config.scaling_factor,
                return_dict=False
             )[0]  
            unscaled_image = unscale_image(scaled_image)
            #scaled_image.requires_grad = False #MJ: somehow pipeline.vae.decode converts the tensor to the tensor with requires_grad= True
            #MJ: you can only change requires_grad flags of leaf variables. 
            # If you want to use a computed variable in a subgraph that doesn't require differentiation 
            # use var_no_grad = var.detach().
            unscaled_image_no_grad =  unscaled_image.detach()
                   
            do_denormalize = [True] * unscaled_image_no_grad.shape[0]
            postprocessed_image =  pipeline.image_processor.postprocess(unscaled_image_no_grad, output_type='pil', do_denormalize= do_denormalize) #.postprocess does not like it
            #Normalize an image array to [-1,1]. => denormalize into [0,1]
            # Now, let's save this image
            postprocessed_image[0].save(self.train_renders_path/f'step={iter}:decoded_latent_image_in_PIL.jpg')
            
            #MJ: unscaled_image has not been de-normalized into (0,1) by the function which returns it; The de-normalization is performed
            #at the end of the denoising.
            decoded_image = pipeline.image_processor.denormalize(unscaled_image_no_grad[0])
            self.log_train_image( decoded_image[None], f'decoded latent imag in tensor-step={iter}:v={v}')   #scaled_image.max(); 7.5703; scaled_image.min(): -7.8477; 
             #  scaled_image.min(); -12.5547; unscaled_image.max(): 12.1094; unscaled_image.shape: [1, 3, 960, 640]
             
             
            if iter > 30:
                print(f'at iter={iter}:decoded_image.max(), decoded_image.min() = {unscaled_image.max()}, {unscaled_image.min()}')
                # Check which pixel values fall within the range (0, 1)
                valid_pixel_mask = (decoded_image >=0) & (decoded_image <= 1)

                # Calculate the percentage of valid pixels
                percentage_valid_pixels = torch.mean(valid_pixel_mask.float()) * 100  # Convert to percentage

                print(f"Percentage of pixel values within (0, 1): {percentage_valid_pixels.item():.2f}%")
                
                import matplotlib.pyplot as plt

                # Plot a histogram of the pixel values
                plt.hist(decoded_image.view(-1).cpu().numpy(), bins=100, range=(0,1))
                plt.title('Pixel Value Distribution')
                plt.xlabel('Pixel Values')
                plt.ylabel('Frequency')
                # Save the plot as a PNG file
                plt.savefig(self.train_renders_path/f'step={iter}:decoded_image_histogram.jpg')
            #We did the reverse of the following:
            # scaled_rgb_renders_small =  scale_image(preprocessed_rgb_renders_small.half() )
        
            # scaled_latent_renders_small =  self.zero123plus.vae.encode(
            #    scaled_rgb_renders_small,
            #    return_dict=False
            # )[0].sample() * self.zero123plus.vae.config.scaling_factor 

            # self.gt_latents = scale_latents(scaled_latent_renders_small) 
            
            #Consult the zero123plus pipeline _call_():
            
        # latents = unscale_latents(latents)
        # if not output_type == "latent":
        #     image = unscale_image(self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])
        # else:
        #     image = latents

        # image = self.image_processor.postprocess(image, output_type=output_type)
        
             
            return callback_kwargs
        
        #End  def on_step_end(pipeline, i, t, callback_kwargs) 
        # JA: Here we call the Zero123++ pipeline

        zero123_start_time = time.perf_counter()  # Record the end time
        
        result = self.zero123plus(
            cond_image,
            depth_image=depth_image,
            num_inference_steps=50,#36,
            callback_on_step_end=on_step_end
        ).images[0]
        #MJ: return  image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        
        zero123_end_time = time.perf_counter()  # Record the end time
        elapsed_time = zero123_end_time - zero123_start_time # Calculate elapsed time

        print(f"Elapsed time in zero123 sampling: {elapsed_time} seconds")

      #MJ: Now that the images for the 7 views are generated, project back them to construct the texture atlas
      
        project_back_prep_start_time =  time.perf_counter()  # Record the end time
               
        grid_image = torchvision.transforms.functional.pil_to_tensor(result).to(self.device).float() / 255

        self.log_train_image(grid_image[None], 'zero123plus:result_grid_image')

        images = split_zero123plus_grid(grid_image, 320)
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6

        #MJ: downscale the generate_mask to the size of the latent space, because it is used to blend those images in the latent space
        self.curr_mask = F.interpolate(  #MJ: 1200x1200 => 320/8 x 320/ 8
            generate_mask,
            (320 // self.zero123plus.vae_scale_factor, 320 // self.zero123plus.vae_scale_factor),
            mode='bilinear', align_corners=False
        )

        #MJ: The end_step call back function that post-processes the latent images being denoised     
        @torch.enable_grad
        def on_step_end(pipeline, iter, t, callback_kwargs): 
            
            self.iter = iter  #MJ: denoising iteration
            #MJ: call_back_kwargs is created if the callback func, callback_on_step_end, is defined
            # as:  compute the previous noisy sample x_t -> x_t-1
            #    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]called.
            
            grid_latent_otherviews = callback_kwargs["latents"] #MJ: The latent image that has been just denoised one step: of shape (B,4,H,W)=(1,4,120,80); 80=320*2/8
            #MJ: grid_latent refers to the current latent that is denoised one step
            
            #MJ: grod_latent_otherviews range over (-1,1)? 
            #   => They range like a Normal distribution centered on the mean and spreading according to the standard deviation
            
            #MJ: Apply a blending operation to the latent for each view: The blending
            # involves the blending of the generated latent image with the gt rendered image as the background
            
            
            #MJ: convert this single tensor representing a 3x2 grid image to a tensor with 6 samples along the batch dimension
            tile_size = 320 //   pipeline.vae_scale_factor #MJ: 320/8 = 40
            grid_latent_otherviews_batch = split_zero123plus_grid_to_components(grid_latent_otherviews, tile_size) 
            #MJ: 320/8 = 40; tensor => a list of  tensors; grid_latent_otherviews_batch  is in latent space; in cpu??
            # grid_latent_otherviews_batch.shape: torch.Size([6, 4, 40, 40]) 
            #MJ: On the callback_step_end, we will inpaint-blend this grid_latent with the noisy version of the ground gt rendered images as background; But note that the inpainting mask itself is based on the object mask of the gt rendered image
            #Thus, the gt rendered image has double roles as the inpaint mask and the background           
            #MJ: Get the view images being denoised; In the case of the front view image, it is already a clean image denoised;
            # So, we will noisify the front view image at the same level of noise as the other view images being denoised by the callback_step_end 
            # function of the zero123plus pipeline
        
            #MJ: self.gt_latents_rendered_allviews;  self.clean_front_image_latent => They are in latent space
           
            noise = torch.randn_like( self.clean_front_image_latent ) #MJ: self.clean_front_image_latent is an latent image in the same style of that of zero123plus
            noisy_latents_frontview = pipeline.scheduler.add_noise( self.clean_front_image_latent, noise, t[None]) #MJ: t (starts from 999) is used to noisify gt_latents
            #MJ: noisy_latents_small_frontview.shape: torch.Size([1, 4, 40, 40])
            #MJ: self.gt_latents_rendered_allviews is already "noisy" being because they are in the process of being denoised; 
            # Note that they  are generated images, but not rendered ones    
            noisy_latents_generated_allviews = torch.cat( [noisy_latents_frontview,  grid_latent_otherviews_batch]) 
               
            #MJ: Noisify the GT rendered image at the same noise level as the latent  in the latent space, 
            # And then  blend them  with the noisy_latents_generated_allviews
                       
            if self.cfg.optim.interleaving:  
                #MJ: In the case of interleaving, get the new GT images rendered using the texture atlas being learned
                #MJ: Render all views (including the front view)   
                self.output_rendered_allviews = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii, background=self.background) #MJ: Why not providing the multiple viewponits as a batch
                #MJ: The texture atlas has been learned using the current view images (still noisy) when the project_back func was called
                
                self.rgb_render_raw_allviews = self.output_rendered_allviews['image'].detach() #MJ: self.rgb_render_raw_allviews.shape:torch.Size([7, 3, 1200, 1200])
                
                self.rgb_render_small_allviews = F.interpolate(self.rgb_render_raw_allviews, (320, 320), mode='bilinear', align_corners=False)

                #MJ: preprocessed_rgb_render_small_allviews = self.zero123plus.image_processor.preprocess(self.rgb_render_small_allviews) #MJ: rgb_render_small=nan;  rgb_render_small  (320,320)  in range of [0,1] => [-1,1]; handles Pil, ndarray, Tensor image
                # => this preprocess func should not be used here, because it assumes that its input is in PIL RGB image, for example; but
                # in our case,   self.rgb_render_small_allviews is already the normalized image in (0,1) range
                self.gt_latents_rendered_allviews = self.scale_latents(
                    self.zero123plus.vae.encode(  #MJ: encode the rendered gt image: (B,3,H,W) => (B,4,H/8, W/8)
                        self.scale_image(preprocessed_rgb_render_small_allviews.half()),
                        return_dict=False
                    )[0].sample() * self.zero123plus.vae.config.scaling_factor
                )
<<<<<<< HEAD
                #MJ: self.gt_latents_rendered_allviews is considered to be already noisy, because it has been rendered using the noisy
                # texture atlas being "denoised"/learned
                noised_gt_latents_rendered_allviews =   self.gt_latents_rendered_allviews
                
                 # This inpainting blending equation is originally from TEXTure: The blending for each viewpoint is done within the viewpoint loop
                # Adjust the current latent which has been denoised one step, by blending it with the gt rendered image as the background
                #MJ: noisy_latents_generated_allviews.shape: torch.Size([7, 4, 40, 40]);  self.curr_mask.shape:torch.Size([1, 1, 40, 40])
                #MJ: noised_gt_latents_rendered_allviews is also in latent space, 40x40
                
                blended_latents_allviews = noisy_latents_generated_allviews * self.curr_mask +  noised_gt_latents_rendered_allviews * (1 - self.curr_mask) #MJ: noised_truth is nan at the first iteration i=0, t= 999

                #MJ: Blending in batch mode: self.curr_mask is in fact the object_mask, whose edges are dilated, and compressed to the latent space of zero123plus, 320/8x320/8
                #MJ: broadcasting: when the shapes of the tensors align from the last dimension backwards are compatible:
                # Dimensions are compatible when:They are equal, or One of them is 1;
                # The 1s in self.curr_mask allow it to stretch to match the corresponding dimensions in noisy_latents_generated_allviews.
                # The result will have a shape of [7, 4, 40, 40].
                
                  #MJ: combine a set of view blended latent images into 3x2 form tensor to set it to   callback_kwargs["latents"]
                tile_size = 320//self.zero123plus.vae_scale_factor
                callback_kwargs["latents"] = combine_components_to_zero123plus_grid(blended_latents_allviews[1:], tile_size).half()
                
                 
                        
                #MJ: We will project the currently denoised latent, grid_image;
                # But before that, we need to decode it, because project_back() requires the images in the pixel space

                #MJ: confer https://github.com/SUDO-AI-3D/zero123plus/blob/main/diffusers-support/pipeline.py
                
                #In Zero123Plus, after the latents are scaled, they are trained in the latent space;
                # So, to convert them to pixel space, after the denoising, they should be unscaled  
                
                
                unscaled_latents_allviews  = self.unscale_latents( blended_latents_allviews )
                #In Zero123Plus, the images are also scaled for training. So, after denoising, they are unscaled
                decoded_unscaled_image_allviews = self.unscale_image( pipeline.vae.decode( unscaled_latents_allviews.half() / pipeline.vae.config.scaling_factor, return_dict=False)[0] )
                #MJ: Unscaling image means restoring to the original state: The scaled images are normalized ones for traning: Tensor decoded_unscaled_image ranges over [0,1] 
                                                   
                postprocessed_image = pipeline.image_processor.postprocess( decoded_unscaled_image_allviews, output_type="pt")   
                #MJ: Change the shape of the decoded image ( [7,3,320,320]) to the full size (1200,1200) of the rendered image
                #MJ: 
                
                grid_image_tensor = postprocessed_image
                #MJ: grid_image (tensor): shape = (3, 960,640) <= PIL size (640, 960)
                for i in range( grid_image_tensor.shape[0]):
                    self.log_train_image(grid_image_tensor[i][None], 'zero123plus_grid_tensor_image')
                
                tile_size = 320
                grid_image_batch  = split_zero123plus_grid_to_components(grid_image_tensor, tile_size) 

                rgb_outputs = []
                for i in range(   grid_image_batch.shape[0] ):
                    
                    cropped_rgb_output_small =   grid_image_batch[i][None]

                    # JA: Since Zero123++ requires cond tensor and each depth tensor to be of size 320x320, we resize this
                    # to match what it used to be prior to scaling down.
                    cropped_rgb_output = F.interpolate(
                        cropped_rgb_output_small,
                        (max_cropped_image_height, max_cropped_image_width),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    min_h, min_w, max_h, max_w = utils.get_nonzero_region( self.object_mask_allviews[i][0] )             
                                                
                    cropped_rgb_output = F.interpolate(
                        cropped_rgb_output,
                        (max_h - min_h, max_w - min_w),
                        mode='bilinear',
                        align_corners=False
                    )

                    # JA: We initialize rgb_output, the image where cropped_rgb_output will be "pasted into." Since the
                    # renderer produces tensors (depth maps, object mask, etc.) with a height and width of 1200, rgb_output
                    # is initialized with the same size so that it aligns pixel-wise with the renderer-produced tensors.
                    # Because Zero123++ generates non-transparent images, that is, images without an alpha channel, with
                    # a background of rgb(0.5, 0.5, 0.5), we initialize the tensor using torch.ones and multiply by 0.5.
                    rgb_output = torch.ones(
                        cropped_rgb_output.shape[0], cropped_rgb_output.shape[1], 1200, 1200
                    ).to(rgb_output.device) * 0.5

                    rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output

                    rgb_output = F.interpolate(rgb_output, (1200, 1200), mode='bilinear', align_corners=False)

                    rgb_outputs.append(rgb_output)

                    #End   for i in range(   grid_image_batch.shape[0] )
                    # cf:  callback_kwargs["latents"] = combine_components_to_zero123plus_grid(blended_latents_allviews[1:], tile_size).half()
                    
                #End  for i in range(   grid_image_batch.shape[0] )    
                rgb_outputs_tensor = torch.cat(rgb_outputs)                      
                rgb_outputs_otherviews = F.interpolate( rgb_outputs_tensor , (1200, 1200), mode='bilinear', align_corners=False) #MJ: we can transform it as a batch mode
                    
                noisy_rgb_outputs_allviews = torch.cat([self.clean_front_image, rgb_outputs_otherviews ])    
                # JA: Create trimap of keep, refine, and generate using the render output
                #MJ: update_masks is not used; it is the same as object_mask = outputs['mask'] #
                # update_masks.append(viewpoint_data[i]["update_mask"])  
                    
                #End for viewpoint, data in enumerate(self.dataloaders['train'])\
                    
            
                #MJ: Display the seven generated images to be projected back to the texture atlas
                # for v in range( noisy_rgb_outputs_full_pixelspace.shape[0]):
                #      self.log_train_image(noisy_rgb_outputs_full_pixelspace[v:v+1], f'rgb_outputs_allviews_denoising{i}-view{v}')

                                      
                #MJ: project_back uses the full size image of rgb_outputs (1200x1200)
                self.project_back_only_texture_atlas( 
                    thetas=self.thetas, phis=self.phis, radii=self.radii,                                 
                    render_cache=None, background=self.background, rgb_output=noisy_rgb_outputs_allviews,
                    object_mask=self.object_mask_allviews, update_mask=self.object_mask_allviews, z_normals=self.z_normals_allviews, 
                    z_normals_cache=None
                )
            
              
                #MJ: Render using the texture atlas being trained at each denoising step: The newly rendered gt image will be used
                # as the background for the inpaint blending at each denoising step:             
                self.output_rendered_allviews = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii, background=self.background) 
                
                return  callback_kwargs  
                #MJ: callback_kwargs["latents"] is the latent postprocessed (blending operation is applied in our case)
                #MJ: callback_kwargs["latents"] will be used as the input to the next denoising step through the Unet
                    
                                             
                
            else:  #MJ:  non-interleaving:  self.output_rendered_allviews, which has been created outside of the caLLback_step_end
                # will be used; the GT images that was rendered using the initial (random) texture atlas are used
                   #MJ: In this case, the GT rendered image has the default background (white?); The object  region will be inpainted
                   # by the blending operation; So in this case the GT rendering has only the role of  providing the object mask for blending
                
            
                #MJ:  Noisify the GT rendered image at the same noise level as the latent  in the latent space
                noises = torch.randn_like(self.gt_latents_rendered_allviews)
                noised_gt_latents_rendered_allviews = pipeline.scheduler.add_noise(self.gt_latents_rendered_allviews , noises, t[None]) #MJ: t is used to noisify gt_latents; g_latents = nan
                 
                    # Adjust the current latent which has been denoised one step, by blending it with the gt rendered image as the background
                #MJ: noisy_latents_generated_allviews.shape: torch.Size([7, 4, 40, 40]);  self.curr_mask.shape:torch.Size([1, 1, 40, 40])
                #MJ: noised_gt_latents_rendered_allviews is also in latent space, 40x40
                
                blended_latents_allviews = noisy_latents_generated_allviews * self.curr_mask +  noised_gt_latents_rendered_allviews * (1 - self.curr_mask) #MJ: noised_truth is nan at the first iteration i=0, t= 999
         
                #MJ: Blending in batch mode: self.curr_mask is in fact the object_mask, whose edges are dilated, and compressed to the latent space of zero123plus, 320/8x320/8
                #MJ: broadcasting: when the shapes of the tensors align from the last dimension backwards are compatible:
                # Dimensions are compatible when:They are equal, or One of them is 1;
                # The 1s in self.curr_mask allow it to stretch to match the corresponding dimensions in noisy_latents_generated_allviews.
                # The result will have a shape of [7, 4, 40, 40].
                
                #MJ: combine a set of view blended latent images into 3x2 form tensor to set it to   callback_kwargs["latents"]
                tile_size = 320//self.zero123plus.vae_scale_factor
                callback_kwargs["latents"] = combine_components_to_zero123plus_grid(blended_latents_allviews[1:], tile_size).half()
             
                #MJ: callback_kwargs["latents"] will be used as the input to the next denoising step through the Unet
                    
                #MJ: non-interleaving mode: call project_back at the end of the denoising steps
                #MJ: combine self.clean_front_image and the result of zero.zero123plus to create the input view images for project-back
                    
                    #noisy_rgb_outputs_full_pixelspace =  self.clean_front_image + result 
                    #But this involves coverting the PIL images of "result" to their tensor versions. 
                    # We could do this operation using ballback of zero123pipeline: https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L117
                    # But doing it with callback_step_end callback when i == len(timesteps) - 1 would be just fine. We will do it.
                if iter == len(pipeline.scheduler.timesteps)-1 :   #MJ: len(pipeline.scheduler.timesteps) =36
                    #MJ: Decode the denoised latent zero123plus grid image (the other views in latent space)
                    unscaled_latents_otherviews  = self.unscale_latents( blended_latents_allviews[1:] )
                    #In Zero123Plus, the images are also scaled for training.
                    decoded_image =  pipeline.vae.decode(unscaled_latents_otherviews.half() / pipeline.vae.config.scaling_factor)[0] #MJ: or .sample
                   #  after denoising, they are unscaled: decoded_image.min()=-0.9849; decoded_image.max()=0.8354: 
                    decoded_unscaled_image_otherviews = self.unscale_image( decoded_image ) #MJ: decoded_unscaled_image: min=-1.57; max=1.3369
            
                         
                    postprocessed_image = pipeline.image_processor.postprocess( decoded_unscaled_image_otherviews, output_type="pt")   
                    #MJ: When output_type ='pt' (Pytorch), postprocessed_image ranges over (0,1)                 
                    
                    # grid_image_tensor = torchvision.transforms.functional.pil_to_tensor(postprocessed_image).to(self.device).float() / 255
                    # #MJ: Change the shape of the decoded image ( [7,3,320,320]) to the full size (1200,1200) of the rendered image
                    # #MJ: 
                    
                    grid_image_tensor = postprocessed_image 
                    #MJ:  grid_image_tensor.shape: torch.Size([6, 3, 320, 320])
                    for i in range(grid_image_tensor.shape[0]):
                        self.log_train_image(grid_image_tensor[i][None], 'zero123plus_grid_tensor_image')
                       
                    
                    # tile_size = 320
                    # grid_image_batch  = split_zero123plus_grid_to_components(grid_image_tensor, tile_size) 

                    rgb_outputs = []
                    for i in range(   grid_image_tensor.shape[0] ): #MJ: grid_image_tensor.shape: torch.Size([6, 3, 320, 320])
                        
                        cropped_rgb_output_small =   grid_image_tensor[i][None]

                        # JA: Since Zero123++ requires cond tensor and each depth tensor to be of size 320x320, we resize this
                        # to match what it used to be prior to scaling down.
                        cropped_rgb_output = F.interpolate(
                            cropped_rgb_output_small,
                            (max_cropped_image_height, max_cropped_image_width),
                            mode='bilinear',
                            align_corners=False
                        )
                        
                        min_h, min_w, max_h, max_w = utils.get_nonzero_region( self.object_mask_allviews[1:][i][0] )             
                                                    
                        cropped_rgb_output = F.interpolate(
                            cropped_rgb_output,
                            (max_h - min_h, max_w - min_w),
                            mode='bilinear',
                            align_corners=False
                        )

                        # JA: We initialize rgb_output, the image where cropped_rgb_output will be "pasted into." Since the
                        # renderer produces tensors (depth maps, object mask, etc.) with a height and width of 1200, rgb_output
                        # is initialized with the same size so that it aligns pixel-wise with the renderer-produced tensors.
                        # Because Zero123++ generates non-transparent images, that is, images without an alpha channel, with
                        # a background of rgb(0.5, 0.5, 0.5), we initialize the tensor using torch.ones and multiply by 0.5.
                        rgb_output = torch.ones(
                            cropped_rgb_output.shape[0], cropped_rgb_output.shape[1], 1200, 1200
                        ).to(  cropped_rgb_output.device) * 0.5

                        rgb_output[0, :, min_h:max_h, min_w:max_w] = cropped_rgb_output

                        rgb_output = F.interpolate(rgb_output, (1200, 1200), mode='bilinear', align_corners=False)

                        rgb_outputs.append(rgb_output)  #MJ: concat the list along dim=0, batch dim                                   

                    #End   for i in range(   grid_image_batch.shape[0] )
                    # cf:  callback_kwargs["latents"] = combine_components_to_zero123plus_grid(blended_latents_allviews[1:], tile_size).half()
                    rgb_outputs_tensor = torch.cat(rgb_outputs)
                    
                    rgb_outputs_otherviews = F.interpolate( rgb_outputs_tensor , (1200, 1200), mode='bilinear', align_corners=False) #MJ: we can transform it as a batch mode
                    
                    rgb_outputs_allviews = torch.cat([self.clean_front_image, rgb_outputs_otherviews ]) #MJ: concat the list along dim=0, batch dim                                   
                    
                    
                     #MJ: project_back  the full size image of rgb_outputs (1200x1200) to the texture atlas
                    project_back_return_msg = self.project_back_only_texture_atlas( 
                        thetas=self.thetas, phis=self.phis, radii=self.radii,                                 
                        render_cache=None, background=self.background, rgb_output=rgb_outputs_allviews, #MJ: rgb_outputs are in [0,1]
                        object_mask=self.object_mask_allviews, update_mask=self.object_mask_allviews, z_normals=self.z_normals_allviews, 
                        z_normals_cache=None
                    )
                    print( project_back_return_msg)
                    #MJ: for debugging
                    here ="stop"
                    
                
                #end if i == len(pipeline.scheduler.timesteps)-1 :   #MJ: len(pipeline.scheduler.timesteps) =36
                
                return   callback_kwargs  #MJ: This is the latent postprocessed (blending operation is applied in our case)
                      
                #end else of  if self.cfg.optim.interleaving                                
            #End  if self.cfg.optim.interleaving                            
            
                   
        #End  def on_step_end(pipeline, i, t, callback_kwargs)
=======

                min_h,min_w, max_h, max_w = self.bounding_boxes[i]
                #MJ: Use self.bounding_boxes instead of the following code:
      
                # nonzero_region = viewpoint_data[i]["nonzero_region"]
                # min_h, min_w = nonzero_region["min_h"], nonzero_region["min_w"]
                # max_h, max_w = nonzero_region["max_h"], nonzero_region["max_w"]

                cropped_rgb_output = F.interpolate(
                    cropped_rgb_output,
                    (max_h - min_h, max_w - min_w),
                    mode='bilinear',
                    align_corners=False
                )

                # JA: We initialize rgb_output, the image where cropped_rgb_output will be "pasted into." Since the
                # renderer produces tensors (depth maps, object mask, etc.) with a height and width of 1200, rgb_output
                # is initialized with the same size so that it aligns pixel-wise with the renderer-produced tensors.
                # Because Zero123++ generates non-transparent images, that is, images without an alpha channel, with
                # a background of rgb(0.5, 0.5, 0.5), we initialize the tensor using torch.ones and multiply by 0.5.
                rgb_output = torch.ones(
                    cropped_rgb_output.shape[0], cropped_rgb_output.shape[1], 1200, 1200
                ).to(rgb_output.device) * 0.5

                rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output

                rgb_output = F.interpolate(rgb_output, (1200, 1200), mode='bilinear', align_corners=False)
            #End else:
            rgb_outputs.append(rgb_output)
        #End for i, data in enumerate(self.dataloaders['train'])
        
        #MJ: Project-back the generated view images, rgb_outputs, to the texture atlas.
        # outputs = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii, background=background)

        # render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)

        # # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        # outputs = self.mesh_model.render(background=background,  #When render_cache is not None, the camera pose is not needed
        #                                 render_cache=render_cache, use_median=True)

        # # Render meta texture map
        # meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
        #                                     use_meta_texture=True, render_cache=render_cache)

        # # JA: Get the Z component of the face normal vectors relative to the camera
        # z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        # z_normals_cache = meta_output['image'].clamp(0, 1)
        # object_mask = outputs['mask'] # JA: mask has a shape of 1200x1200

        #MJ:  self.project_back_only_texture_atlas:
        # self.project_back_only_texture_atlas(
        #      render_cache=render_cache, background=background, rgb_output=torch.cat(rgb_outputs),
        #     object_mask=object_mask, update_mask=object_mask, z_normals=z_normals, z_normals_cache=z_normals_cache
            
        # )
        
        project_back_prep_end_time =  time.perf_counter()  # Record the end time
        
        elapsed_time = project_back_prep_end_time - project_back_prep_start_time
        print(f'project-back prep time={elapsed_time:0.4f}')
        
        
        project_back_start_time =  time.perf_counter()  # Record the end time
        self.project_back_only_texture_atlas(   #When render_cache is not None, the camera pose is not needed
             render_cache=self.render_cache, background=background_gray, rgb_output=torch.cat(rgb_outputs),
            object_mask=self.object_mask, update_mask=self.object_mask, z_normals=None, z_normals_cache=None
            
        )
        
                
        project_back_end_time =  time.perf_counter()  # Record the end time 
        
        elapsed_time = project_back_end_time - project_back_start_time
        print(f'project-back  time={elapsed_time:0.4f}')
        
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
        
        # JA: Here we call the Zero123++ pipeline
        result = self.zero123plus(
            self.cond_image,
            depth_image=self.depth_grid_image,
            num_inference_steps=36,
            callback_on_step_end=on_step_end
        ).images[0]

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        
        #MJ: Render the mesh using the learned texture atlas from a lot of regularly sampled viewpoints
        # and create a video from the rendered image sequence and 
        self.full_eval() 
        logger.info('\t All Done!')

    def paint_legacy(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        pbar = tqdm(total=len(self.dataloaders['train']), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # JA: The following loop computes the texture atlas for the given mesh using ten render images. In other words,
        # it is the inverse rendering process. Each of the ten views is one of the six view images.
        for data in self.dataloaders['train']:
            self.paint_step += 1
            pbar.update(1)
            self.paint_viewpoint(data) # JA: paint_viewpoint computes the part of the texture atlas by using a specific view image
            self.evaluate(self.dataloaders['val'], self.eval_renders_path)  # JA: This is the validation step for the current
                                                                            # training step
            self.mesh_model.train() # JA: Set the model to train mode because the self.evaluate sets the model to eval mode.

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        self.full_eval()
        logger.info('\tDone!')

<<<<<<< HEAD
    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        #MJ: logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        logger.info(f'Evaluating and saving model, painting iteration...')
=======
    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False): #MJ: dataloader=self.dataloaders['val']
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video: #MJ: false in our experiment
            all_preds = []
<<<<<<< HEAD
        for i, data in enumerate(dataloader):  #MJ: len(dataloader) = 100; 
            preds, textures, depths, normals = self.eval_render(data)
=======
        for i, data in enumerate(dataloader):
            preds, textures, depths, normals = self.eval_render(data) #MJ: preds, textures, depths, normals = rgb_render, texture_rgb, depth_render, pred_z_normals
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6

            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
<<<<<<< HEAD
                Image.fromarray(pred).save(save_path / f"{i:04d}_rgb.jpg")
                Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                    save_path / f'{i:04d}_normals_cache.jpg')
=======
                Image.fromarray(pred).save(save_path / f"eval:rendered_image:{i:04d}_rgb.jpg")
                Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                    save_path / f'eval:normal_map:{i:04d}_normals_cache.jpg')
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
                if self.paint_step == 0:
                    # Also save depths for debugging
                    torch.save(depths[0], save_path / f"eval:depth_map:{i:04d}_depth.pt")

        # MJ: There are 100 predicted rendered images; the texture maps are also returned, but the texture is the same for all views
        # just take the last result, the 100th
        texture = tensor2numpy(textures[0])
<<<<<<< HEAD
        Image.fromarray(texture).save(save_path / f"texture.png")
=======
        Image.fromarray(texture).save(save_path / f"eval:texture_atlas:texture.png")
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

<<<<<<< HEAD
            dump_vid = lambda video, name: imageio.mimsave(save_path / f"{name}.mp4", video,
=======
            dump_vid = lambda video, name: imageio.mimsave(save_path / f"eval:constructed_video:{name}.mp4", video,
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
                                                           fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'all_rendered_rgb')
        logger.info('Eval Done!')

    def full_eval(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = self.final_renders_path #MJ: = "results"
        #MJ: Create the render images from a 100 random viewpoints dataset and create a video from it    
        self.evaluate(self.dataloaders['val_large'], output_dir, save_as_video=True)
        # except:
        #     logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\t Full Eval Done!")

    # JA: paint_viewpoint computes a portion of the texture atlas for the given viewpoint
    def paint_viewpoint(self, data: Dict[str, Any], should_project_back=True): #MJ: called with  should_project_back=False
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius'] # JA: data represents a viewpoint which is stored in the dataset
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

        # Set background image
        if  True: #self.cfg.guide.second_model_type in ["zero123", "control_zero123"]: #self.view_dirs[data['dir']] != "front":
            # JA: For Zero123, the input image background is always white
            background = torch.Tensor([1, 1, 1]).to(self.device)
        elif self.cfg.guide.use_background_color: # JA: When use_background_color is True, set the background to the green color
            background = torch.Tensor([0, 0.8, 0]).to(self.device)
        else: # JA: Otherwise, set the background to the brick image
            background = F.interpolate(self.back_im.unsqueeze(0),
                                       (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                       mode='bilinear', align_corners=False)

        # Render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background, use_batch_render=False) #MJ: MJ ADD use_batch_render=False
        render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)
        rgb_render_raw = outputs['image']  # Render where missing values have special color
        depth_render = outputs['depth']
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
<<<<<<< HEAD
        #MJ: The following call is not used in zero123plus version; MJ: use_media = False by default (that is at self.paint_step=0)
        outputs = self.mesh_model.render(background=background,  #MJ: self.paint_step  refers to the viewpoint step
                                         render_cache=render_cache, use_median=self.paint_step > 1, use_batch_render=False)
        rgb_render = outputs['image']
        # Render meta texture map
        
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                              use_meta_texture=True, render_cache=render_cache, use_batch_render=False)

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        z_normals_cache = meta_output['image'].clamp(0, 1)
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]
        
        self.log_train_image(rgb_render, 'rgb_render')
        self.log_train_image(depth_render[0, 0], 'depth', colormap=True)
        self.log_train_image(z_normals[0, 0], 'z_normals', colormap=True)
        #MJ: self.log_train_image(z_normals_cache[0, 0], 'z_normals_cache', colormap=True)
=======
        #MJ: In ConTEXTure, the use of the following render does not change anything, because use_median is False
        # when self.paint_step = 1 (in the case of the front view painting)
        outputs = self.mesh_model.render(background=background,
                                         render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render = outputs['image']
        # # Render meta texture map: MJ: in ConTEXTure, we use paint_viewpoint only for the front viewpoint, and we do not need to consider the meta_texture (max_z_normals) here
        # meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
        #                                      use_meta_texture=True, render_cache=render_cache)

        # z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        # z_normals_cache = meta_output['image'].clamp(0, 1)
        # edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

        self.log_train_image(rgb_render, 'paint_viewpoint:rgb_render')
        self.log_train_image(depth_render[0, 0], 'paint_viewpoint:depth', colormap=True)
        # self.log_train_image(z_normals[0, 0], 'paint_viewpoint:z_normals', colormap=True)
        # self.log_train_image(z_normals_cache[0, 0], 'paint_viewpoint:z_normals_cache', colormap=True)
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs] # JA: dirs is one of the six directions. text_z is the embedding vector of the specific view prompt
            text_string = self.text_string[dirs]
        else:
            text_z = self.text_z
            text_string = self.text_string
        logger.info(f'text: {text_string}')

<<<<<<< HEAD
        # JA: Create trimap of keep, refine, and generate using the render output
        
        update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache,
                                                                        edited_mask=edited_mask,
                                                                        mask=outputs['mask'])
=======
        #MJ: Because we do not consider each viewpoint in a sequence to project-back the view image to the texture-atlas, we do not use the trimap
        # # JA: Create trimap of keep, refine, and generate using the render output
        # update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
        #                                                                 depth_render=depth_render,
        #                                                                 z_normals=z_normals,
        #                                                                 z_normals_cache=z_normals_cache,
        #                                                                 edited_mask=edited_mask,
        #                                                                 mask=outputs['mask'])
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6

        # update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
        # if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
        #     logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
        #     return

        # self.log_train_image(rgb_render * (1 - update_mask), name='paint_viewpoint:masked_rgb_render')
        # self.log_train_image(rgb_render * refine_mask, name='paint_viewpoint:refine_rgb_render')

        # Crop to inner region based on object mask
<<<<<<< HEAD
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
        
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]  #MJ: select the object region from tensor x
        
=======
        min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(outputs['mask'][0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
        cropped_rgb_render = crop(rgb_render) # JA: This is rendered image which is denoted as Q_0.
                                              # In our experiment, 1200 is cropped to 827
        cropped_depth_render = crop(depth_render)
        
        #MJ: Because we do not consider each viewpoint in a sequence to project-back the view image to the texture-atlas,
        # we do not use the trimap and so not use update_mask and refine_mask; object_mask is used as the update_mask
        #MJ: cropped_update_mask = crop(update_mask)
        
        update_mask = outputs['mask']
        cropped_update_mask = crop(update_mask)
<<<<<<< HEAD
       
        self.log_train_image(cropped_rgb_render, name='cropped_input')
        self.log_train_image(cropped_depth_render.repeat_interleave(3, dim=1), name='cropped_depth')
=======
        self.log_train_image(cropped_rgb_render, name='paint_viewpoint:cropped_rgb_render')
        self.log_train_image(cropped_depth_render.repeat_interleave(3, dim=1), name='paint_viewpoint:cropped_depth')
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6

        checker_mask = None
        
         #MJ: Because we do not consider each viewpoint in a sequence to project-back the view image to the texture-atlas,
        # we do not use the trimap and so not use update_mask and refine_mask
        
        # if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
        #     # JA: generate_checkerboard is defined in formula 2 of the paper
        #     checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask),
        #                                               crop(generate_mask))
        #     self.log_train_image(F.interpolate(cropped_rgb_render, (512, 512)) * (1 - checker_mask),
        #                          'checkerboard_input')
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1
        # JA: self.zero123_front_input has been added for Zero123 integration
        if self.zero123_front_input is None:
            resized_zero123_front_input = None
        else: # JA: Even though zero123 front input is fixed, it will be resized to the rendered image of each viewpoint other than the front view
            resized_zero123_front_input = F.interpolate(
                self.zero123_front_input,
                (cropped_rgb_render.shape[-2], cropped_rgb_render.shape[-1]) # JA: (H, W)
            )

        condition_guidance_scales = None
        if self.cfg.guide.individual_control_of_conditions:
            if self.cfg.guide.second_model_type != "control_zero123":
                raise NotImplementedError

            assert self.cfg.guide.guidance_scale_i is not None
            assert self.cfg.guide.guidance_scale_t is not None

            condition_guidance_scales = {
                "i": self.cfg.guide.guidance_scale_i,
                "t": self.cfg.guide.guidance_scale_t
            }

        # JA: Compute target image corresponding to the specific viewpoint, i.e. front, left, right etc. image
        # In the original implementation of TEXTure, the view direction information is contained in text_z. In
        # the new version, text_z 
        # D_t (depth map) = cropped_depth_render, Q_t (rendered image) = cropped_rgb_render.
        # Trimap is defined by update_mask and checker_mask. cropped_rgb_output refers to the result of the
        # Modified Diffusion Process.

        # JA: So far, the render image was created. Now we generate the image using the SD pipeline
        # Our pipeline uses the rendered image in the process of generating the image.
        
    
        start_time = time.perf_counter()  # Record the start time
        #rgb_output_front, object_mask_front = self.paint_viewpoint(data, should_project_back=True)
        
        
        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, cropped_rgb_render.detach(), # JA: We use the cropped rgb output as the input for the depth pipeline
                                                                    cropped_depth_render.detach(),
                                                                    guidance_scale=self.cfg.guide.guidance_scale,
                                                                    strength=1.0, update_mask=cropped_update_mask,
                                                                    fixed_seed=self.cfg.optim.seed,
                                                                    check_mask=checker_mask,
                                                                    intermediate_vis=self.cfg.log.vis_diffusion_steps,

                                                                    # JA: The following were added to use the view image
                                                                    # created by Zero123
                                                                    view_dir=self.view_dirs[dirs], # JA: view_dir = "left", this is used to check if the view direction is front
                                                                    front_image=resized_zero123_front_input,
                                                                    phi=data['phi'],
                                                                    theta=data['base_theta'] - data['theta'],
                                                                    condition_guidance_scales=condition_guidance_scales)

<<<<<<< HEAD
        self.log_train_image(cropped_rgb_output, name='cropped_rgb_output')
=======

        end_time = time.perf_counter()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        print(f"Elapsed time in self.diffusion.img2img_step in TEXTureWithZero123: {elapsed_time:0.4f} seconds")
        
        self.log_train_image(cropped_rgb_output, name='paint_viewpoint:cropped_rgb_output')
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
        self.log_diffusion_steps(steps_vis)
        # JA: cropped_rgb_output, as the output of sd pipeline, always has a shape of (512, 512); recover the resolution of the nonzero rendered image (e.g. (827, 827))
        cropped_rgb_output = F.interpolate(cropped_rgb_output, 
                                           (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
                                           mode='bilinear', align_corners=False)

        # Extend rgb_output to full image size
        # JA: After the image is generated, we insert it into the original RGB output
        rgb_output = rgb_render.clone() # JA: rgb_render shape is 1200x1200
        rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output # JA: For example, (189, 1016, 68, 895) refers to the nonzero region of the render image
        self.log_train_image(rgb_output, name='rgb_output_noncropped')

        # Project back
        object_mask = outputs['mask'] # JA: mask has a shape of 1200x1200
        # JA: Compute a part of the texture atlas corresponding to the target render image of the specific viewpoint
<<<<<<< HEAD
        
        if should_project_back:  #MJ: not used in zero123plus version
            fitted_pred_rgb, _ = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                                object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                                z_normals_cache=z_normals_cache)
            self.log_train_image(fitted_pred_rgb, name='fitted')
=======
        if should_project_back:
            fitted_pred_rgb = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                                object_mask=object_mask, update_mask=update_mask, z_normals=None, #MJ z_normals=z_normals,
                                                z_normals_cache=None #MJ: z_normals_cache=z_normals_cache
                                                )
                                                                                             
            self.log_train_image(fitted_pred_rgb, name='paint_viewpoint:fitted_pred_rgb')
            #MJ: for debugging: what happens if the following update of rgb_output is not made, 
            # when the previous rgb_output which was not rendered using the learned texture atlas from the front viewpoint.
            # rgb_output = fitted_pred_rgb  #MJ: fitted_pred_rgb  is the rendered image obtained using the learned texture atlas from the current view
            # outputs['image'] =  rgb_output  #MJ: This is not need for the current experiment
            
            
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6

        # JA: Zero123 needs the input image without the background
        # rgb_output is the generated and uncropped image in pixel space
        zero123_input = crop(
            rgb_output * object_mask
            + torch.ones_like(rgb_output, device=self.device) * (1 - object_mask)
        )   # JA: In the case of front view, the shape is (930,930).
            # This rendered image will be compressed to the shape of (512, 512) which is the shape of the diffusion
            # model.

        if self.view_dirs[dirs] == "front":
            self.zero123_front_input = zero123_input
        
        # if self.zero123_inputs is None:
        #     self.zero123_inputs = []
        
        # self.zero123_inputs.append({
        #     'image': zero123_input,
        #     'phi': data['phi'],
        #     'theta': data['theta']
        # })

<<<<<<< HEAD
        self.log_train_image(zero123_input, name='zero123_input(front_image)')
=======
        self.log_train_image(zero123_input, name='paint_viewpoint:zero123_cond_image')

        return rgb_output, object_mask
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6

        #MJ: return rgb_output, object_mask
        return outputs
    
    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        rgb_render = rgb_render * (1 - uncolored_mask) + \
            utils.color_with_shade( [0.85, 0.85, 0.85], z_normals=z_normals,
                                                        light_coef=0.3) * uncolored_mask

        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                     dims=(dim, dim), use_median=True,
                                                     render_cache=outputs['render_cache'])

        meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                             background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=outputs['render_cache'])
        
        pred_z_normals = meta_output['image'][:, :1].detach()
        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        
        texture_rgb = outputs_with_median['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        
        depth_render = outputs['depth'].permute(0, 2, 3, 1).contiguous().detach()

        return rgb_render, texture_rgb, depth_render, pred_z_normals

    def calculate_trimap(self, rgb_render_raw: torch.Tensor,
                         depth_render: torch.Tensor,
                         z_normals: torch.Tensor, z_normals_cache: torch.Tensor, edited_mask: torch.Tensor,
                         mask: torch.Tensor):
        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0)

        # Extend mask
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

        update_mask = generate_mask.clone()

        object_mask = torch.ones_like(update_mask)
        object_mask[depth_render == 0] = 0
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        # Generate the refine mask based on the z normals, and the edited mask

        refine_mask = torch.zeros_like(update_mask)
        
        #MJ: we do not use z_normals_cache
        #refine_mask[z_normals > (z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr) ] = 1
        
        if self.cfg.guide.initial_texture is None:
            #MJ: we do not use z_normals_cache
            #refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0
            pass
        elif self.cfg.guide.reference_texture is not None:
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            # Don't use bad angles here: It assumes that z_normals will be greater than 0.4 in other views
            refine_mask[z_normals < 0.4] = 0
        else:
            # Update all regions inside the object
            refine_mask[mask == 0] = 0

        refine_mask = torch.from_numpy(
            cv2.erode(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        refine_mask = torch.from_numpy(
            cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        update_mask[refine_mask == 1] = 1

        update_mask[torch.bitwise_and(object_mask == 0, generate_mask == 0)] = 0

        # Visualize trimap
        if self.cfg.log.log_images:
            trimap_vis = utils.color_with_shade(color=[112 / 255.0, 173 / 255.0, 71 / 255.0], z_normals=z_normals)
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1
            trimap_vis = trimap_vis * (1 - exact_generate_mask) + utils.color_with_shade(
                [255 / 255.0, 22 / 255.0, 67 / 255.0],
                z_normals=z_normals,
                light_coef=0.7) * exact_generate_mask  #MJ: exact_generate_mask = (diff < 0.1): The area which is close to the background, to be written on

            shaded_rgb_vis = rgb_render_raw.detach()
            shaded_rgb_vis = shaded_rgb_vis * (1 - exact_generate_mask) + utils.color_with_shade([0.85, 0.85, 0.85],
                                                                                                 z_normals=z_normals,
                                                                                                 light_coef=0.7) * exact_generate_mask

            if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
                refinement_color_shaded = utils.color_with_shade(color=[91 / 255.0, 155 / 255.0, 213 / 255.0],
                                                                 z_normals=z_normals)
                only_old_mask_for_vis = torch.bitwise_and(refine_mask == 1, exact_generate_mask == 0).float().detach()
                trimap_vis = trimap_vis * 0 + 1.0 * (trimap_vis * (
                        1 - only_old_mask_for_vis) + refinement_color_shaded * only_old_mask_for_vis)
            self.log_train_image(shaded_rgb_vis, 'newly_shaded_rgb_vis')
            self.log_train_image(trimap_vis, 'trimap_vis')

        return update_mask, generate_mask, refine_mask

    def generate_checkerboard(self, update_mask_inner, improve_z_mask_inner, update_mask_base_inner):
        checkerboard = torch.ones((1, 1, 64 // 2, 64 // 2)).to(self.device)
        # Create a checkerboard grid
        checkerboard[:, :, ::2, ::2] = 0
        checkerboard[:, :, 1::2, 1::2] = 0
        checkerboard = F.interpolate(checkerboard,
                                     (512, 512))
        checker_mask = F.interpolate(update_mask_inner, (512, 512))
        only_old_mask = F.interpolate(torch.bitwise_and(improve_z_mask_inner == 1,
                                                        update_mask_base_inner == 0).float(), (512, 512))
        checker_mask[only_old_mask == 1] = checkerboard[only_old_mask == 1]
        return checker_mask
    
    #MJ: project_back() used only for the front view image, so we do not need to learn the meta-texture-img which holds
    #    the max_z_normals

    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor):
        eroded_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            eroded_mask = cv2.erode(object_mask[i, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))
            eroded_masks.append(torch.from_numpy(eroded_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        eroded_object_mask = torch.cat(eroded_masks, dim=0)
        # object_mask = torch.from_numpy(
        #     cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
        #     object_mask.device).unsqueeze(0).unsqueeze(0)
        # render_update_mask = object_mask.clone()
        render_update_mask = eroded_object_mask.clone()

        # render_update_mask[update_mask == 0] = 0
        render_update_mask[update_mask == 0] = 0

        # blurred_render_update_mask = torch.from_numpy(
        #     cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
        #     render_update_mask.device).unsqueeze(0).unsqueeze(0)
        dilated_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            dilated_mask = cv2.dilate(render_update_mask[i, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))
            dilated_masks.append(torch.from_numpy(dilated_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        blurred_render_update_mask = torch.cat(dilated_masks, dim=0)
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0

        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
            # Do not use bad normals
            if z_normals is not None and z_normals_cache is not None:
                z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
                blurred_render_update_mask[z_was_better] = 0

        render_update_mask = blurred_render_update_mask
        for i in range(rgb_output.shape[0]):
            self.log_train_image(rgb_output[i][None] * render_update_mask[i][None], f'project_back_input_{i}')

        # Update the normals:
        # project_back() used only for the front view image, so we do not need to learn the meta-texture-img which holds
        #    the max_z_normals
        # if z_normals is not None and z_normals_cache is not None:
        #     z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])

        optimizer = torch.optim.Adam(self.mesh_model.get_params_texture_atlas(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
            
        # JA: Create the texture atlas for the mesh using each view. It updates the parameters
        # of the neural network and the parameters are the pixel values of the texture atlas.
        # The loss function of the neural network is the render loss. The loss is the difference
        # between the specific image and the rendered image, rendered using the current estimate
        # of the texture atlas.
        # losses = []
        with  tqdm(range(200), desc='project_back (SD2): fitting mesh colors for the front view') as pbar:
          for i in pbar:  
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache)
            rgb_render = outputs['image']

            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]
            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean()

            #MJ: 
            # project_back() used only for the front view image, so we do not need to learn the meta-texture-img which holds
            #    the max_z_normals
            # if z_normals is not None and z_normals_cache is not None:
            #     meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
            #                                         use_meta_texture=True, render_cache=render_cache)
            #     current_z_normals = meta_outputs['image']
            #     current_z_mask = meta_outputs['mask'].flatten()
            #     masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
            #                             current_z_mask == 1][:, :1]
            #     masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
            #                             current_z_mask == 1][:, :1]
            #     loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
            # losses.append(loss.cpu().detach().numpy())
            
            pbar.set_description(f"project_back (SD2): Fitting mesh colors -Epoch {i + 1}, Loss: {loss.item():.7f}")
            loss.backward() # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                            # the network, that is, the pixel value of the texture atlas
            optimizer.step()

        # if z_normals is not None and z_normals_cache is not None:
        #     return rgb_render, current_z_normals
        # else:
        
        #MJ: with  tqdm(range(200), desc='project_back (SD2): fitting mesh colors for the front view') as pbar:
        return rgb_render
        
    def project_back_only_texture_atlas(self,  thetas: List[float], phis: List[float], radii: List[float], 
                                        render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
<<<<<<< HEAD
                     z_normals_cache: torch.Tensor
=======
                      z_normals_cache: torch.Tensor                   
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
                     ):
        
        optimizer = torch.optim.Adam(self.mesh_model.get_params_texture_atlas(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
            
        eroded_masks = []
        
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            eroded_mask = cv2.erode(object_mask[i, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))
            eroded_masks.append(torch.from_numpy(eroded_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        eroded_object_mask = torch.cat(eroded_masks, dim=0)
        render_update_mask = eroded_object_mask.clone()
        render_update_mask[update_mask == 0] = 0

        dilated_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            dilated_mask = cv2.dilate(render_update_mask[i, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))
            dilated_masks.append(torch.from_numpy(dilated_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        blurred_render_update_mask = torch.cat(dilated_masks, dim=0)
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0

        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0

        render_update_mask = blurred_render_update_mask
<<<<<<< HEAD
        
        if self.cfg.optim.interleaving:
=======
               
        for i in range(rgb_output.shape[0]):
           self.log_train_image(rgb_output[i][None] * render_update_mask[i][None], f'project_back_input_{i}')  
           
        optimizer = torch.optim.Adam(self.mesh_model.get_params_texture_atlas(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
            
            for i in range(rgb_output.shape[0]):
                self.log_train_image(rgb_output[i][None] * render_update_mask[i][None], f'iter={self.iter}: project_back_input_{i}')
        else:
            for i in range(rgb_output.shape[0]):
                self.log_train_image(rgb_output[i][None] * render_update_mask[i][None], f'project_back_input_{i}')
                 
       
        # JA: Create the texture atlas for the mesh using each view. It updates the parameters
        # of the neural network and the parameters are the pixel values of the texture atlas.
        # The loss function of the neural network is the render loss. The loss is the difference
        # between the specific image and the rendered image, rendered using the current estimate
        # of the texture atlas.
        # losses = []
<<<<<<< HEAD
        if self.cfg.optim.interleaving:
          weight_of_curr_project_back =  (self.iter+1) // 36
        else:
          weight_of_curr_project_back = 1
               
        with tqdm( range(self.cfg.optim.epochs * weight_of_curr_project_back),desc='Project_back:fitting mesh colors') as pbar: #MJ: epochs = 150
           
          for i in pbar:
            optimizer.zero_grad() 
            # Calling optimizer.zero_grad() resets the gradients of all optimized parameters to zero
            # before the backward pass, effectively preparing them for the next iteration of optimization. 
            # This step is crucial to ensure that gradients from previous iterations do not accumulate 
            # and affect the current iteration's update.
            
           
            #MJ: self.mesh_model.render that produces outputs is the computational graph, that is, the network model
            outputs = self.mesh_model.render(theta=thetas, phi=phis, radius=radii, background=background,
                                            render_cache=render_cache)   #MJ: In my experiment, render_cache = None
            #MJ: render the mesh using the texture atlas being learned within this epoch loop
            rgb_render = outputs['image']

            # loss = (render_update_mask * (rgb_render - rgb_output.detach()).pow(2)).mean()
            loss = (render_update_mask * self.view_weights * (rgb_render - rgb_output.detach()).pow(2)).mean()
            
            if self.cfg.optim.interleaving:
                    pbar.set_description(f"Project-back: Fitting mesh colors -At Iter ={self.iter}, Epoch {i + 1}, Loss: {loss.item():.4f}")
            else:
                    pbar.set_description(f"Project-back: Fitting mesh colors -Epoch {i + 1}, Loss: {loss.item():.4f}")
            
            
            #MJ: rgb)output is the result of blending the generated latent image with the gt rendered image; It contains the computation tree
            # used to render the image; This tree contains the learnable parameters. To make these not affected during training, we detach rgb_output.
            loss.backward(retain_graph=True) #MJ: The computational graph is used outside the training loop by the render function output; so retain the graph
            # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
            # the network, that is, the pixel value of the texture atlas
            
            # Trying to backward through the graph a second time (or directly access saved tensors 
            # after they have already been freed) causes an error. Saved intermediate values of the graph
            # are freed when you call .backward() or autograd.grad(). Specify retain_graph=True 
            # if you need to backward through the graph a second time or if you need to access saved tensors 
            # after calling backward.
                      
            optimizer.step()
            
            #MJ:  Update the description of the progress bar with current loss:
            #MJ: The following code performs tensor accesses after the first backward pass, which could lead to the error.
            # if self.cfg.optim.interleaving:
            #         pbar.set_description(f"Project-back: Fitting mesh colors -At Iter ={self.iter}, Epoch {_ + 1}, Loss: {loss.item():.4f}")
            # else:
            #         pbar.set_description(f"Project-back: Fitting mesh colors -Epoch {_ + 1}, Loss: {loss.item():.4f}")
          #End  for _ in pbar
          here = "stop"  #MJ: I am returning to the callback_step_end func for the last time
        #End    with tqdm( range(self.cfg.optim.epochs * weight_of_curr_project_back),desc='fitting mesh colors') as pbar
        return "project_back_texture_atlas"     
        #MJ: return rgb_render

    def project_back_max_z_normals(self):
        
        optimizer = torch.optim.Adam(self.mesh_model.get_params_max_z_normals(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                        eps=1e-15)
            
        for v in range( len(self.thetas) ):    # scan over the 7 views 
            
            #MJ: Render the max_z_normals using  self.meta_texure_img which has been learned using the previous view z_normals
            # At the beginning of the for loop, self.meta_texture_img is set to 0
            
            meta_output = self.mesh_model.render(theta=self.thetas[v], phi=self.phis[v], radius=self.radii[v],
                                                  background=torch.Tensor([0, 0, 0]).to(self.device),
                                                        use_meta_texture=True, render_cache=None)
            render_cache =  meta_output['render_cache']
            curr_max_z_normals_pred = meta_output['image'][:,0:1,:,:].clamp(0,1)   #MJ: z_normals_cache is the projected meta_texture_img, which is zero at first
            curr_z_normals = meta_output['normals'][:,2:3,:,:].clamp(0,1)   
           
            #MJ: z_normals is the z component of the normal vectors of the faces seen by each view
            curr_z_mask = meta_output['mask']   #MJ: shape = (1,1,1200,1200)
            curr_z_mask_flattened = curr_z_mask.flatten()  #MJ: curr_z_mask_flattened.shape: torch.Size([1440000]) = 1200 x 1200
                       
            max_z_normals_target = torch.tensor(curr_max_z_normals_pred)   #MJ: max_z_normals_target will refer to a copy of max_z_normals_pred
            max_z_normals_target[0, 0, :, :] = torch.max(curr_max_z_normals_pred[0, 0, :, :], curr_z_normals[0, 0, :, :])    
            
            self.log_train_image(torch.cat([max_z_normals_target, max_z_normals_target,max_z_normals_target], dim=1),
                                 f'max_z_normals_target')
            masked_curr_z_normals_target = curr_z_normals.reshape(1, 1, -1)[:, :,curr_z_mask_flattened == 1]  
            #MJ:  curr_z_normals.reshape(1, 1, -1)[:, :,curr_z_mask_flattened == 1]: shape = (1,1,390898)   
            #MJ: curr_z_normals: (1,1,1200,1200)                    
            with  tqdm(range(200), desc='fitting max_z_normals') as pbar:
                for i in pbar:
                    optimizer.zero_grad()
                    
                    
                    masked_max_z_normals_pred = \
                         curr_max_z_normals_pred.reshape(1, 1, -1)[:, :,curr_z_mask_flattened == 1]                   
              
                    loss = (masked_max_z_normals_pred - masked_curr_z_normals_target.detach()).pow(2).mean()      
                    
                    
                #confer the code of project_back():
                # meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                #                                     use_meta_texture=True, render_cache=render_cache)
                # current_z_normals = meta_outputs['image']
                # current_z_mask = meta_outputs['mask'].flatten()
                # masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                #                         current_z_mask == 1][:, :1]
                # masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                #                         current_z_mask == 1][:, :1]
                # loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()         
                                
                    loss.backward() # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                                    # the network, that is, the pixel value of the texture atlas
                    optimizer.step()
                    pbar.set_description(f"Fitting max_z_normals -view={v}: Epoch={i + 1}, Loss: {loss.item():.4f}")
                    
                    #MJ: Render the learned meta_texture_img to produce the new  curr_max_z_normals_pred 
                    meta_output = self.mesh_model.render( background=torch.Tensor([0, 0, 0]).to(self.device),
                                                        use_meta_texture=True, render_cache=render_cache)
                    curr_max_z_normals_pred = meta_output['image'][:,0:1,:,:].clamp(0,1)  
                    #MJ: z_normals_cache is the projected meta_texture_img, which is zero at first
            
            #End for _ in tqdm(range(200), desc='fitting max_z_normals')    
           
        #End for v in range(self.thetas)    # scan over the 7 views 
                
=======

        # JA: TODO: Add num_epochs hyperparameter
        with tqdm(range(200), desc='project_back_only_texture_atla: fitting mesh colors') as pbar:
            for iter in pbar:
                optimizer.zero_grad()
                outputs = self.mesh_model.render(background=background,
                                                render_cache=render_cache)  
                #MJ: with render_cache not None => render() omits the raterization process, uses the cache of its results
                #  and only performs the texture mapping projection => take much less time than performing a full rendering 
                                                   
                rgb_render = outputs['image']
                
                # for i in range(rgb_render.shape[0]):
                #     self.log_train_image(rgb_render[i][None] * render_update_mask[i][None], f'project_back: rgb-output_{i}')  
           
                #MJ: z_normals = outputs['normal'][:,-1,:,:]

                # loss = (render_update_mask * (rgb_render - rgb_output.detach()).pow(2)).mean()
                #loss = (render_update_mask * z_normals * (rgb_render - rgb_output.detach()).pow(2)).mean()
                #BY MJ:
                #MJ: Try the following three weight_masks
                #1) weight_masks computed by compare_between_views
                #2) use weight-masks based on z_normals
                # weight_masks = self.compute_view_weights( z_normals )
                #3) use weight-masks as best_z_normals
                
                #MJ: loss = (render_update_mask * weight_masks * (rgb_render - rgb_output.detach()).pow(2)).mean()
                loss = (render_update_mask * self.view_weights * (rgb_render - rgb_output.detach()).pow(2)).mean()
                loss.backward(retain_graph=True) # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                                # the network, that is, the pixel value of the texture atlas
                                #MJ:  if the loss is now dependent on some stateful or iterative operation, it might need to retain the graph.
              
                #loss.backward()
                optimizer.step()
                pbar.set_description(f"zero123plus: Fitting mesh colors -Epoch {iter + 1}, Loss: {loss.item():.7f}")

               
        #MJ:  with tqdm(range(200), desc='project_back_only_texture_atla: fitting mesh colors') as pbar
        
        return rgb_render
    #MJ:  self.project_back_max_z_normals(
        #     background=None, object_mask=self.mask, z_normals=self.normals_image[:,2:3,:,:]
        #     face_normals = self.face_normals, face_idx=self.face_idx           
        # )
    
    def project_back_max_z_normals(self):
        optimizer = torch.optim.Adam(self.mesh_model.get_params_max_z_normals(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                        eps=1e-15)
        
        
            
        for v in range( len(self.thetas) ):    # scan over the 7 views 
            #MJ: Compute the max_z_normals viewed in view v1 by comparing it with z_normals of the other views v2
    
            #MJ: Render the max_z_normals (self.meta_texure_img) which has been learned using the previous view z_normals
            # At the beginning of the for loop, self.meta_texture_img is set to 0
            
            meta_output_v = self.mesh_model.render(theta=self.thetas[v], phi=self.phis[v], radius=self.radii[v],
                                                background=torch.Tensor([0, 0, 0]).to(self.device),
                                                        use_meta_texture=True, render_cache=None)
            render_cache_v =  meta_output_v['render_cache']
            curr_max_z_normals_pred_v = meta_output_v['image'][:,0:1,:,:]   #MJ: meta_output['image'] is the projected meta_texture_img
            
            z_normals_v = meta_output_v['normals'][:,2:3,:,:]   
        
            #MJ: z_normals is the z component of the normal vectors of the faces seen by each view
            z_normals_mask_v = meta_output_v['mask']   #MJ: shape = (1,1,1200,1200)
            
            #MJ: Try blurring the object-mask "curr_z_mask" with Gaussian blurring: 
            # The following code is a simply  cut and paste from project-back:
            object_mask_v = z_normals_mask_v
            #MJ: erode the boundary of the mask
            object_mask_v = torch.from_numpy( cv2.erode(object_mask_v[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8)) ).to(
                                 object_mask_v.device).unsqueeze(0).unsqueeze(0)
                                 
            # object_mask = torch.from_numpy(
            #     cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            #     object_mask.device).unsqueeze(0).unsqueeze(0)
            # render_update_mask = object_mask.clone()
            render_update_mask_v =  object_mask_v.clone()

            #MJ: dilate the bounary of the mask
            blurred_render_update_mask_v = torch.from_numpy(
                 cv2.dilate(render_update_mask_v[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
                 render_update_mask_v.device).unsqueeze(0).unsqueeze(0)
                
          
            blurred_render_update_mask_v = utils.gaussian_blur(blurred_render_update_mask_v, 21, 16)

            # Do not get out of the object
            blurred_render_update_mask_v[object_mask_v == 0] = 0
                         
            #MJ: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True),
            # rather than torch.tensor(sourceTensor).        
            # max_z_normals_target = torch.tensor(curr_max_z_normals_pred)   #MJ: max_z_normals_target will refer to a copy of max_z_normals_pred
            max_z_normals_target_v = curr_max_z_normals_pred_v.clone().detach() 
            #MJ: Compare the curr max_z_normals and the curr_z_normals and update  max_z_normals_target
            max_z_normals_target_v[0, 0, :, :] = torch.max(max_z_normals_target_v[0, 0, :, :], z_normals_v[0, 0, :, :])    
            
            self.log_train_image(torch.cat([max_z_normals_target_v, max_z_normals_target_v,max_z_normals_target_v], dim=1),
                                f'project_back_max_z_normals:max_z_normals_target_v')
                       
            curr_max_z_normals_pred_v = curr_max_z_normals_pred_v *    blurred_render_update_mask_v.float()    
            max_z_normals_target_v = max_z_normals_target_v *    blurred_render_update_mask_v.float()    
            #MJ: z_normals: (1,1,1200,1200)                 
                               
            with  tqdm(range(300), desc='project_back_max_z_normals:fitting max_z_normals') as pbar:
                for iter in pbar:
                    optimizer.zero_grad()                   
                                        
                    # loss = 0
                    # for i in range( masked_max_z_normals_pred.shape[0]):
                    #    # Compute the mean squared error for each batch element separately                          
                    #     view_loss =  (masked_max_z_normals_pred[i] - masked_curr_z_normals_target[i].detach()).pow(2).mean()      
                    
                    #     loss += view_loss #Accumulate the individual loss for each viewpoint
                                
                    # loss /=  masked_max_z_normals_pred.shape[0]
                    
                    loss_v = ( curr_max_z_normals_pred_v -   max_z_normals_target_v.detach()).pow(2).mean()      
                              
                                
                    loss_v.backward() # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                                    # the network, that is, the pixel value of the texture atlas
                    optimizer.step()
                    pbar.set_description(f"project_back_max_z_normals:Fitting max_z_normals -view={v}: Epoch={iter + 1}, Loss: {loss_v.item():.7f}")
                    
                    #MJ: Render the learned meta_texture_img to produce the new  curr_max_z_normals_pred 
                    meta_output_v = self.mesh_model.render( background=torch.Tensor([0, 0, 0]).to(self.device),
                                                        use_meta_texture=True, render_cache=render_cache_v)
                    
                    curr_max_z_normals_pred_v = meta_output_v['image'][:,0:1,:,:]   #MJ: meta_output['image'] is the projected meta_texture_img
            
                    curr_max_z_normals_pred_v = curr_max_z_normals_pred_v *    blurred_render_update_mask_v.float()    
                             
            #End for _ in tqdm(range(300), desc='fitting max_z_normals')    
        #End for v in range(self.thetas)    # scan over the 7 views 
  
                
         
    def compute_view_weights(self, z_normals, max_z_normals, alpha=-10.0 ):        
        
        """
        Compute view weights where the weight increases exponentially as z_normals approach max_z_normals.
        
        Args:
            z_normals (torch.Tensor): The tensor containing the z_normals data.
            max_z_normals (torch.Tensor): The tensor containing the max_z_normals data.
            alpha (float): A scaling parameter that controls how sharply the weight increases (should be negative).
        
        Returns:
            torch.Tensor: The computed weights with the same shape as the input tensors (B, 1, H, W).
        """
        # Ensure inputs have the same shape
        assert z_normals.shape == max_z_normals.shape, "Input tensors must have the same shape"
        
        # Compute the difference between max_z_normals and z_normals
        delta = max_z_normals - z_normals
        for i in range( delta.shape[0]):
            print(f'min delta for view-{i}:{delta[i].min()}')
            print(f'max  delta for view-{i}:{delta[i].max()}')
        #MJ: delta is supposed to be greater than 0; But sometimes, z_normals is greater than max_z_normals.
        # It means that project_back_max_z_normals() was not fully successful.
        abs_delta = torch.abs( delta )
        # Calculate the weights using an exponential function, multiplying by negative alpha
        weights = torch.exp(alpha * abs_delta)  #MJ: the max value of torch.exp(alpha * delta)   will be torch.exp(alpha * 0) = 1 
        
        for i in range( weights.shape[0]):
            print(f'min weights  for view-{i}:{weights[i].min()}')
            print(f'max  weights for view-{i}:{weights[i].max()}')
        # Normalize to have the desired shape (B, 1, H, W)
        #weights = weights.view(weights.size(0), 1, weights.size(1), weights.size(2))
        
        return weights
       
    
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6
    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
        if self.cfg.log.log_images:
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
            else: #MJ: for debugging
                tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
<<<<<<< HEAD
               # tensor = einops.rearrange(tensor, 'c h w -> h w c').detach().cpu().numpy()
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                self.train_renders_path / f'{name}.jpg')    #MJ:    self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
=======
            
            if np.any(np.isnan(tensor)) or np.any(np.isinf(tensor)):
    #     # Raise an exception if there are any NaNs or infinite values
    #      tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
    #      Image.fromarray( (tensor * 255).astype(np.uint8) ).save('experiments'/f'debug:NanOrInf.jpg')

                raise ValueError("Tensor contains NaNs or infinite values")
            
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                self.train_renders_path / f'debug:{name}.jpg')
>>>>>>> 177114c8972206cca551eab0ae6156e378e1f8e6

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image]):
        if len(intermediate_vis) > 0:
            #MJ step_folder = self.train_renders_path / f'{self.paint_step:04d}_diffusion_steps'
            step_folder = self.train_renders_path / f'diffusion_steps'
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(
                    step_folder / f'{k:02d}_diffusion_step.jpg')

    def save_image(self, tensor: torch.Tensor, path: Path):
        if self.cfg.log.log_images:
            Image.fromarray(
                (einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy() * 255).astype(np.uint8)).save(
                path)

