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

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.stable_diffusion_depth import StableDiffusion
from src.training.views_dataset import ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy


class TEXTure:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.paint_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)  #MJ: self.cfg.optim.seed = 0

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
        self.diffusion = self.init_diffusion() #MJ: initialize the vae, the unets for depth pipeline, inpainting pipeline, and the zero123 unet, the single scheduler for all the unets
        self.text_z, self.text_string = self.calc_text_embeddings()
        self.dataloaders = self.init_dataloaders()
        self.back_im = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0,
                                 1) / 255.0

        self.zero123_cond_image = None

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_mesh_model(self) -> nn.Module:
        cache_path = Path('cache') / Path(self.cfg.guide.shape_path).stem
        cache_path.mkdir(parents=True, exist_ok=True)
        model = TexturedMeshModel(self.cfg.guide, device=self.device,
                                  render_grid_size=self.cfg.render.train_grid_size,
                                  cache_path=cache_path,
                                  texture_resolution=self.cfg.guide.texture_resolution,
                                  augmentations=False)

        model = model.to(self.device)
        logger.info(
            f'Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_diffusion(self) -> Any:
        # JA: The StableDiffusion class composes a pipeline by using individual components such as VAE encoder,
        # CLIP encoder, and UNet
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
        init_train_dataloader = MultiviewDataset(self.cfg.render, device=self.device).dataloader()
        #MJ: Instead, an object is considered an iterator if it implements two methods: __iter__() and __next__(). The __iter__() method returns the iterator object itself and is called when an iterator is required, such as at the start of loops. The __next__() method returns the next item from the sequence and is called at each loop iteration. When there are no more items to return, __next__() should raise a StopIteration exception to signal that the iteration is complete.
        val_loader = ViewsDataset(self.cfg.render, device=self.device,
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device,
                                        size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': init_train_dataloader, 'val': val_loader,
                       'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def paint(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        pbar = tqdm(total=len(self.dataloaders['train']), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        strength = 1.0
        num_inference_steps = 50

        self.diffusion.scheduler.set_timesteps(num_inference_steps)
        timesteps, num_inference_steps = self.diffusion.get_timesteps(num_inference_steps, strength)

        for i, t in tqdm(enumerate(timesteps)): #MJ: timesteps=[981,961,..., 21,1]
            # JA: The following loop computes the texture atlas for the given mesh using ten render images. In other words,
            # it is the inverse rendering process. Each of the ten views is one of the six view images.
            #MJ:
            pbar.reset()  # Reuse the same progress bar, which ranges over the data items in dataloaders
            #self.paint_step += 1
            for data in self.dataloaders['train']: #MJ: data = {'dir': tensor([0]), 'theta': 1.0471975803375244, 'phi': 0.0, 'radius': 1.5, 'base_theta': 1.0471975511965976}
                self.paint_step += 1
                pbar.update(1) #MJ: Out of timesteps units, increment by 1 unit
                self.paint_viewpoint(data, i, t, timesteps) 
                # MJ: The same i and t used for the first viewpoint scheduler is used for the second viewpoint scheduler, without incrementing i and t; This breaks the scheduler logic; 
                # To solve this problem, we need to denoise all the view images simultaneously. This can be done by creating a random tensor whose batch dim is
                # # the number of viewpoints.
               
                self.evaluate(self.dataloaders['val'], self.eval_renders_path)  # JA: This is the validation step for the current
                                                                                # training step
                self.mesh_model.train() # JA: Set the model to train mode because the self.evaluate sets the model to eval mode.

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):
            preds, textures, depths, normals = self.eval_render(data)

            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(save_path / f"step_{self.paint_step:05d}_{i:04d}_rgb.jpg")
                Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                    save_path / f'{self.paint_step:04d}_{i:04d}_normals_cache.jpg')
                if self.paint_step == 0:
                    # Also save depths for debugging
                    torch.save(depths[0], save_path / f"{i:04d}_depth.pt")

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        Image.fromarray(texture).save(save_path / f"step_{self.paint_step:05d}_texture.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"step_{self.paint_step:05d}_{name}.mp4", video,
                                                           fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'rgb')
        logger.info('Done!')

    def full_eval(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = self.final_renders_path
        self.evaluate(self.dataloaders['val_large'], output_dir, save_as_video=True)
        # except:
        #     logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\tDone!")

    # JA: paint_viewpoint computes a portion of the texture atlas for the given viewpoint
    def paint_viewpoint(self, data: Dict[str, Any], i, t, timesteps):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius'] # JA: data represents a viewpoint which is stored in the dataset
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

        # Set background image: MJ: background is not None in our experiment
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
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background) #MJ: with render_cache = None; This is the only case where render_cache = None is used
        render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)
        rgb_render_raw = outputs['image']  # MJ: image: pred_map = pred_back * (1 - mask) + pred_features * mask, where pred_features = foreground
        depth_render = outputs['depth']
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(background=background,
                                         render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render = outputs['image'] #MJ: torch.Size([1, 3, 1200, 1200])
        # Render meta texture map: MJ: the meta texture map stores the z-normals of the triangles of the mesh, which represents the visibility of each triangle of the mesh
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=render_cache)

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)  #MJ:z_normals: torch.Size([1, 1, 1200, 1200]); outputs['normals']: (1,3,1200,1200)
        z_normals_cache = meta_output['image'].clamp(0, 1)  #MJ: torch.Size([1, 3, 1200, 1200]); = pred_map = the combination of the foreground (meta_texture_img) and the background
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2] #MJ: It has a shape of (1, 1, 1200, 1200), it contains the 2nd channel of the meta_texture_img

        self.log_train_image(rgb_render, 'rendered_input')
        self.log_train_image(depth_render[0, 0], 'depth', colormap=True)
        self.log_train_image(z_normals[0, 0], 'z_normals', colormap=True)
        self.log_train_image(z_normals_cache[0, 0], 'z_normals_cache', colormap=True) #MJ: z_normals_cache[0, 0]: the 1st channel of    z_normals_cache = meta_output['image'] = the z-normal of each face 

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]; dirs: Size([1])
            text_z = self.text_z[dirs] # JA: dirs is one of the six directions. text_z is the embedding vector of the specific view prompt
            text_string = self.text_string[dirs]
        else:
            text_z = self.text_z
            text_string = self.text_string
        logger.info(f'text: {text_string}')

        # JA: Create trimap of keep, refine, and generate using the render output:  #MJ: update_mask is a more "refined" mask than refine_mask (derived from refine_mask)
        update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw, #MJ:   rgb_render_raw = outputs['image']  # MJ: image: pred_map = pred_back * (1 - mask) + pred_features * mask
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache, #MJ: The meta_texture_img: z_normals_cache = meta_output['image'].clamp(0, 1)
                                                                        edited_mask=edited_mask, #MJ: The 2nd channel of the meta_texture_img: edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]
                                                                        mask=outputs['mask'])  #MJ: mask with 1 refers to the object area

        update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
        if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
            logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
            return

        self.log_train_image(rgb_render * (1 - update_mask), name='masked_input')
        self.log_train_image(rgb_render * refine_mask, name='refine_regions')

        # Crop to inner region based on object mask
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_rgb_render = crop(rgb_render) # JA: This is rendered image which is denoted as Q_0.
                                              # In our experiment, 1200 is cropped to 827
        cropped_depth_render = crop(depth_render)
        cropped_update_mask = crop(update_mask)
        self.log_train_image(cropped_rgb_render, name='cropped_input')
        self.log_train_image(cropped_depth_render.repeat_interleave(3, dim=1), name='cropped_depth')

        checker_mask = None
        if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
            # JA: generate_checkerboard is defined in formula 2 of the paper
            checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask),
                                                      crop(generate_mask))
            self.log_train_image(F.interpolate(cropped_rgb_render, (512, 512)) * (1 - checker_mask),
                                 'checkerboard_input')
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1

        # JA: self.zero123_cond_image has been added for Zero123 integration
        if self.view_dirs[dirs] == "front":
            resized_zero123_cond_image = None
        else:
            assert self.zero123_cond_image is not None
            resized_zero123_cond_image = F.interpolate(
                self.zero123_cond_image,
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
        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, cropped_rgb_render.detach(), # JA: cropped_rgb_render is Q_0
                                                                    cropped_depth_render.detach(),
                                                                    i=i, t=t, timesteps=timesteps,
                                                                    guidance_scale=self.cfg.guide.guidance_scale,
                                                                    # strength=1.0, update_mask=cropped_update_mask,
                                                                    update_mask=cropped_update_mask,
                                                                    #MJ: fixed_seed=self.cfg.optim.seed, do not use the fixed seed
                                                                    check_mask=checker_mask,
                                                                    intermediate_vis=self.cfg.log.vis_diffusion_steps,

                                                                    # JA: The following were added to use the view image
                                                                    # created by Zero123
                                                                    view_dir=self.view_dirs[dirs], # JA: view_dir = "left", this is used to check if the view direction is front
                                                                    front_image=resized_zero123_cond_image,
                                                                    phi=data['phi'],
                                                                    theta=data['base_theta'] - data['theta'], #MJ: data['base_theta']=59.99999999999
                                                                    condition_guidance_scales=condition_guidance_scales)

        self.log_train_image(cropped_rgb_output, name='direct_output')
        self.log_diffusion_steps(steps_vis)
        # JA: cropped_rgb_output always has a shape of (512, 512); recover the resolution of the nonzero rendered image (e.g. (827, 827))
        cropped_rgb_output = F.interpolate(cropped_rgb_output, 
                                           (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
                                           mode='bilinear', align_corners=False)

        # Extend rgb_output to full image size
        rgb_output = rgb_render.clone() # JA: rgb_render shape is 1200x1200
        rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output # JA: For example, (189, 1016, 68, 895) refers to the nonzero region of the render image
        self.log_train_image(rgb_output, name='full_output')

        # Project back
        object_mask = outputs['mask'] # JA: mask has a shape of 1200x1200
        # JA: Compute a part of the texture atlas corresponding to the target render image of the specific viewpoint
        fitted_pred_rgb, _ = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                               object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                               z_normals_cache=z_normals_cache)
        self.log_train_image(fitted_pred_rgb, name='fitted')

        if self.view_dirs[dirs] == "front":
            # JA: Zero123 needs the input image without the background
            # rgb_output is the generated and uncropped image in pixel space
            self.zero123_cond_image = crop(
                rgb_output * object_mask
                + torch.ones_like(rgb_output, device=self.device) * (1 - object_mask)
            )   # JA: In the case of front view, the shape is (930,930).
                # This rendered image will be compressed to the shape of (512, 512) which is the shape of the diffusion
                # model.

            self.log_train_image(self.zero123_cond_image, name='zero123_input')

        return

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
        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals,
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
        diff = ( rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device) ).abs().sum(axis=1)  #MJ:self.mesh_model.default_color= list [0.8, 0.1, 0.8] = magenta color
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0) #MJ: diff: (1,1200,1200);      exact_generate_mask: (1,1,1200,1200)
        #MJ: If the rendered image region is close to the default color, the magenta,  the region is newly created without using the texture atlas being learned (the initial color of the texture atlas is magenta)=> So it should be generated
        # Extend mask
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

        update_mask = generate_mask.clone()  #MJ: initialize the update_mask to the generate_mask; it may be changed

        object_mask = torch.ones_like(update_mask) #MJ: initialize  all pixels of object_mask to one, indicating the object area
        object_mask[depth_render == 0] = 0 #MJ: But Set the pixels  of the object_mask to zero, where depth_render is 0, that is the background region
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        # Generate the refine mask based on the z normals, and the edited mask (the image of the z-normals (meta_texture_img's G channel))

        refine_mask = torch.zeros_like(update_mask) #MJ: Initialize  all pixels of refine_mask  to zero (not to be refined); The pixels whose z_normals are greater than the cached ones from the previous view plus the threshold WILL BE refined
        refine_mask[ z_normals > z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr ] = 1 #MJ:  z_normals_cache[:, :1, :, :] = the z-normal of the faces= the 1st channel of meta_texture; self.cfg.guide.z_update_thr=0.2 (The normal vectors are unit vectors)
        if self.cfg.guide.initial_texture is None: #MJ: This is the case in our experiment
            refine_mask[ z_normals_cache[:, :1, :, :] == 0 ] = 0 #MJ: What does the 1nd channel of z_normals_cache, z_normals_cache[:, :1, :, :], represent? => The z-normals of the faces
        elif self.cfg.guide.reference_texture is not None:
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            # Don't use bad angles here
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
            
        #MJ: With generate_mask and refine_mask defined, 
        # Update update_mask which was defined to be  update_mask = generate_mask.clone(); Some regions where two conditions are satisfied
        # will be updated as follows:    
        update_mask[refine_mask == 1] = 1 #MJ: Initialize the update_mask to 1 where refine_mask is 1; So the region to be refined is a candidate of the region to be updated

        update_mask[ torch.bitwise_and( object_mask == 0, generate_mask == 0) ] = 0 #MJ: Set update_mask to 0 where is the background region (object_mask=0) and   generate_mask is 0 (keep region)

        # Visualize trimap
        if self.cfg.log.log_images:
            trimap_vis = utils.color_with_shade(color=[112 / 255.0, 173 / 255.0, 71 / 255.0], z_normals=z_normals)
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1
            trimap_vis = trimap_vis * (1 - exact_generate_mask) + utils.color_with_shade(
                [255 / 255.0, 22 / 255.0, 67 / 255.0],
                z_normals=z_normals,
                light_coef=0.7) * exact_generate_mask

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
            self.log_train_image(shaded_rgb_vis, 'shaded_input')
            self.log_train_image(trimap_vis, 'trimap')

        return update_mask, generate_mask, refine_mask #MJ: update_mask is a more "refined" mask than refine_mask (derived from refine_mask)
    #MJ: checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask),crop(generate_mask))
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

    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor):
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)
        render_update_mask = object_mask.clone() #Initialize render_update_mask (from trimap) to the foreground object mask

        render_update_mask[update_mask == 0] = 0 #MJ: Do not update the rendering where update_mask is zero, that is, keep-region

        blurred_render_update_mask = torch.from_numpy(
            cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
            render_update_mask.device).unsqueeze(0).unsqueeze(0)
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0

        if self.cfg.guide.strict_projection: #MJ: This is the case
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
            # Do not use bad normals: z_normals: torch.Size([1, 1, 1200, 1200])
            z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
            blurred_render_update_mask[z_was_better] = 0

        render_update_mask = blurred_render_update_mask  #MJ: Set render_update_mask to blurred_render_update_mask
        self.log_train_image(rgb_output * render_update_mask, 'project_back_input') #MJ: rgb_output=cropped_rgb_output

        # Update the z_normals_cache to the maximum from the current z_normals and  z_normals_cache
        z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])

        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)  #MJ: self.mesh_model.get_params()= [self.background_sphere_colors, self.texture_img, self.meta_texture_img] 
            
        # JA: Create the texture atlas for the mesh using each view. It updates the parameters
        # of the neural network and the parameters are the pixel values of the texture atlas.
        # The loss function of the neural network is the render loss. The loss is the difference
        # between the specific image and the rendered image, rendered using the current estimate
        # of the texture atlas.
        for _ in tqdm(range(200), desc='fitting mesh colors'): # JA: We use 200 epochs
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache) #MJ: Render the mesh using the current texture_img, meta_texture_img which is being learned
            rgb_render = outputs['image']

            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]  #MJ: "masked" regions are the real meaningful regions
            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean() #MJ: define the rendering loss

            meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                  use_meta_texture=True, render_cache=render_cache)
            current_z_normals = meta_outputs['image'] #MJ: torch.Size([1, 3, 1200, 1200])
            current_z_mask = meta_outputs['mask'].flatten() #MJ: current_z_mask: torch.Size([1440000]); 1,440,000 = 1200 x 1200
            masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                       current_z_mask == 1][:, :1]
            masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                    current_z_mask == 1][:, :1]  #MJ:  current_z_normals.reshape(1, current_z_normals.shape[1], -1).shape=(1,3,1440000);  z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,                                    current_z_mask == 1]: torch.Size([1, 3, 182750]);  masked_last_z_normals: torch.Size([1, 1, 182750])
            loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean() #MJ:z_normals_cache: updated at the beginning of project_back func; the loss for the z_normals
            loss.backward() # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                            # the network, that is, the pixel value of the texture atlas
            optimizer.step()

        return rgb_render, current_z_normals
    
    #MJ: 
    #Q:  When computing the gradient vector of loss in loss.backward(), when happens when the tree of tensors in loss do not contain some parameters in self.mesh_model.get_params()?
    #CHATGPT:

#   If a parameter tensor has requires_grad=True and is included in the computation graph but is not involved in the computation of the loss,
#   its gradients will still be computed during the backward pass. However, these gradients will be zero because they are not contributing to the loss.

# So, to amend the previous explanation: Gradients are computed for all tensors with requires_grad=True, regardless of whether they are directly 
# involved in the computation of the loss. However, if a parameter tensor is not contributing to the loss, its gradient will be zero during
# backpropagation.

# In summary, unused parameter tensors with requires_grad=True will still have their gradients computed during backpropagation,
# but these gradients will be zero if the parameters are not contributing to the loss. This behavior is consistent with 
# PyTorch's automatic differentiation mechanism.

# You're correct. Computing gradients for unused parameters can indeed be inefficient, especially if those parameters are not 
# contributing to the loss and are not being updated during optimization. In such cases, computing gradients for these parameters
# only adds unnecessary computational 
# overhead and consumes additional memory.

# To address this inefficiency, one common approach is to detach or exclude the unused parameters from the computation graph during
# the forward pass, effectively preventing gradients from flowing through them during backpropagation. 
# This can be achieved by setting requires_grad=False for the unused parameters or by detaching them from the computational graph explicitly.

# By excluding unused parameters from the computation graph, you can reduce unnecessary computations and memory usage, 
# leading to more efficient training. It's good practice to review the computational graph and ensure that only the relevant parameters 
# are included in the gradient computation to optimize training efficiency.

# But, I can achieve the same purpose by changing the function  self.mesh_model.get_params()
# ChatGPT
# Yes, you're absolutely correct. Adjusting the implementation of self.mesh_model.get_params() is indeed another viable approach to efficiently 
# handle the computation of gradients for only the relevant parameters.

# By modifying self.mesh_model.get_params() to return only the parameters that are directly involved in the computation of the loss,
# you ensure that gradients are computed only for those parameters during backpropagation. This approach effectively prevents
# unnecessary computations and reduces memory usage, leading to more efficient training.

# Therefore, by refining the logic within self.mesh_model.get_params(), you can achieve the same purpose of excluding unused
# parameters from gradient computation without needing to modify the rest of the training code. This approach offers flexibility 
# and allows for fine-grained control over which parameters are included in the optimization process, optimizing both computational efficiency 
# and training effectiveness.



    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
        if self.cfg.log.log_images:
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
            else:
                tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                self.train_renders_path / f'{self.paint_step:04d}_{name}.jpg')

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image]):
        if len(intermediate_vis) > 0:
            step_folder = self.train_renders_path / f'{self.paint_step:04d}_diffusion_steps'
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(
                    step_folder / f'{k:02d}_diffusion_step.jpg')

    def save_image(self, tensor: torch.Tensor, path: Path):
        if self.cfg.log.log_images:
            Image.fromarray(
                (einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy() * 255).astype(np.uint8)).save(
                path)
