import kaolin as kal
import torch
import numpy as np
from loguru import logger
class Renderer:
    # from https://github.com/threedle/text2mesh

    def __init__(self, device, dim=(224, 224), interpolation_mode='nearest', fovyangle=np.pi / 3):
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'

        camera = kal.render.camera.generate_perspective_projection(fovyangle).to(device) # JA: This is the field of view
                                                                                        # It is currently set to 60deg.

        self.device = device
        self.interpolation_mode = interpolation_mode
        self.camera_projection = camera
        self.dim = dim
        self.background = torch.ones(dim).to(device).float()

    @staticmethod
    def get_camera_from_view(elev, azim, r=3.0, look_at_height=0.0):
        x = r * torch.sin(elev) * torch.sin(azim)
        y = r * torch.cos(elev)
        z = r * torch.sin(elev) * torch.cos(azim)

        pos = torch.tensor([x, y, z]).unsqueeze(0)
        #MJ: torch.tensor teis to interpret each item in the list as a scalar value;
        # To combine multiple tensors into a single tensor along a new dimension, 
        # you should use either torch.stack or torch.cat depending on the desired shape:
        # torch.stack: Combines tensors along a new dimension (e.g., converts three 1D tensors into a single 2D tensor).
        # torch.cat: Concatenates tensors along an existing dimension (e.g., concatenates two 2D tensors into a larger 2D tensor).
        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height
        camera_up_direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

        camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, camera_up_direction)
        return camera_proj
    
    @staticmethod
    def get_camera_from_multiple_view(elev, azim, r, look_at_height=0.0):
        x = r * torch.sin(elev) * torch.sin(azim)
        y = r * torch.cos(elev)
        z = r * torch.sin(elev) * torch.cos(azim)

        pos = torch.stack([x, y, z], dim=1)
        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height
        camera_up_direction = torch.ones_like(pos) * torch.tensor([0.0, 1.0, 0.0]).to(pos.device)

        camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, camera_up_direction)
        return camera_proj


    def normalize_depth(self, depth_map):
        assert depth_map.max() <= 0.0, 'depth map should be negative' # JA: In the standard computer graphics, the camera view direction is the negative z axis
        object_mask = depth_map != 0 # JA: The region where the depth map is not 0 means that it is the object region
        # depth_map[object_mask] = (depth_map[object_mask] - depth_map[object_mask].min()) / (
        #             depth_map[object_mask].max() - depth_map[object_mask].min())
        # depth_map = depth_map ** 4
        min_val = 0.5
        depth_map[object_mask] = ((1 - min_val) * (depth_map[object_mask] - depth_map[object_mask].min()) / (
                depth_map[object_mask].max() - depth_map[object_mask].min())) + min_val
        # depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # depth_map[depth_map == 1] = 0 # Background gets largest value, set to 0

        return depth_map

    def normalize_multiple_depth(self, depth_maps):
        assert (depth_maps.amax(dim=(1, 2)) <= 0).all(), 'depth map should be negative'
        object_mask = depth_maps != 0  # Mask for non-background pixels

        # To handle operations for masked regions, we need to use masked operations
        # Set default min and max values to avoid affecting the normalization
        masked_depth_maps = torch.where(object_mask, depth_maps, torch.tensor(float('inf')).to(depth_maps.device))
        min_depth = masked_depth_maps.amin(dim=(1, 2), keepdim=True)[0]

        masked_depth_maps = torch.where(object_mask, depth_maps, torch.tensor(-float('inf')).to(depth_maps.device))
        max_depth = masked_depth_maps.amax(dim=(1, 2), keepdim=True)[0]

        # Replace 'inf' with zeros in cases where no valid object pixels are found
        min_depth[min_depth == float('inf')] = 0
        max_depth[max_depth == -float('inf')] = 0

        range_depth = max_depth - min_depth
        # Prevent division by zero
        range_depth[range_depth == 0] = 1

        # Calculate normalized depth maps
        min_val = 0.5
        normalized_depth_maps = torch.where(
            object_mask,
            ((1 - min_val) * (depth_maps - min_depth) / range_depth) + min_val,
            torch.zeros_like(depth_maps)
        )

        return normalized_depth_maps

    def render_single_view(self, mesh, face_attributes, elev=0, azim=0, radius=2, look_at_height=0.0,calc_depth=True,dims=None, background_type='none'):
        dims = self.dim if dims is None else dims

        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(self.device), mesh.faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        if calc_depth:
            depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:])
            depth_map = self.normalize_depth(depth_map)
        else:
            depth_map = torch.zeros(1,64,64,1)

        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_attributes)

        mask = (face_idx > -1).float()[..., None]
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        if background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), depth_map.permute(0, 3, 1, 2)

    def render_multiple_view(self, mesh, face_attributes, elev, azim, radius, look_at_height=0.0,calc_depth=True,dims=None, background_type='none'):
        dims = self.dim if dims is None else dims

        camera_transform = self.get_camera_from_view(elev, azim, r=radius,
                                                look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(self.device), mesh.faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        if calc_depth:
            depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:])
            depth_map = self.normalize_multiple_depth(depth_map)
        else:
            depth_map = torch.zeros(1,64,64,1)

        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_attributes)

        mask = (face_idx > -1).float()[..., None]
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        if background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), depth_map.permute(0, 3, 1, 2)


    def render_single_view_texture(self, verts, faces, uv_face_attr, texture_map, elev=0, azim=0, radius=2,
                                   look_at_height=0.0, dims=None, background_type='none', render_cache=None):
        dims = self.dim if dims is None else dims

        if render_cache is None:

            camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                    look_at_height=look_at_height).to(self.device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)
            # JA: face_vertices_camera[:, :, :, -1] likely refers to the z-component (depth component) of these coordinates, used both for depth mapping and for determining how textures map onto the surfaces during UV feature generation.
            depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:]) 
            depth_map = self.normalize_depth(depth_map)

            uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, uv_face_attr)
            uv_features = uv_features.detach()

        else:
            # logger.info('Using render cache')
            face_normals, uv_features, face_idx, depth_map = render_cache['face_normals'], render_cache['uv_features'], render_cache['face_idx'], render_cache['depth_map']
        mask = (face_idx > -1).float()[..., None]

        image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode)
                        # JA: Interpolates texture_maps by dense or sparse texture_coordinates (uv_features).
                        # This function supports sampling texture coordinates for:
                        # 1. An entire 2D image
                        # 2. A sparse point cloud of texture coordinates.

        image_features = image_features * mask # JA: image_features refers to the render image
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        elif background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        #MJ: normals_image = face_normals[0][face_idx, :] => it should be changed as follows:
        
        #MJ: In the line normals_image = face_normals[0][face_idx, :], if face_idx contains negative values and is used directly, 
        # it will lead to unintended behavior.
        # The negative index would attempt to access an element from the end of face_normals, 
        # potentially retrieving incorrect normal data or leading to an error if not handled correctly.
        # Prepare normals for only valid indices
        valid_indices_mask = (face_idx >= 0) #MJ: face_idx: shape =(7,1200,1200,1)
        normals_image = torch.zeros_like(face_idx).repeat(1, 1,1,3)  # The shape change:  (7,1200,1200,1) => (7,1200,1200,3)
        normals_image[valid_indices_mask] = face_normals[0][face_idx[valid_indices_mask], :]

        render_cache = {'uv_features':uv_features, 'face_normals':face_normals,'face_idx':face_idx, 'depth_map':depth_map}

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2),\
               depth_map.permute(0, 3, 1, 2), normals_image.permute(0, 3, 1, 2), render_cache

    def render_multiple_view_texture(self, verts, faces, uv_face_attr, texture_map, elev, azim, radius,
                                   look_at_height=0.0, dims=None, background_type='none', render_cache=None):
        dims = self.dim if dims is None else dims

        if render_cache is None:

            camera_transform = self.get_camera_from_multiple_view(
                elev, azim, r=radius,
                look_at_height=look_at_height
            )
            # JA: Since the function prepare_vertices is specifically designed to move and project vertices to camera space and then index them with faces, the face normals returned by this function are also relative to the camera coordinate system. This follows from the context provided by the documentation, where the operations involve transforming vertices into camera coordinates, suggesting that the normals are computed in relation to these transformed vertices and thus would also be in camera coordinates.
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                verts, faces, self.camera_projection, camera_transform=camera_transform)
            # JA: face_vertices_camera[:, :, :, -1] likely refers to the z-component (depth component) of these coordinates, used both for depth mapping and for determining how textures map onto the surfaces during UV feature generation.
            depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:]) 
            depth_map = self.normalize_multiple_depth(depth_map)  #MJ: face_vertices_camera[:, :, :, -1:].shape=torch.Size([7, 98998, 3, 1])

            uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, uv_face_attr)  #MJ: uv_face_attr.shape: torch.Size([7, 98998, 3, 2])
            #MJ:  face_vertices_z =  face_vertices_camera[:, :, :, -1],
             #    face_vertices_image:  2D positions of the face vertices on image plane,
            #    of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 2)`,
            #   Features (per-vertex per-face) to be drawn = uv_face_attr: of shape :math:`(\text{batch_size}, \text{num_faces}, 3, \text{feature_dim})`
            uv_features = uv_features.detach()
            #MJ: The uv_features tensor, having a shape of (7, 1200, 1200, 2), represents the UV coordinates 
            # for texture mapping over the mesh surfaces in the 3D scene. 
            # The fourth dimension (2) contains pairs of UV coordinates for each pixel

        else:
            # logger.info('Using render cache')
            face_normals, uv_features, face_idx, depth_map = render_cache['face_normals'], render_cache['uv_features'], render_cache['face_idx'], render_cache['depth_map']
       
        mask = (face_idx > -1).float()[..., None]
        #MJ: Each projected point (or pixel) on the 2D image is linked back to a point on the 3D surface, 
        # which has a specific UV coordinate pair (U, V).
        # uv_features: This tensor contains the UV coordinates (U, V) for each pixel in the projected 2D image.
        #The function reads the color values from the texture_map based on these coordinates 
        # and assigns them to the pixels in the projected 2D image.
        image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode)
                        # JA: Interpolates texture_maps by dense or sparse texture_coordinates (uv_features).
                        # This function supports sampling texture coordinates for:
                        # 1. An entire 2D image: texture_map: (7,3,1024,1024) => image_features.shape: torch.Size([7, 1200, 1200, 3])
                        # 2. A sparse point cloud of texture coordinates.

        image_features = image_features * mask # JA: image_features refers to the render image
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        elif background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        # JA: face_normals[0].shape:[14232, 3], face_idx.shape: [1, 1024, 1024]
        # normals_image = face_normals[0][face_idx, :] # JA: normals_image: [1024, 1024, 3]
        # Generate batch indices
        batch_size = face_normals.shape[0]
        batch_indices = torch.arange(batch_size).view(-1, 1, 1)

        # Expand batch_indices to match the dimensions of face_idx
        batch_indices = batch_indices.expand(-1, *face_idx.shape[1:])

        # Use advanced indexing to gather the results
        normals_image = face_normals[batch_indices, face_idx]

        render_cache = {'uv_features':uv_features, 'face_normals':face_normals,'face_idx':face_idx, 'depth_map':depth_map}

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2),\
               depth_map.permute(0, 3, 1, 2), normals_image.permute(0, 3, 1, 2), render_cache

    def project_uv_single_view(self, verts, faces, uv_face_attr, elev=0, azim=0, radius=2,
                               look_at_height=0.0, dims=None, background_type='none'):
        # project the vertices and interpolate the uv coordinates

        dims = self.dim if dims is None else dims

        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                     look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                          face_vertices_image, uv_face_attr)
        return face_vertices_image, face_vertices_camera, uv_features, face_idx

    def project_single_view(self, verts, faces, elev=0, azim=0, radius=2,
                               look_at_height=0.0):
        # only project the vertices
        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                     look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        return face_vertices_image
