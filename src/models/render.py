import kaolin as kal
import torch
import numpy as np
from loguru import logger
class Renderer:
    # from https://github.com/threedle/text2mesh

    def __init__(self, device, dim=(224, 224), interpolation_mode='nearest'):
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'

        camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)  #MJ: 60 or 30?

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
        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height
        direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

        camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
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

    #MJ: self.mesh.vertices = verts; self.mesh.faces =  faces;  self.face_attributes = uv_face_attr; texture_map is either texture_img or meta_texture_img
    def render_single_view_texture(self, verts, faces, uv_face_attr, texture_map, elev=0, azim=0, radius=2,
                                   look_at_height=0.0, dims=None, background_type='none', render_cache=None):
        dims = self.dim if dims is None else dims

        if render_cache is None: #MJ: render_cache is None when the mesh is really rendered; elev, azim, radius should not be None

            camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                    look_at_height=look_at_height).to(self.device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)
            # MJ: rasterize(height, width, face_vertices_z, face_vertices_image, face_features): face_features = features (per-vertex per-face) to be drawn, of shape (B,num_faces,3, feature_dim), feature_dim=3 for RGB, 2 for Texture coords, 1 for depth
            depth_map, _ = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1], #MJ: ace_vertices_z: 3D points depth (z) value of the face vertices in camera coordinate, of shape (B,num_faces,3)
                                                              face_vertices_image, face_vertices_camera[:, :, :, -1:]) #MJ: face_vertices_camera is projected on image plane (z=-1) and forms face_vertices_image
            depth_map = self.normalize_depth(depth_map) #MJ: face_vertices_image: 2D positions of the face vertices on image plane, of shape (B, num_faces, 3,2) in [-1,1] 

            uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, uv_face_attr)  #MJ:  uv_face_att: torch.Size([1, 14232, 3, 2]); face_vertices_camera[:, :, :, -1:]:(1,14232,3,1)
            uv_features = uv_features.detach() #MJ: depth_map: (1,1200,1200,1); uv_features:(1,1200,1200,2) = The texture coordinate for each pixel

        else: #MJ: The render_cache exits, and use it
            # logger.info('Using render cache'): MJ: face_normals: torch.Size([1, 14232, 3]); uv_features: torch.Size([1, 1200, 1200, 2]);
            face_normals, uv_features, face_idx, depth_map = render_cache['face_normals'], render_cache['uv_features'], render_cache['face_idx'], render_cache['depth_map']
        mask = (face_idx > -1).float()[..., None] #MJ: face_idx: torch.Size([1, 1200, 1200]); mask: torch.Size([1, 1200, 1200, 1]); -1 is None
        #MJ:  texture_map: torch.Size([1, 3, 1024, 1024]): 3= RGB; self.texture_img or self.meta_texture_img; initially set to random; to be learned
        image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode)
                        # JA: Interpolates texture_maps (1024x1024) by dense or sparse texture_coordinates (uv_features) [1200x1200].
                        #MJ: Coordinates are expected to be normalized between [0, 1]. Note that opengl tex coord is different 
                        # from pytorch’s coord. opengl coord ranges from 0 to 1, y axis is from bottom to top
                        # and it supports circular mode(-0.1 is the same as 0.9) pytorch coord ranges from -1 to 1, 
                        # y axis is from top to bottom
                        
                        #MJ: texture_map = either self.texture_img  or self.meta_texture_img. The former represents the real texture atlas being learned
                        #    self.meta_texture_img represents the meta texture whose 2nd channel is used to  represent the z-normal value of each face in the camera frame

        image_features = image_features * mask # JA: image_features refers to the "rendered" texture coordinates;MJ: Here rendering means the simple projection of the texture map or meta_texture (being lerned) defined on the mesh to the camera screen
        if background_type == 'white':
            image_features += 1 * (1 - mask)
        elif background_type == 'random':
            image_features += torch.rand((1,1,1,3)).to(self.device) * (1 - mask)

        normals_image = face_normals[0][face_idx, :]  #MJ: face_normals has 13232 faces, with the normal vector (x,y,z) for each face; [face_idx,:] = the subset selection
        #MJ:  the number of indices in face_idx (1200 x 1200 = 1,440,000) exceeds the total number of faces (42696)=> some faces are being indexed multiple times
        render_cache = {'uv_features':uv_features, 'face_normals':face_normals,'face_idx':face_idx, 'depth_map':depth_map}
        #MJ: In the case of meta_texture tracking, the uv_features track the seen region and the face_normal tracks the cross sections of the faces
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
