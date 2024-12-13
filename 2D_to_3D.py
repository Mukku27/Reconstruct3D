import numpy as np
import cv2
import open3d as o3d
from typing import List, Tuple
import matplotlib.pyplot as plt

class GeometricShapeReconstructor:
    """Advanced 3D reconstruction class for various geometric shapes"""
    
    SHAPE_PROPERTIES = {
        'cube': {
            'depth_strategy': 'linear',
            'color': [1, 0.706, 0],
            'edge_length': 1.0,
            'num_faces': 6
        },
        'cuboid': {
            'depth_strategy': 'linear_variable',
            'color': [0.8, 0.8, 1],
            'length': 1.5,
            'width': 1.0,
            'height': 0.75
        },
        'pyramid': {
            'depth_strategy': 'triangular',
            'color': [0.5, 1, 0.5],
            'base_side': 1.0,
            'height': 0.75
        },
        'cone': {
            'depth_strategy': 'radial',
            'color': [1, 0.5, 0.5],
            'radius': 0.5,
            'height': 1.0
        },
        'cylinder': {
            'depth_strategy': 'curved',
            'color': [0.7, 0.7, 0.7],
            'radius': 0.5,
            'height': 1.0
        }
    }

    def __init__(self, shape_type: str = 'cube'):
        if shape_type not in self.SHAPE_PROPERTIES:
            raise ValueError(f"Unsupported shape: {shape_type}")
        self.shape_type = shape_type
        self.views = {'front': None, 'side': None, 'top': None}
        self.depth_maps = {}
        self.point_clouds = {}

    def load_views(self, front_path: str, side_path: str, top_path: str,
                   target_size: Tuple[int, int] = (640, 480)):
        """Load and preprocess input images from different views"""
        try:
            self.views['front'] = self._preprocess_image(front_path, target_size)
            self.views['side'] = self._preprocess_image(side_path, target_size)
            self.views['top'] = self._preprocess_image(top_path, target_size)
        except Exception as e:
            raise ValueError(f"Image loading error: {e}")

    def _preprocess_image(self, image_path: str,
                          target_size: Tuple[int, int]) -> np.ndarray:
        """Advanced image preprocessing"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        
        # Resize with aspect ratio preservation
        image = cv2.resize(image, target_size)
        
        # Apply CLAHE for improved contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image

    def estimate_depth_advanced(self, image: np.ndarray) -> np.ndarray:
        """Shape-specific depth estimation"""
        height, width = image.shape
        depth_map = np.zeros((height, width), dtype=np.float32)

        strategy = self.SHAPE_PROPERTIES[self.shape_type]['depth_strategy']
        
        if strategy == 'linear':
            depth_map[:, :] = np.linspace(255, 0, width).reshape(1, -1)
        
        elif strategy == 'linear_variable':
            depth_map[:, :width//2] = np.linspace(255, 127, width//2).reshape(1,-1)
            depth_map[:, width//2:] = np.linspace(127, 0, width - width//2).reshape(1,-1)

        elif strategy == 'triangular':
            for i in range(height):
                depth_map[i,:] = max(0, 255 - (255 / height) * i)

        elif strategy == 'radial':
            center_x, center_y = width // 2, height // 2
            for y in range(height):
                for x in range(width):
                    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    depth_map[y,x] = max(0, 255 - (255 / center_x) * distance)

        elif strategy == 'curved':
            center_x = width // 2
            for x in range(width):
                depth_map[:, x] = max(0, 255 - (255 / center_x) * abs(x - center_x))

        return depth_map.astype(np.uint8)

    def create_point_cloud(self, image: np.ndarray,
                           depth_map: np.ndarray) -> np.ndarray:
        """Create point cloud with advanced filtering"""
        h,w = image.shape
        points = []
        
        for y in range(h):
            for x in range(w):
                depth_value = depth_map[y,x]
                if depth_value > 50:  # Configurable threshold
                    points.append([x,y,depth_value])
        
        return np.array(points)
    
    def reconstruct_3d_model(self) -> o3d.geometry.TriangleMesh:
        """Reconstructs the 3D model from multiple views with advanced processing."""

        # Estimate depth maps for each view
        for view_name in self.views.keys():
            self.depth_maps[view_name] = self.estimate_depth_advanced(self.views[view_name])

        # Create point clouds for each view
        for view_name in self.views.keys():
            self.point_clouds[view_name] = self.create_point_cloud(
                self.views[view_name],
                self.depth_maps[view_name]
            )

        # Combine all point clouds
        combined_point_cloud = np.vstack(list(self.point_clouds.values()))

        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_point_cloud)

        # Pre-process point cloud to remove outliers and downsample
        pcd = self.pre_process_point_cloud(pcd)

        # Attempt reconstruction with Poisson surface reconstruction
        try:
            mesh = self.poisson_surface_reconstruction(pcd)
        except RuntimeError:
            # If Poisson reconstruction fails, fall back to Alpha Shapes
            mesh = self.alpha_shapes_with_noise(pcd)

        # Compute normals and apply shape-specific refinements
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(self.SHAPE_PROPERTIES[self.shape_type]['color'])

        # Refine mesh based on shape type
        shape_refinement_methods = {
            'cube': self._refine_cube_mesh,
            'cuboid': self._refine_cuboid_mesh,
            'pyramid': self._refine_pyramid_mesh,
            'cone': self._refine_cone_mesh,
            'cylinder': self._refine_cylinder_mesh,
        }
        if self.shape_type in shape_refinement_methods:
            shape_refinement_methods[self.shape_type](mesh)

        return mesh

    
    def pre_process_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Pre-process the point cloud to remove noise and downsample."""
        # Remove statistical outliers
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)
        
        # Downsample the point cloud
        voxel_size = 0.02  # Adjust based on your point cloud density
        pcd = pcd.voxel_down_sample(voxel_size)
        
        return pcd
    
    def poisson_surface_reconstruction(self, pcd: o3d.geometry.PointCloud, depth: int = 8) -> o3d.geometry.TriangleMesh:
        """Perform Poisson surface reconstruction."""
        # Estimate normals if not already computed
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        
        # Remove low-density vertices to improve mesh quality
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        if not mesh.is_valid():
            raise RuntimeError("Poisson surface reconstruction failed.")
        
        return mesh

    def alpha_shapes_with_noise(self, pcd: o3d.geometry.PointCloud, alpha: float = 0.03, noise_std: float = 0.001) -> o3d.geometry.TriangleMesh:
        """Reconstruct mesh using Alpha Shapes with noise injection."""
        # Add small random noise to points to avoid degeneracies
        pcd_noisy = o3d.geometry.PointCloud()
        noisy_points = np.asarray(pcd.points) + np.random.normal(scale=noise_std, size=np.asarray(pcd.points).shape)
        pcd_noisy.points = o3d.utility.Vector3dVector(noisy_points)
        
        # Create TetraMesh and Alpha Shape
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd_noisy)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_noisy, alpha, tetra_mesh, pt_map)
        
        return mesh


        
   


    def _refine_cube_mesh(self, mesh: o3d.geometry.TriangleMesh):
        edge_length = self.SHAPE_PROPERTIES['cube']['edge_length']
        num_faces = self.SHAPE_PROPERTIES['cube']['num_faces']
        
        # Simplify mesh to approximate cube shape
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(num_faces * 2))
        
        # Calculate scale factor
        current_size = mesh.get_max_bound() - mesh.get_min_bound()
        scale_factor = edge_length / max(current_size)
        mesh.scale(scale_factor, center=mesh.get_center())

    def _refine_cuboid_mesh(self, mesh: o3d.geometry.TriangleMesh):
        dimensions = np.array([self.SHAPE_PROPERTIES['cuboid']['length'],
                               self.SHAPE_PROPERTIES['cuboid']['width'],
                               self.SHAPE_PROPERTIES['cuboid']['height']])
        
        # Scale mesh to match target dimensions
        current_size = mesh.get_max_bound() - mesh.get_min_bound()
        scale_factors = dimensions / current_size
        mesh.scale(min(scale_factors), center=mesh.get_center())

    def _refine_pyramid_mesh(self, mesh: o3d.geometry.TriangleMesh):
        base_side = self.SHAPE_PROPERTIES['pyramid']['base_side']
        height = self.SHAPE_PROPERTIES['pyramid']['height']
        
        # Scale to approximate pyramid dimensions
        current_size = mesh.get_max_bound() - mesh.get_min_bound()
        scale_factors = np.array([base_side, base_side, height]) / current_size
        mesh.scale(min(scale_factors), center=mesh.get_center())

    def _refine_cone_mesh(self, mesh: o3d.geometry.TriangleMesh):
        radius = self.SHAPE_PROPERTIES['cone']['radius']
        height = self.SHAPE_PROPERTIES['cone']['height']
        
        # Scale to match cone dimensions
        current_size = mesh.get_max_bound() - mesh.get_min_bound()
        scale_factors = np.array([radius * 2, radius * 2, height]) / current_size
        mesh.scale(min(scale_factors), center=mesh.get_center())

    def _refine_cylinder_mesh(self, mesh: o3d.geometry.TriangleMesh):
        radius = self.SHAPE_PROPERTIES['cylinder']['radius']
        height = self.SHAPE_PROPERTIES['cylinder']['height']
        
        # Scale to match cylinder dimensions
        current_size = mesh.get_max_bound() - mesh.get_min_bound()
        scale_factors = np.array([radius * 2, radius * 2, height]) / current_size
        mesh.scale(min(scale_factors), center=mesh.get_center())

    def visualize_mesh(self, mesh: o3d.geometry.TriangleMesh):
        """Visualize the reconstructed 3D mesh"""
        o3d.visualization.draw_geometries([mesh])

    def save_mesh(self, mesh: o3d.geometry.TriangleMesh, filepath: str):
        """Save the reconstructed mesh to a file"""
        o3d.io.write_triangle_mesh(filepath, mesh)

def main():
            shape='cylinder'
            print(f"Reconstructing {shape}...")
            reconstructor = GeometricShapeReconstructor(shape_type=shape)
            
            # Modify paths as needed to your local images
            reconstructor.load_views(
                f'/content/front.jpg',
                f'/content/side.jpg',
                f'/content/top.jpg'
            )
            
            mesh = reconstructor.reconstruct_3d_model()
            reconstructor.save_model(mesh,f'{shape}_model.ply')
            
            # Visualize the reconstructed model
            reconstructor.visualize_model(mesh)
            
        
if __name__ == "__main__":
    main()
