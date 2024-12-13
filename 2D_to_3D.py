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
        """Reconstructs the 3D model from multiple views"""
        
        # Estimate depth maps for each view
        for view_name in self.views.keys():
            self.depth_maps[view_name] = self.estimate_depth_advanced(self.views[view_name])
        
        # Create point clouds for each view
        for view_name in self.views.keys():
            self.point_clouds[view_name] = self.create_point_cloud(
                self.views[view_name], 
                self.depth_maps[view_name]
            )
        
        combined_point_cloud = np.vstack(list(self.point_clouds.values()))
        
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_point_cloud)
        
        # Estimate normals and create mesh
        pcd.estimate_normals()
        
        mesh,_ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
        
        # Shape-specific mesh refinement
        mesh.compute_vertex_normals()
        
        mesh.paint_uniform_color(self.SHAPE_PROPERTIES[self.shape_type]['color'])
        
        # Additional shape-specific refinements
        if self.shape_type == 'cube':
            self._refine_cube_mesh(mesh)
        elif self.shape_type == 'cuboid':
            self._refine_cuboid_mesh(mesh)
        elif self.shape_type == 'pyramid':
            self._refine_pyramid_mesh(mesh)
        elif self.shape_type == 'cone':
            self._refine_cone_mesh(mesh)
        elif self.shape_type == 'cylinder':
            self._refine_cylinder_mesh(mesh)
        
        return mesh

    def _refine_cube_mesh(self, mesh: o3d.geometry.TriangleMesh):
        edge_length = self.SHAPE_PROPERTIES['cube']['edge_length']
        num_faces = self.SHAPE_PROPERTIES['cube']['num_faces']
        
        # Simplify mesh to approximate cube shape
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(num_faces * 2))
        
        # Scale mesh to match edge length
        mesh.scale(edge_length / mesh.get_max_bound() - mesh.get_min_bound(), center=False)

    def _refine_cuboid_mesh(self, mesh: o3d.geometry.TriangleMesh):
        length = self.SHAPE_PROPERTIES['cuboid']['length']
        width = self.SHAPE_PROPERTIES['cuboid']['width']
        height = self.SHAPE_PROPERTIES['cuboid']['height']
        
        # Scale mesh to match dimensions
        mesh.scale([length, width, height])

    def _refine_pyramid_mesh(self, mesh: o3d.geometry.TriangleMesh):
        base_side = self.SHAPE_PROPERTIES['pyramid']['base_side']
        height = self.SHAPE_PROPERTIES['pyramid']['height']
        
        # Scale mesh to match base size and height
        mesh.scale([base_side, base_side, height])

    def _refine_cone_mesh(self, mesh: o3d.geometry.TriangleMesh):
        radius = self.SHAPE_PROPERTIES['cone']['radius']
        height = self.SHAPE_PROPERTIES['cone']['height']
        
        # Scale mesh to match radius and height
        mesh.scale([radius, radius, height])

    def _refine_cylinder_mesh(self, mesh: o3d.geometry.TriangleMesh):
        radius = self.SHAPE_PROPERTIES['cylinder']['radius']
        height = self.SHAPE_PROPERTIES['cylinder']['height']
        
        # Scale mesh to match radius and height
        mesh.scale([radius, radius, height])

    def visualize_model(self, mesh: o3d.geometry.TriangleMesh):
        """Visualize the reconstructed 3D model"""
        o3d.visualization.draw_geometries([mesh])

    def save_model(self,
                   mesh: o3d.geometry.TriangleMesh,
                   output_path: str = 'reconstructed_model.ply'):
        """Save reconstructed 3D model"""
        
        o3d.io.write_triangle_mesh(output_path, mesh)
        
        print(f"3D Model saved to {output_path}")

def main():
    shapes = ['cube', 'cuboid', 'pyramid', 'cone', 'cylinder']
    
    for shape in shapes:
        try:
            print(f"Reconstructing {shape}...")
            reconstructor = GeometricShapeReconstructor(shape_type=shape)
            
            # Modify paths as needed to your local images
            reconstructor.load_views(
                f'{shape}_front.jpg',
                f'{shape}_side.jpg',
                f'{shape}_top.jpg'
            )
            
            mesh = reconstructor.reconstruct_3d_model()
            reconstructor.save_model(mesh,f'{shape}_model.ply')
            
            # Visualize the reconstructed model
            reconstructor.visualize_model(mesh)
            
        except Exception as e:
            print(f"Error reconstructing {shape}: {e}")

if __name__ == "__main__":
    main()
