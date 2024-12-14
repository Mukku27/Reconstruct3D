import cv2
import numpy as np
import matplotlib.pyplot as plt
import pymeshlab as pymesh
import os

# Utility Functions for Image Processing
def load_and_preprocess_image(image_path, resize_dim=(512, 512)):
    """
    Load and preprocess an image: resizing and converting to grayscale.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, resize_dim)
    return image

def detect_edges(image):
    """
    Perform edge detection using Canny edge detection.
    """
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    return edges

def display_image(title, image):
    """
    Display an image using matplotlib.
    """
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def generate_3d_mesh_from_views(front_edges, side_edges, top_edges, output_file="output.stl"):
    """
    Generate a 3D mesh using processed front, side, and top view edges.
    This function creates a point cloud and interpolates surfaces.
    """
    # Create a blank 3D point cloud
    points = []

    # Map front view edges to 3D coordinates
    for y, row in enumerate(front_edges):
        for x, pixel in enumerate(row):
            if pixel == 255:  # Edge pixel
                points.append([x, y, 0])  # (x, y, z=0)

    # Map side view edges
    for y, row in enumerate(side_edges):
        for z, pixel in enumerate(row):
            if pixel == 255:
                points.append([0, y, z])  # (x=0, y, z)

    # Map top view edges
    for z, row in enumerate(top_edges):
        for x, pixel in enumerate(row):
            if pixel == 255:
                points.append([x, 0, z])  # (x, y=0, z)

    # Convert points to numpy array
    points = np.array(points, dtype=np.float32)

    # Save as a point cloud in .ply format
    point_cloud_file = "temp_point_cloud.ply"
    with open(point_cloud_file, "w") as file:
        file.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(points)))
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("end_header\n")
        for p in points:
            file.write("{} {} {}\n".format(p[0], p[1], p[2]))

    # Use MeshLab to reconstruct the surface from point cloud
    ms = pymesh.MeshSet()
    ms.load_new_mesh(point_cloud_file)
    ms.apply_filter("surface_reconstruction_screened_poisson")  # Poisson surface reconstruction
    ms.save_current_mesh(output_file)
    print(f"3D Mesh saved as: {output_file}")

# Main Program
def reconstruct_3d_shape(front_image_path, side_image_path, top_image_path):
    """
    Main function to reconstruct 3D shapes from front, side, and top views.
    """
    # Step 1: Load and preprocess images
    print("Loading and preprocessing images...")
    front_image = load_and_preprocess_image(front_image_path)
    side_image = load_and_preprocess_image(side_image_path)
    top_image = load_and_preprocess_image(top_image_path)

    # Step 2: Detect edges
    print("Detecting edges...")
    front_edges = detect_edges(front_image)
    side_edges = detect_edges(side_image)
    top_edges = detect_edges(top_image)

    # Display edges for verification
    display_image("Front View Edges", front_edges)
    display_image("Side View Edges", side_edges)
    display_image("Top View Edges", top_edges)

    # Step 3: Generate 3D mesh
    print("Generating 3D mesh from views...")
    generate_3d_mesh_from_views(front_edges, side_edges, top_edges)

# Example usage
if __name__ == "__main__":
    # Paths to input images (front, side, and top views)
    front_image_path = "images/front_view.png"
    side_image_path = "images/side_view.png"
    top_image_path = "images/top_view.png"

    # Ensure the images folder exists
    if not os.path.exists("images"):
        print("Error: 'images' folder not found. Please provide valid image paths.")
    else:
        reconstruct_3d_shape(front_image_path, side_image_path, top_image_path)
