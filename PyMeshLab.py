import cv2
import numpy as np
import matplotlib.pyplot as plt
import pymeshlab as pymesh
import os

# Utility Functions
def load_and_preprocess_image(image_path, resize_dim=(512, 512)):
    """
    Load and preprocess an image: resizing and converting to grayscale.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, resize_dim)
    return image

def detect_edges(image, shape_type="general"):
    """
    Perform edge detection with Canny and refine contours based on the shape type.
    """
    edges = cv2.Canny(image, threshold1=50, threshold2=150)

    # Refine edges for straight-edged shapes (cubes, cuboids, pyramids)
    if shape_type in ["cube", "cuboid", "pyramid"]:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_edges = np.zeros_like(edges)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            cv2.drawContours(refined_edges, [approx], -1, 255, 1)
        return refined_edges

    # Default: Return edges directly for curved shapes
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

def map_edges_to_3d(front_edges, side_edges, top_edges):
    """
    Map edge pixels from 2D views (front, side, top) to 3D coordinates.
    """
    points = []

    # Front view (x, y, z=0)
    for y, row in enumerate(front_edges):
        for x, pixel in enumerate(row):
            if pixel == 255:
                points.append([x, y, 0])

    # Side view (x=0, y, z)
    for y, row in enumerate(side_edges):
        for z, pixel in enumerate(row):
            if pixel == 255:
                points.append([0, y, z])

    # Top view (x, y=0, z)
    for z, row in enumerate(top_edges):
        for x, pixel in enumerate(row):
            if pixel == 255:
                points.append([x, 0, z])

    return np.array(points, dtype=np.float32)

def save_point_cloud(points, filename="output_point_cloud.ply"):
    """
    Save 3D points to a PLY file for mesh reconstruction.
    """
    with open(filename, "w") as file:
        file.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(points)))
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("end_header\n")
        for p in points:
            file.write("{} {} {}\n".format(p[0], p[1], p[2]))
    print(f"Point cloud saved as: {filename}")

def reconstruct_surface(input_ply, output_stl="output_mesh.stl"):
    """
    Reconstruct a surface mesh from the point cloud using MeshLab.
    """
    ms = pymesh.MeshSet()
    ms.load_new_mesh(input_ply)
    ms.apply_filter("surface_reconstruction_screened_poisson")  # Poisson reconstruction
    ms.save_current_mesh(output_stl)
    print(f"3D Mesh saved as: {output_stl}")

# Main Function
def reconstruct_3d_shape(front_path, side_path, top_path, shape_type="general"):
    """
    Full pipeline to reconstruct a 3D shape from front, side, and top views.
    """
    print("Step 1: Loading and preprocessing images...")
    front_image = load_and_preprocess_image(front_path)
    side_image = load_and_preprocess_image(side_path)
    top_image = load_and_preprocess_image(top_path)

    print("Step 2: Detecting edges...")
    front_edges = detect_edges(front_image, shape_type)
    side_edges = detect_edges(side_image, shape_type)
    top_edges = detect_edges(top_image, shape_type)

    # Visualize edges
    display_image("Front View Edges", front_edges)
    display_image("Side View Edges", side_edges)
    display_image("Top View Edges", top_edges)

    print("Step 3: Mapping edges to 3D coordinates...")
    points = map_edges_to_3d(front_edges, side_edges, top_edges)

    print("Step 4: Saving point cloud...")
    save_point_cloud(points)

    print("Step 5: Reconstructing 3D surface...")
    reconstruct_surface("output_point_cloud.ply")
    print("3D Reconstruction complete!")

# Example Usage
if __name__ == "__main__":
    # Paths to input images
    front_image_path = "images/front_view.png"
    side_image_path = "images/side_view.png"
    top_image_path = "images/top_view.png"

    # Check existence of images
    if not os.path.exists("images"):
        print("Error: 'images' folder not found. Please provide valid images.")
    else:
        print("Starting 3D reconstruction...")
        reconstruct_3d_shape(front_image_path, side_image_path, top_image_path, shape_type="general")
