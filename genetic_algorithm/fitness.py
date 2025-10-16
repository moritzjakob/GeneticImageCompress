from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def apply_palette(image, palette):
    """
    Applies the given color palette to the image by mapping each pixel to the nearest palette color
    """

    h, w, _ = image.shape
    flat_img = image.reshape(-1, 3)
    distances = euclidean_distances(flat_img, palette)
    nearest_indices = distances.argmin(axis=1) # Find the index of the nearest palette color for each pixel
    quantized = palette[nearest_indices] # Maps each pixel to its nearest palette color
    return quantized.reshape(h, w, 3) #Reshape to original shape


def compute_fitness_mse(img1, img2):
    """
    Computes the Mean squared error between two images, 
    also used for calculating the psnr fitness.
    """

    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def compute_psnr(original, compressed):
    """
    Computes the psnr value between the two images.
    """
    
    mse = compute_fitness_mse(original, compressed)
    if mse == 0:
        return float('inf')  # exact same quality 
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) #psnr formula 
    return psnr

def compute_fitness_psnr(image, reconstructed):
    """
    Returns fitness score based on the psnr.
    """

    psnr = compute_psnr(image, reconstructed)
    return -psnr #returns negative value so user can also use mse (has to be minimized)



