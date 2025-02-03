import numpy as np
from PIL import Image
from matplotlib import cm

def Compress(img, ComponentCount):
    """
    Compresses an image using Singular Value Decomposition.
    
    Parameters:
      img: 2D NumPy array (grayscale image)
      ComponentCount: number of singular values to keep
      
    Returns:
      (U_reduced, SingularValues_reduced, V_reduced, CompressionRatio)
    """
    U, Sigma, V = np.linalg.svd(img, full_matrices=False)
    U_reduced = U[:, :ComponentCount]
    Sigma_reduced = Sigma[:ComponentCount]
    V_reduced = V[:ComponentCount, :]
    original_size = img.size
    compressed_size = U_reduced.size + Sigma_reduced.size + V_reduced.size
    Ratio = original_size / compressed_size
    return (U_reduced, Sigma_reduced, V_reduced, Ratio)

def Decompress(U, SingularValues, V):
    """
    Reconstructs the image from the compressed SVD representation.
    """
    Sigma = np.diag(SingularValues)
    A = U @ Sigma @ V
    return A

if __name__ == "__main__":
    image_path = "Lena.png"  # Make sure this image exists for local testing.
    img = Image.open(image_path).convert("L")
    imgmat = np.array(img, dtype=float)
    U, singular_values, V, ratio = Compress(imgmat, 32)
    decompressed = Decompress(U, singular_values, V)
    print("Compression ratio:", ratio)
