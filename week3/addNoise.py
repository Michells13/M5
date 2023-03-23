import numpy as np

def addGaussianNoise(image, mask):
    """
    This function adds gaussian noise in the image, in the pixels that the mask shows

    Parameters
    ----------
    image : numpy array
        Image to add noise.
    mask : numpy array
        Mask with bool values. True if it needs to replace by noise.

    Returns
    -------
    image : numpy array
        Image with added gaussian noise.

    """
    
    noiseImage = np.random.randn(*image.shape) * 50 + 128
    noiseImage = noiseImage.astype(np.uint8)
    image[mask] = noiseImage[mask]
    
    return image
    
def addRandomNoise(image, mask):
    """
    This function adds random noise in the image, in the pixels that the mask shows

    Parameters
    ----------
    image : numpy array
        Image to add noise.
    mask : numpy array
        Mask with bool values. True if it needs to replace by noise.

    Returns
    -------
    image : numpy array
        Image with added random noise.

    """
    noiseImage = np.random.rand(*image.shape) * 255
    noiseImage = noiseImage.astype(np.uint8)
    image[mask] = noiseImage[mask]
    
    return image

def addSPNoise(image, mask):
    """
    This function adds salt and pepper noise in the image, in the pixels that the mask shows

    Parameters
    ----------
    image : numpy array
        Image to add noise.
    mask : numpy array
        Mask with bool values. True if it needs to replace by noise.

    Returns
    -------
    image : numpy array
        Image with added random noise.

    """
    noiseImage = np.random.rand(*image.shape)
    noiseImage[noiseImage < 0.5] = 0
    noiseImage[noiseImage > 0.5] = 255
    noiseImage = noiseImage.astype(np.uint8)
    image[mask] = noiseImage[mask]
    
    return image
    
def addBlack(image, mask):
    """
    This function adds black areas in the image, in the pixels that the mask shows

    Parameters
    ----------
    image : numpy array
        Image to add black areas.
    mask : numpy array
        Mask with bool values. True if it needs to replace by noise.

    Returns
    -------
    image : numpy array
        Image with black areas.

    """
    black = np.zeros(image.shape, dtype = np.uint8)
    image[mask] = black[mask]
    
    return image
    



