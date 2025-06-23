import numpy as np
import random


def fixed_pattern(image, intensity=30, pattern_type="checkerboard"):
    """
    Adds fixed pattern noise to an image.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).
        intensity (int): Maximum deviation for noise pattern (0-255).
        pattern_type (str): Type of pattern ('horizontal', 'vertical', 'grid', 'diagonal', 'checkerboard', 'sinusoidal', 'random_rows', 'random_columns').

    Returns:
        numpy.ndarray: Noisy image with fixed pattern noise.
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Generate different types of fixed pattern noise
    if pattern_type == "horizontal":
        pattern = np.tile(np.linspace(-intensity, intensity, width), (height, 1))

    elif pattern_type == "vertical":
        pattern = np.tile(np.linspace(-intensity, intensity, height), (width, 1)).T

    elif pattern_type == "grid":
        pattern = np.fromfunction(
            lambda y, x: ((x // 10) % 2 - 0.5) * intensity
            + ((y // 10) % 2 - 0.5) * intensity,
            (height, width),
        )

    elif pattern_type == "diagonal":
        pattern = np.fromfunction(
            lambda y, x: ((x + y) % 20 - 10) * (intensity / 10), (height, width)
        )

    elif pattern_type == "checkerboard":
        pattern = np.fromfunction(
            lambda y, x: ((x.astype(int) // 5 % 2) ^ (y.astype(int) // 5 % 2))
            * intensity,
            (height, width),
            dtype=int,
        )

    elif pattern_type == "sinusoidal":
        x_pattern = np.sin(np.linspace(0, np.pi * 4, width)) * intensity
        y_pattern = np.sin(np.linspace(0, np.pi * 4, height)) * intensity
        pattern = np.outer(y_pattern, x_pattern) / np.max(
            np.abs(y_pattern)
        )  # Normalize effect

    elif pattern_type == "random_rows":
        row_noise = np.random.uniform(-intensity, intensity, size=(height, 1))
        pattern = np.tile(row_noise, (1, width))

    elif pattern_type == "random_columns":
        col_noise = np.random.uniform(-intensity, intensity, size=(1, width))
        pattern = np.tile(col_noise, (height, 1))

    else:
        raise ValueError(
            "Invalid pattern type. Choose from 'horizontal', 'vertical', 'grid', 'diagonal', 'checkerboard', 'sinusoidal', 'random_rows', or 'random_columns'."
        )

    # Expand pattern for color images
    if len(image.shape) == 3:
        pattern = np.repeat(pattern[:, :, np.newaxis], 3, axis=2)

    # Convert image to float to prevent clipping issues
    noisy_image = image.astype(np.float32) + pattern

    # Add pattern noise and clip values
    # noisy_image = np.clip(image + pattern, 0, 255).astype(np.uint8)

    return noisy_image


def gaussian(image, mean=0, std_dev=25):
    """
    Adds Gaussian noise to an input image.

    Args:
        image (numpy.ndarray): The input image (grayscale or color).
        mean (float): Mean of the Gaussian noise.
            Default: 0
        std_dev (float): Standard deviation of the Gaussian noise.
            Default: 25

    Returns:
        numpy.ndarray: The noisy image.
    """

    noise = np.random.normal(mean, std_dev, image.shape)

    noisy_image = image + noise

    # noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image


def hurl(image: np.ndarray, ratio_of_changed_pixels: float = 0.2):
    """
    Change randomly the ratio_of_changed_pixels * image.size pixel values of the inputted image.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        ratio_of_changed_pixels (float): The ratio of pixels to change (0.0 to 1.0).

    Returns:
        numpy.ndarray: The modified image with randomly changed pixels.
    """
    # Ensure the ratio is between 0 and 1
    if not (0.0 <= ratio_of_changed_pixels <= 1.0):
        raise ValueError("ratio_of_changed_pixels must be between 0.0 and 1.0")

    # Calculate the number of pixels to change
    num_pixels_to_change = int(ratio_of_changed_pixels * image.size)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Generate random pixel coordinates
    random_coords = [
        (random.randint(0, height - 1), random.randint(0, width - 1))
        for _ in range(num_pixels_to_change)
    ]

    # Create noisy hurl image
    noisy = image.copy()

    for coord in random_coords:
        y, x = coord
        # Change the pixel value to a random value (assuming the image is grayscale)
        noisy[y, x] = np.random.uniform(0.0, image.max())

    return noisy


def mixed(
    image: np.ndarray,
    scale_shot: float = 10,
    std_gaussian: float = 0.01,
    intensity_pattern: float = 10,
    pattern_type="checkerboard",
    weights=[0.01, 0.79, 0.2],
):
    weight_shot, weight_gaussian, weight_pattern = weights
    return (
        weight_shot * shot(image, scale=scale_shot)
        + weight_gaussian * gaussian(image, std_dev=std_gaussian)
        + weight_pattern
        * fixed_pattern(image, intensity=intensity_pattern, pattern_type=pattern_type)
    )


def shot(image, scale=30):
    """
    Adds shot noise (Poisson noise) to an image.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color).
        scale (float): Scaling factor to control noise intensity.

    Returns:
        numpy.ndarray: Noisy image.
    """
    # Convert image to float to prevent clipping issues
    image = image.astype(np.float32)

    # Normalize image to range [0, 1] if it's not already
    if image.max() > 1:
        image = image / image.max()

    # Apply Poisson noise
    noisy = np.random.poisson(image * scale) / scale

    # Rescale back to original range
    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)

    return noisy.astype(np.float32) / noisy.max()


def speckle(image: np.ndarray, salt_prob: float = 0.3, pepper_prob: float = 0.3):
    """
    Add salt and pepper noise to an image.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        salt_prob (float): Probability of a pixel being replaced with salt (white) noise.
        pepper_prob (float): Probability of a pixel being replaced with pepper (black) noise.

    Returns:
        numpy.ndarray: The image with added salt and pepper noise.
    """
    # Ensure the probabilities are between 0 and 1
    if not (0.0 <= salt_prob <= 1.0) or not (0.0 <= pepper_prob <= 1.0):
        raise ValueError("Probabilities must be between 0.0 and 1.0")

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Create a copy of the image to modify
    noisy_image = image.copy()

    # Add salt noise
    num_salt = int(salt_prob * image.size)
    salt_coords = [
        (random.randint(0, height - 1), random.randint(0, width - 1))
        for _ in range(num_salt)
    ]
    for coord in salt_coords:
        y, x = coord
        noisy_image[y, x] = image.max()  # Assuming the maximum value is 255 (white)

    # Add pepper noise
    num_pepper = int(pepper_prob * image.size)
    pepper_coords = [
        (random.randint(0, height - 1), random.randint(0, width - 1))
        for _ in range(num_pepper)
    ]
    for coord in pepper_coords:
        y, x = coord
        noisy_image[y, x] = 0  # Assuming the minimum value is 0 (black)

    return noisy_image
