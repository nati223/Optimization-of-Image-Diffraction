# Imports

import numpy as np
import matplotlib.pyplot as plt

# Functions

def convert_to_pi_scale(arr):
    """
    Converts an array of pixel values to an array of angles in radians between 0 and pi.
    Any pixel value above 127 is mapped to 0, and any pixel value below or equal to 127 is mapped to pi.

    Parameters:
    arr (numpy.ndarray): An array of pixel values.

    Returns:
    numpy.ndarray: An array of angles in radians between 0 and pi.
    """

    # convert the array to a float type
    arr = arr.astype("float64")
    # set threshold
    threshold = 127
    # set white to pi and black to 0
    arr = np.where(arr > threshold, 0, np.pi)

    return arr


def from_phase_modulation(arr):
    """
    Converts an array of complex numbers representing the phase-modulated pixel values to an array of pixel values between 0 and 255.

    Parameters:
    arr (numpy.ndarray): An array of complex numbers representing the phase-modulated pixel values.

    Returns:
    numpy.ndarray: An array of pixel values between 0 and 255.
    """

    # calculate the magnitude of the complex array
    magnitude = np.abs(arr)

    # convert the magnitude back to the range of 0 to 255
    pixel_values = convert_to_255(magnitude)

    # round the pixel values to the nearest integer
    pixel_values = np.round(pixel_values)

    # convert the data type of the array to integers
    pixel_values = pixel_values.astype(np.uint8)

    return pixel_values


def convert_to_255(arr):
    """
    Converts an array of angles in radians between 0 and pi to an array of pixel values between 0 and 255.

    Parameters:
    arr (numpy.ndarray): An array of angles in radians between 0 and pi.

    Returns:
    numpy.ndarray: An array of pixel values between 0 and 255.
    """

    # multiply each value in the array by 255/pi to convert it back to the range of 0 to 255
    result = arr * 255 / np.pi

    return result


def expj(arr):
    """
    Calculates the complex exponential of an array of angles in radians.

    Parameters:
    arr (numpy.ndarray): An array of angles in radians.

    Returns:
    numpy.ndarray: An array of complex numbers representing the complex exponential of the input array.
    """

    # convert the input array to complex numbers
    arr_c = arr.astype("complex128")

    # calculate exp(j*arr)
    result = np.exp(1j * arr_c)

    return result


def phase_modulation(arr):
    """
    Applies phase modulation to an array of pixel values.

    Parameters:
    arr (numpy.ndarray): An array of pixel values.

    Returns:
    numpy.ndarray: An array of complex numbers representing the phase-modulated pixel values.
    """

    arr = convert_to_pi_scale(arr)
    return expj(arr)


def inv_expj(arr):
    """
    Calculates the inverse of the complex exponential of an array of angles in radians.

    Parameters:
    arr (numpy.ndarray): An array of complex numbers.

    Returns:
    numpy.ndarray: An array of angles in radians representing the inverse of the complex exponential of the input array.
    """

    # extract the angle component of the complex numbers
    angles = np.angle(arr)

    # divide the angles by j to get the original array
    result = angles / 1j

    return result


def inv_phase_modulation(arr):
    """
    Applies inverse phase modulation to an array of complex numbers.

    Parameters:
    arr (numpy.ndarray): An array of complex numbers.

    Returns:
    numpy.ndarray: An array of pixel values representing the inverse phase-modulated complex numbers.
    """

    arr = inv_expj(arr)
    return convert_to_255(arr)


def random_phase_mask(size):
    """
    Generates a random phase mask of a given size.

    Parameters:
    size (int): The size of the phase mask.

    Returns:
    numpy.ndarray: An array of complex numbers representing the random phase mask.
    """

    arr = np.exp(1j * 2 * np.pi * np.random.rand(size * size).astype("complex128"))
    return arr


def plot_scores(scores, size):
    """
    Plots the percentage scores achieved by a generation of data.

    Parameters:
    scores (numpy array): The scores achieved by each member of a generation.
    size (int): The number of members in the generation.

    Returns:
    None
    """
    percent_scores = (scores / size) * 100
    x = np.arange(len(percent_scores))
    y = percent_scores
    plt.plot(x, y)
    plt.title("Optimization Process")
    plt.xlabel("Generation")
    plt.ylabel("Best Score (%)")
    plt.show()


def display(arr):
    """
    Displays an image represented as a numpy array.

    Parameters:
    arr (numpy array): The image to display, represented as a 2D numpy array.

    Returns:
    None
    """
    # Rescale array to 0-1 range
    arr = arr.astype(np.float32) / 255.0

    # Display image as black and white
    plt.imshow(arr, cmap="gray")
    plt.show()


def display_example(image, optimal_mask, intensity):
    """
    Displays the digit image, the optimal mask and the intensity.

    Parameters:
    image (numpy array): The image to display, represented as a 2D numpy array.
    optimal_mask (numpy array): The optimal mask, represented as a 2D numpy array.
    intensity (numpy array): The intensity shown on the output screen, represented as a 2D numpy array.

    Returns:
    None
    """
    # Get the dimensions of the result array
    X, Y = intensity.shape
    upper_left_intens = np.sum(intensity[:X//2, :Y//2])
    # Calculate the intensity in the upper right quarter of the frame
    upper_right_intens = np.sum(intensity[:X//2, Y//2:])
    # Calculate the intensity in the lower left quarter of the frame
    lower_left_intens = np.sum(intensity[X//2:, :Y//2])
    # Calculate the intensity in the lower right quarter of the frame
    lower_right_intens = np.sum(intensity[X//2:, Y//2:])

    quarters = [
        upper_right_intens,
        upper_left_intens,
        lower_right_intens,
        lower_left_intens,
    ]
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot a image from the test dataset
    axs[0].imshow((image.astype(np.float32) / 255.0), cmap="gray")
    axs[0].set_title("Original Image")

    # Plot a optimal mask from the test dataset
    axs[1].imshow(np.angle(optimal_mask), cmap="gray")
    axs[1].set_title("Optimal Mask")

    # Plot a output image from the test dataset
    axs[2].imshow(intensity, cmap="gray")
    axs[2].set_title("Output Image")

    # Remove the axis ticks and labels
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Adjust the spacing beteen the plots
    plt.subplots_adjust(wspace=0.1)

    # Display the plots
    plt.show()
    
    # Check classification
    if max(quarters) == upper_left_intens:
        print("Predicted digit: 0")
    elif max(quarters) == upper_right_intens:
        print("Predicted digit: 1")
    elif max(quarters) == lower_left_intens:
        print("Predicted digit: 2")
    else:
        print("Predicted digit: 3")
    
  
