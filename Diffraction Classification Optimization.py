import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import default_rng
pi = np.pi

def make_image_stack(path):
    df = pd.read_csv(path)
    labels = df['label'].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    img_stack = df.to_numpy()
    return img_stack, labels

def get_image_by_index(index, img_stack, dim=28):
    img = img_stack[index]
    img = img.reshape(dim,dim)
    return img

def phase_modulate_img(img):
    width, height = img.shape[0], img.shape[1]

    #Perform binary modulation to phases of 0 or pi, according to threshold
    threshold = 127
    phase = np.where(img > threshold, 0, pi)
    # Convert phase to complex exponent
    modulated = np.exp(1j * phase)
    return modulated

def random_phase_mask(dimx=28, dimy=28):
    random_phases = np.exp(1j * 2 * pi * np.random.rand(dimx*dimy))
    return random_phases

def fresnel_integral(img, wavelength, z):
    # Compute the size of the input image
    height, width = img.shape
    # Compute the sampling interval
    dx = dy = 1 / img.shape[0]
    # Create a meshgrid of sampling points
    x, y = np.meshgrid(np.linspace(-0.5, 0.5-dx, width), np.linspace(-0.5, 0.5-dy, height))
    # Compute the diffraction kernel
    k = 2 * pi / wavelength
    kernel = (np.exp(1j*k*z)/ (1j * wavelength * z))*np.exp(1j * k * (x**2 + y**2) / (2 * z))
    # Compute the Fourier transform of the image
    fft_img = np.fft.fft2(img)
    # Multiply the Fourier transform by the diffraction kernel
    fft_kernel = np.fft.fft2(kernel)
    fft_result = fft_img * fft_kernel
    # Compute the inverse Fourier transform of the result
    result = np.fft.ifft2(fft_result)
    # Compute the intensity of the result
    intensity = np.abs(result)**2
    return intensity

def intensity_comparison(result, label):
    # Get the dimensions of the result array
    M, N = result.shape

    # Calculate the intensity in the upper half of the result array
    upper_intensity = np.sum(np.abs(result[:M//2, :])**2)
    # Calculate the intensity in the lower half of the result array
    lower_intensity = np.sum(np.abs(result[M//2:, :])**2)
    
    #Check if classification was performed correctly
    if ((upper_intensity>lower_intensity and label == 0) or (upper_intensity<lower_intensity and label ==1)):
        return 1
    else:
        return 0

def fitness(mask, img_arr, labels, wavelength, z):
    
    score = 0
    
    #Iterate over all images in the training set and check if 0's field intensity is projected upwards, and 1's downwards
    for i in range(img_arr.shape[0]):
        #Get image from stack
        img = get_image_by_index(i, img_arr)
        #Modulate image
        modulated = phase_modulate_img(img)
        #Adjust mask to the image shape
        mask = mask.reshape((28,28))
        #Apply mask to the image
        phase_shifted_image = modulated * mask
        #Calculate the intensity after diffraction
        intensity = fresnel_integral(phase_shifted_image, 10**-7, 100)
        #Update score according to classification performance
        score += intensity_comparison(intensity, labels[i])
    
    return score

def genetic_algorithm(wavelength, z, pic_dim=28, pop_size=20, num_generations=50, mutation_prob=0.1):
    
    #Load training data
    train_arr, labels = make_image_stack("mnist_train_cleaned.csv")

    rng = default_rng()
    
    population = np.empty([pop_size,pic_dim**2], dtype=complex)
    for i in range(population.shape[0]):
        population[i] = random_phase_mask(28,28) #FIXME
            
    # Loop through generations
    for i in range(num_generations):
        # Evaluate fitness for each mask in the population
        fitness_values = [fitness(mask, train_arr, labels, wavelength, z) for mask in population]
        # Select the top 50% performers to be the parents
        num_parents = pop_size // 2
        sorted_indices = np.argsort(fitness_values)
        parents = population[sorted_indices[-num_parents:]]
        # Generate offspring using crossover and mutation
        offspring = np.zeros_like(parents)
        for j in range(num_parents):
            # Crossover: take half the genes from each parent
            parent1 = parents[j]
            parent2 = parents[np.random.randint(num_parents)]
            crossover_point = rng.integers(pic_dim**2)
            offspring[j, :crossover_point] = parent1[:crossover_point]
            offspring[j, crossover_point:] = parent2[crossover_point:]
            # Mutation: randomly shift some masking elements according to mutation probability
            mutation_mask = rng.random(size=pic_dim**2) < mutation_prob
            offspring[j][mutation_mask] = (np.exp(1j * 2 * pi * np.random.random()))*offspring[j][mutation_mask]
        # Combine parents and offspring to form new population
        population = np.vstack((parents, offspring))
        
    # Return the best mask found
    best_index = np.argmax(fitness_values)
    best_mask = population[best_index].reshape(pic_dim,pic_dim)
    return best_mask

def check_result(mask):
    test_arr, labels = make_image_stack("mnist_test_cleaned.csv")
    
    score = fitness(mask, test_arr, labels, 10**-7, 100)
    print(f"The optimized mask classified correctly {score}/{labels.shape[0]} images, which are {100*score/test_arr.shape[0]}%!")

if(__name__ == "__main__"):
    #z has to signifcantly larger than x,y, see fresnel function. FIXME - smaller z might be enough?
    best = genetic_algorithm(wavelength=10**-7, z=100)
    check_result(best)