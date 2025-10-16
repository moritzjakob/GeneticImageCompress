import numpy as np

def gaussian_mutation_multi(palette, sigma=10, max_col=2):
    """
    Applies gaussian noise to randomly selected colors in the palette.
    """

    mutated = palette.copy()
    num_colors = mutated.shape[0]
    num_col = np.random.randint(1, max_col + 1) #Chooses how many colors to mutate
    for _ in range(num_col):
        index = np.random.randint(0, num_colors) # Picks index for a color from the palette 
        noise = np.random.normal(0, sigma, 3)
        mutated[index] = np.clip(mutated[index] + noise, 0, 255)
    return mutated.astype(np.uint8)


def component_wise_mutation(palette, delta=10):
    """
    Mutates a single color channel of RGB of a random color in the palette
    """

    mutated = palette.copy()
    index = np.random.randint(0, len(mutated))
    channel = np.random.randint(0, 3) #Decides if mutation to R,G,B channel
    change = np.random.randint(-delta, delta + 1)
    mutated[index][channel] = np.clip(int(mutated[index][channel]) + change, 0, 255)
    return mutated.astype(np.uint8)


def random_resetting_mutation(palette):
    """
    Replaces a randomly selected color in the palette with a new random color.
    """

    mutated = palette.copy()
    index = np.random.randint(0, len(mutated))
    mutated[index] = np.random.randint(0, 256, 3)
    return mutated.astype(np.uint8)


def mutate_palette_random(palette):
    """
    Randomly selects one of the mutation operators.
    """
    mutation_functions = [
        gaussian_mutation_multi,  
        random_resetting_mutation,
        component_wise_mutation
    ]
    mutation_fn = np.random.choice(mutation_functions)
    return mutation_fn(palette)




