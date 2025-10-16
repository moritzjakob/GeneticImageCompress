import numpy as np



def uniform_crossover_palette(p1, p2):
    """
    Perform uniform crossover between two palettes.
    """

    #Boolean Mask with random True or False values
    mask = np.random.randint(0, 2, size=(p1.shape[0], 1), dtype=bool)
    return np.where(mask, p1, p2) # Color of p1 if mask value == True


def segmented_crossover(p1, p2):
    """
    Perform one-point segmented crossover between two palettes.
    """

    n = p1.shape[0]
    split = np.random.randint(1, n)
    return np.vstack((p1[:split], p2[split:]))

def blx_alpha_crossover_palette(p1, p2, alpha=0.2):
    """
    Perform BLX-alphacrossover for palettes.
    
    For each color channel, sample child values uniformly in an interval 
    extended by alpha beyond the min and max of parents.
    """

    min_vals = np.minimum(p1, p2)
    max_vals = np.maximum(p1, p2)
    diff = max_vals - min_vals
    lower = min_vals - alpha * diff
    upper = max_vals + alpha * diff
    child = np.random.uniform(lower, upper) # Sample each color channel value uniformly between the bounds
    return np.clip(np.round(child), 0, 255).astype(np.uint8)  #Round values + clip to [0, 255] and convert to uint8 for color value


def random_crossover(p1, p2):
    """
    Randomly select one of the defined crossover methods .
    """
    
    crossover_methods = [
        uniform_crossover_palette,
        blx_alpha_crossover_palette,
        segmented_crossover
    ]
    crossover_function = np.random.choice(crossover_methods)
    
    return crossover_function(p1, p2)


