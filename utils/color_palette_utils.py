import numpy as np

def generate_palette(num_colors=16):
    """Returns one color palette with random color values"""
    return np.random.randint(0, 256, size=(num_colors, 3))

def generate_palettes(num_palettes=20, num_colors_per_palette=16, seed=45):
    """
    Generates random color palettes using a seed for reproducibility,
    to enable testing the genetic algorithm.
    """

    if seed is not None:
        np.random.seed(seed)
    palettes = []
    for _ in range(num_palettes):
        palettes.append(generate_palette(num_colors_per_palette))
    return np.array(palettes)
