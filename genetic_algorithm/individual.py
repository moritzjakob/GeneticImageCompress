import numpy as np

class Individual:
    """
    Represents an individual in the genetic algorithm, 
    characterized by a color palette and its fitness value.
    """

    def __init__(self, num_colors):
        """
        Initializes the individual with a random color palette.
        """
        
        self.palette = np.random.randint(0, 256, size=(num_colors, 3))
        self.fitness = None
