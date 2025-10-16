import numpy as np
import random
from genetic_algorithm.individual import Individual
from genetic_algorithm.fitness import apply_palette, compute_fitness_psnr
from genetic_algorithm.crossover_operators import random_crossover
from genetic_algorithm.mutation_operators import mutate_palette_random
from utils.image_io import create_palette_image

def run_genetic_algorithm(
    image,
    num_colors,
    max_generations=50,
    population_size=20,
    crossover_prob=0.8,
    mutation_prob=0.15,
    fixed_palette=None,
    fitness_function=None,         
    crossover_function=None,       
    mutation_function=None,
    target_fitness=50        
):

    #Set default functions if not provided
    if fitness_function is None:
        fitness_function = compute_fitness_psnr
    if crossover_function is None:
        crossover_function = random_crossover
    if mutation_function is None:
        mutation_function = mutate_palette_random

    #Metadata for output of color palette image 
    image_name = "Scenery"
    fitness_name = fitness_function.__name__
    crossover_name = crossover_function.__name__
    mutation_name = mutation_function.__name__


    #Population initialization
    if fixed_palette is not None:
        #If a fixed palette is provided, initialize all individuals with it
        population = [Individual(num_colors) for _ in range(population_size)]
        for i, ind in enumerate(population):
            ind.palette = fixed_palette[i % len(fixed_palette)].copy()
    else:
        population = [Individual(num_colors) for _ in range(population_size)]
    

    fitness_history = []

    overall_best = None  #Keep best individual ever

    for gen in range(max_generations):
        #Fitness Evaluation
        for ind in population:
            reconstructed = apply_palette(image, ind.palette)
            ind.fitness = fitness_function(image, reconstructed)

        #Sort by fitness (lower is better)
        population.sort(key=lambda x: x.fitness)
        best_individual = population[0]
        

        # Track best ever
        if overall_best is None or best_individual.fitness < overall_best.fitness:
            overall_best = Individual(num_colors)
            overall_best.palette = best_individual.palette.copy()
            overall_best.fitness = best_individual.fitness
            fitness_history.append(best_individual.fitness)

        print(f"Generation {gen + 1}, Best fitness: {best_individual.fitness:.4f}")


        # Check stopping criteria (target fitness value)
        if (
            target_fitness is not None and
            fitness_function == compute_fitness_psnr and
            best_individual.fitness <= -target_fitness
        ):
            print(f"Early stopping at generation {gen + 1}, fitness reached target: {best_individual.fitness:.4f}")
            break


        #Selection (Tournament selection)
        def tournament_select(k=3):
            return min(random.sample(population, k), key=lambda x: x.fitness)

        new_population = []

        while len(new_population) < population_size:
            parent1 = tournament_select()
            parent2 = tournament_select()

            #Crossover
            if random.random() < crossover_prob:
                child_palette = crossover_function(parent1.palette, parent2.palette)
            else:
                child_palette = parent1.palette.copy()

            #Mutation
            if random.random() < mutation_prob:
                child_palette = mutation_function(child_palette)

            child = Individual(num_colors)
            child.palette = child_palette
            new_population.append(child)

            
        #Add elitism
        new_population = [overall_best] + new_population[:-1]

        population = new_population

    #Final evaluation for results
    for ind in population:
        reconstructed = apply_palette(image, ind.palette)
        ind.fitness = fitness_function(image, reconstructed)

    best_individual = min(population, key=lambda x: x.fitness)
    final_image = apply_palette(image, best_individual.palette)

    #Print the final best fitness and RGB values
    print(f"Final Best Fitness: {best_individual.fitness:.4f}")

    print("Final color palette (RGB values):")
    print(overall_best.palette)

    #Create the color palette with rgb values and metadata
    palette_img = create_palette_image(palette=overall_best.palette, fitness_value=f"{best_individual.fitness:.4f}", image_name=image_name,
                                    fitness_type=fitness_name, mutation_name=mutation_name, crossover_name=crossover_name)
    palette_img.save("final_palette.png")
    palette_img.show()  



    return best_individual.palette, final_image, fitness_history

