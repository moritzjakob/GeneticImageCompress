from utils.image_io import load_image, save_image
from genetic_algorithm.ga import run_genetic_algorithm
from genetic_algorithm.fitness import compute_fitness_mse, compute_fitness_psnr
from genetic_algorithm.crossover_operators import uniform_crossover_palette, segmented_crossover, blx_alpha_crossover_palette, random_crossover
from genetic_algorithm.mutation_operators import gaussian_mutation_multi, component_wise_mutation, random_resetting_mutation, mutate_palette_random
from utils.color_palette_utils import generate_palettes
import numpy as np
import time
import statistics


"""
# Has to match the number of color and Population size (Population = number of palettes)
palettes = generate_palettes(num_colors_per_palette=16)

image = load_image("images/bird.jpg")
start_time = time.time()
palette, result_image, history = run_genetic_algorithm(
    image=image,
    num_colors=16,
    max_generations=50,
    population_size=20,
    crossover_prob=0.85,
    mutation_prob=0.2,
    fixed_palette=palettes,
    fitness_function=compute_fitness_psnr,          
    crossover_function=random_crossover,  
    mutation_function=mutate_palette_random,
    target_fitness=50
)
save_image(result_image, "experiments/results/bird.jpg")
end_time = time.time()

runtime = end_time - start_time
print(runtime)
"""

# Configurations
image_path = "images/bird.jpg"
num_runs = 1
results = []

# Has to match the number of color and Population size (Population = number of palettes)
palettes = generate_palettes(num_colors_per_palette=16)



for run in range(num_runs):
    print(f"\n[Run {run + 1}/{num_runs}]")

    
    image = load_image(image_path)

    start_time = time.time()
    palette, result_image, history = run_genetic_algorithm(
        image=image,
        num_colors=16,
        max_generations=50,
        population_size=20,
        crossover_prob=0.85,
        mutation_prob=0.2,
        fixed_palette=palettes,  
        fitness_function=compute_fitness_psnr,
        crossover_function=random_crossover,
        mutation_function=mutate_palette_random,
        target_fitness=50
    )
    end_time = time.time()

    runtime = end_time - start_time

    results.append({
        "run": run + 1,
        "runtime": runtime
    })

    # Save image from this run
    save_image(result_image, f"experiments/results/testbirdesult_run{run + 1}.jpg")

# Average Runtime
runtime_vals = [r["runtime"] for r in results]
print(f"Runtime (mean): {statistics.mean(runtime_vals):.2f}")

