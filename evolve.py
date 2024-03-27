import merge
import os
import random
import uuid
import yaml
from safetensors.torch import save_file

from pathlib import Path

class Candidate:
    def __init__(self, file_path, p, lambda_val, initial_population=False):
        self.file_path = file_path
        self.initial_population = initial_population
        self.p = p
        self.lambda_val = lambda_val
        self.fitness = None

def random_p():
    return random.random() / 1.5

def random_lambda():
    return (random.random() + 1)/2*4

def selection(population, num_parents):
    print("Selecting random parents...")
    return random.sample(population, num_parents)

def crossover(parents, output_folder):
    print("Crossover(DARE merge)...")
    file_path = str(Path(output_folder) / (str(uuid.uuid4())+".safetensors"))
    offspring = Candidate(file_path, parents[0].p, parents[0].lambda_val)
    tensor_map = merge.merge_safetensors(parents[0].file_path, parents[1].file_path, parents[1].p, parents[1].lambda_val)

    for parent in parents[2:]:
        tensor_map = merge.merge_safetensors(offspring.file_path, parent.file_path, parent.p, parent.lambda_val)

    print(f"Built tensor_map, saving to {offspring.file_path}")
    save_file(tensor_map, offspring.file_path)
    del tensor_map
    return offspring

def mutation(offspring, threshold=0.05):
    print("Mutating offspring...")
    if random.random() <= threshold:
        offspring.p = random_p()
        offspring.lambda_val = random_lambda()

def evolve(population, population_size, num_parents, mutation_rate, children_count=2, output_folder="evolve_output"):
    seed_population = list(population)
    log_candidates(population)
    while len(population) < population_size:
        parents = selection(seed_population, num_parents)
        for i in range(min(children_count, population_size - len(population))):
            offspring = crossover(parents, output_folder)
            mutation(offspring, mutation_rate)
            population.append(offspring)

    log_candidates(population)
    return population

async def evaluate(candidate, criteria):
    print(f"Evaluating... {candidate.fitness}")
    candidate.fitness = await criteria(candidate)

async def run_evolution(population, elite_size, num_parents, population_size, mutation_rate, evaluation_criteria):
    population = evolve(population, population_size, num_parents, mutation_rate)

    for candidate in population:
        await evaluate(candidate, evaluation_criteria)

    population.sort(key=lambda c: c.fitness, reverse=True)
    for tokill in population[elite_size:]:
        if not tokill.initial_population:
            os.remove(tokill.file_path)
    return population[:elite_size]

def log_candidates(population):
    format_str = "{{0}}. {{1:<24}} - {{2}}".format()
    for index, candidate in enumerate(population, start=1):
        print(format_str.format(index, candidate.file_path, candidate.fitness))

def load_candidates(file_path):
    candidates = []
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    for candidate_data in data["models"]:
        p = candidate_data.get('p', random_p())
        lambda_val = candidate_data.get("lambda", random_lambda())
        candidate = Candidate(candidate_data['model'], p=p, lambda_val=lambda_val, initial_population=True)
        candidates.append(candidate)
    return candidates
