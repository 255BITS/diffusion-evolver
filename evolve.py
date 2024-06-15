import logging
import merge
import numpy as np
import os
import random
import torch
import uuid
import yaml
import sys
from safetensors.torch import save_file

from pathlib import Path

class Candidate:
    def __init__(self, file_path, p, lambda_val, initial_population=False, generation=0, seed=None):
        self.file_path = file_path
        self.initial_population = initial_population
        self.p = p
        self.lambda_val = lambda_val
        self.generation = generation
        if seed is None:
            self.seed = random.randint(-sys.maxsize-1, sys.maxsize)
        if initial_population:
            rand_point = torch.randn(4)
            self.location = rand_point / torch.norm(rand_point)

    def to_dict(self):
        return {
            "model": self.file_path,
            "p": self.p,
            "lambda": self.lambda_val,
            "generation": self.generation,
            "seed": self.seed
        }

def random_p():
    return (random.random() / 2.0)+0.1

def random_lambda():
    return random.random()*2.5+0.5

def calculate_diversity_scores(candidates):
    locations = torch.stack([candidate.location for candidate in candidates])
    centroid = torch.mean(locations, dim=0)
    distances = torch.norm(locations - centroid, dim=1)
    return distances

def adjust_selection_probabilities(distances):
    # Normalize distances to get a base probability, ensuring not overwhelming influence
    min_dist, max_dist = distances.min(), distances.max()
    if min_dist - max_dist < 0.001:
        return [1.0/len(distances) for d in distances]
    # Simple linear transformation to ensure max 30% selection chance increase
    adjusted_probs = (distances - min_dist) / (max_dist - min_dist) * 0.3 + 0.7
    adjusted_probs /= adjusted_probs.sum()  # Normalize to ensure it sums to 1
    return adjusted_probs

def selection(population, num_parents):
    logging.info("Selecting candidates.")
    distances = calculate_diversity_scores(population)
    adjusted_probs = adjust_selection_probabilities(distances)
    selected_indices = np.random.choice(range(len(population)), size=num_parents, replace=False, p=adjusted_probs)

    return [population[i] for i in selected_indices]

def perturb_tensor_map(tensor_map):
    from safetensors.torch import safe_open
    tensor_map = {}
    for key, value in tensor_map.items():
        if 'diffusion_model' in key:
            tensor_map[key] = value + torch.normal(torch.zeros_like(v), value.std() * 0.01)
        else:
            tensor_map[key] = f1.get_tensor(key)
    return tensor_map

def perturb(candidate):
    from safetensors.torch import safe_open
    tensor_map = {}
    with safe_open(candidate.file_path, framework="pt", device="cpu") as f1:
        for key in f1.keys():
            if 'diffusion_model' in key:
                v = f1.get_tensor(key)
                tensor_map[key] = v + torch.normal(torch.zeros_like(v), v.std() * 0.01)
            else:
                tensor_map[key] = f1.get_tensor(key)
    return tensor_map

def mutation(offspring):
    offspring.p = random_p()
    offspring.lambda_val = random_lambda()

def breed(parents, mutation_rate, output_path):
    logging.info("Crossover and mutation...")
    file_path = str(Path(output_path) / (str(uuid.uuid4())+".safetensors"))
    offspring = Candidate(file_path, parents[0].p, parents[0].lambda_val)
    mutation_event = random.random() <= mutation_rate
    if mutation_event:
        mutation(offspring)
    tensor_map = merge.merge_safetensors(parents[0].file_path, parents[1].file_path, offspring.p, offspring.lambda_val)
    mutation_event = random.random() <= mutation_rate
    if mutation_event:
        tensor_map = perturb_tensor_map(tensor_map)


    for parent in parents[2:]:
        tensor_map = merge.merge_safetensors(offspring.file_path, parent.file_path, offspring.p, offspring.lambda_val)

    offspring.generation = max([parent.generation for parent in parents]) + 1
    offspring.location = torch.mean(torch.stack([parent.location for parent in parents]), dim=0)

    logging.info(f"Saving to {offspring.file_path}, from {','.join([p.file_path for p in parents])} p={offspring.p} λ={offspring.lambda_val} gen={offspring.generation}")
    save_file(tensor_map, offspring.file_path)
    del tensor_map
    return offspring

def evolve(population, population_size, num_parents, mutation_rate, output_path, children_count=1):
    seed_population = list(population)
    while len(population) < population_size:
        parents = selection(seed_population, num_parents)
        for i in range(min(children_count, population_size - len(population))):
            offspring = breed(parents, mutation_rate, output_path)
            population.append(offspring)

    return population

async def correct_insert_element(item, sorted_list, compare, top_k):
    if not sorted_list:
        return [item]
    # find a place for insertion
    insert_pos = await find_insertion_point(item, sorted_list, compare, top_k)
    # insert item tentatively
    sorted_list.insert(insert_pos, item)
    return sorted_list

async def find_insertion_point(item, sorted_list, compare, top_k):
    # binary search variant that accounts for potential comparison errors
    low, high = 0, len(sorted_list) - 1
    while low <= high:
        if low > top_k and top_k > 0:
            return low
        mid = (low + high) // 2
        result = await compare(item, sorted_list[mid])
        # adjust binary search based on comparison, considering potential inaccuracies
        if result == 1:
            high = mid - 1
        else:
            low = mid + 1
    return low

async def sort_with_correction(buffer, compare, top_k=-1):
    sorted_list = []
    for item in buffer:
        sorted_list = await correct_insert_element(item, sorted_list, compare, top_k)
    # correction mechanism here
    sorted_list = await correction_pass(sorted_list)
    return sorted_list

async def correction_pass(sorted_list):
    # implement a correction pass, potentially re-comparing elements
    # this could involve heuristic-based swaps or reinsertions
    return sorted_list

def choose_first_occurrence(s, opta, optb):
    # find the index of a and b
    index_a = s.find(opta)
    index_b = s.find(optb)
    # check if both a and b are found
    if index_a != -1 and index_b != -1:
        # return the one that occurs first
        if index_a < index_b:
            return opta
        else:
            return optb
    elif index_a != -1:
        # only a is found
        return opta
    elif index_b != -1:
        # only b is found
        return optb
    else:
        # neither a nor b is found
        return none

async def run_evolution(population, elite_size, num_parents, population_size, mutation_rate, output_path, evaluation_criteria):
    logging.info("Before evolve")
    log_candidates(population)
    population = evolve(population, population_size, num_parents, mutation_rate, output_path)

    logging.info("Before sorting")
    log_candidates(population)
    population = await sort_with_correction(population, evaluation_criteria)
    logging.info("After sorting")
    log_candidates(population)
    for tokill in population[elite_size:]:
        if not tokill.initial_population:
            os.remove(tokill.file_path)
    return population[:elite_size]

def log_candidates(population):
    format_str = "{{0}}. {{1:<24}}".format()
    for index, candidate in enumerate(population, start=1):
        logging.info(format_str.format(index, candidate.file_path))

def load_candidates(file_path):
    candidates = []
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    for candidate_data in data["models"]:
        p = candidate_data.get('p', random_p())
        lambda_val = candidate_data.get("lambda", random_lambda())
        generation = candidate_data.get("generation", 0)
        seed = candidate_data.get("seed", None)
        candidate = Candidate(candidate_data['model'], p=p, lambda_val=lambda_val, initial_population=True, generation=generation, seed=seed)
        candidates.append(candidate)
    return candidates

def write_yaml(population, path):
    yaml_str = yaml.dump({"models": [c.to_dict() for c in population]}, sort_keys=False)

    # Write the YAML string to a file
    with open(path, "w") as file:
        file.write(yaml_str)
