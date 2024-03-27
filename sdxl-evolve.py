import argparse
import asyncio
import evolve
import os
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evolutionary merge")
    parser.add_argument('model_list', type=str, help='yml file containing list of models in the initial population')
    parser.add_argument('-seed', dest='seed', type=int, default=None, help='Random seed')
    parser.add_argument('-cycles', dest='cycles', type=int, default=10, help='Number of evolutionary cycles')
    parser.add_argument('-elite', dest='elite', type=int, default=10, help='Number of elite candidates to keep every iteration')
    parser.add_argument('-parents', dest='parents', type=int, default=2, help='Number of parents for each child')
    parser.add_argument('-population', dest='population', type=int, default=50, help='Size of population')
    parser.add_argument('-mutation', dest='mutation', type=float, default=0.05, help='Chance of mutation')
    parser.add_argument('-output_path', dest='output_path', type=str, default="evolve_output", help='Directory to save results')
    parser.add_argument('-criteria', dest='criteria', type=str, default='Which image has the highest quality?', help='Prompt for vlm evaluation')
    return parser.parse_args()

async def evaluate_candidate(candidate):
    # TODO generate images with candidate
    # TODO evaluate with vlm on scale 1-100
    # TODO average the values

    #  TODO research comparrator
    # for image count:
    # 1. generate image with each candidate
    #.2. vlm decision on 'better'
    # return winner
    score = int(random.random() * 100)
    return score

async def main():
    # Parse command-line arguments
    args = parse_arguments()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    population = evolve.load_candidates(args.model_list)
    print("Beginning evolution")
    for i in range(args.cycles):
        population = await evolve.run_evolution(population, args.elite, args.parents, args.population, args.mutation, evaluate_candidate)

    print("Resulting population:")
    evolve.log_candidates(population)
if __name__ == "__main__":
    asyncio.run(main())
