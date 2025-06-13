import itertools
import random
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from utils.c2n_map import C2NMap
from utils.evaluate import evaluate
from utils.data import Data, KP20KDataset

from .hirata_search import SearchDecipher


class GeneticDecipher(SearchDecipher):
    def __init__(self, elite_size: int = 10, mutation_rate: float = 0.05, sex_rate: float = 0.9):
        super().__init__(dataset=[])

        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.sex_rate = sex_rate

    def init_maps(self, dataset: list[Data]) -> None:
        self.chars = set([char for data in dataset for char in data.text])

        self.population = [C2NMap(c2n={char: random.randint(0, 9) for char in self.chars}) for _ in range(self.elite_size**2)]

    def __sex(self, map1: C2NMap, map2: C2NMap) -> C2NMap:
        new_c2n = {}
        for char in self.chars:
            if random.random() < self.sex_rate:
                new_c2n[char] = map1.c2n[char]
            else:
                new_c2n[char] = map2.c2n[char]
        return C2NMap(c2n=new_c2n)

    def __mutate(self, c2n_map: C2NMap) -> C2NMap:
        if random.random() > self.mutation_rate:
            return c2n_map

        return self.__sex(c2n_map, C2NMap(c2n={char: random.randint(0, 9) for char in self.chars}))

    def evaluate_individual(self, c2n_map: C2NMap, train_data: list[Data], remote: bool = True) -> tuple[C2NMap, float]:
        self.c2n_map = c2n_map
        score = evaluate(self, train_data, save=False, remote=remote)
        return (c2n_map, score)

    def evaluate_func(self, args: tuple[C2NMap, list[Data]]):
        return self.evaluate_individual(*args)

    def evolve(
        self,
        train_dataset: list[Data],
        valid_dataset: list[Data],
        generations: int = 1000,
        train_data_per_generation: int = 256,
        early_stopping_count: int = 5,
        parallel: bool = True,
    ) -> None:
        self.init_maps(train_dataset)

        best_score = 0
        best_c2n_map = None
        non_evolvution_count = 0

        for generation in tqdm(range(generations), desc="Generation"):
            train_data = random.sample(train_dataset, train_data_per_generation)

            scores = []
            if parallel:
                with Pool(processes=cpu_count()) as pool:
                    for result in tqdm(
                        pool.imap_unordered(self.evaluate_func, [(c2n_map, train_data) for c2n_map in self.population]),
                        total=len(self.population),
                        desc="Evaluating population",
                        leave=False,
                    ):
                        scores.append(result)
            else:
                for c2n_map in tqdm(self.population, desc="Evaluating population", leave=False):
                    scores.append(self.evaluate_individual(c2n_map, train_data, remote=False))

            scores.sort(key=lambda x: x[1], reverse=True)
            best_population = [c2n_map for c2n_map, _ in scores[: self.elite_size]]

            self.c2n_map = best_population[0]
            best_score_in_generation = evaluate(self, valid_dataset, save=False)

            if best_score_in_generation > best_score:
                best_score = best_score_in_generation
                best_c2n_map = best_population[0]
                non_evolvution_count = 0
            else:
                non_evolvution_count += 1

            self.population = list(
                map(self.__mutate, [self.__sex(map1, map2) for map1, map2 in itertools.product(best_population, repeat=2)])
            )

            tqdm.write(f"Geneartion {generation}")
            tqdm.write(f"Best train score: {scores[0][1]}")
            tqdm.write(f"Best score: {best_score}, Generation Best: {best_score_in_generation}")
            tqdm.write(f"Best map: {best_c2n_map}")

            if non_evolvution_count > early_stopping_count:
                print(f"No evolution for {early_stopping_count} generations, stopping")
                break

        print(f"Best score: {best_score}, {best_c2n_map}")

        assert best_c2n_map
        self.c2n_map = best_c2n_map


if __name__ == "__main__":
    dataset = KP20KDataset()

    model = GeneticDecipher()
    model.obtain_words(dataset.train_dataset)

    model.evolve(train_dataset=dataset.train_dataset, valid_dataset=dataset.valid_dataset, parallel=False)

    evaluate(model, dataset.test_dataset)
