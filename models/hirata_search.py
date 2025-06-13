import itertools

from tqdm import tqdm

from utils.c2n_map import C2NMap
from utils.evaluate import evaluate
from utils.data import Data, KP20KDataset

from .utils.interface import DecipherBase


class SearchDecipher(DecipherBase):
    def __init__(self, c2n_map: C2NMap):
        super().__init__(c2n_map)

    def create_map(self, dataset: list[Data]) -> None:
        """
        E
        T, A, O, N, I, R, S, H
        D, L, U, C, M
        P, F, Y, W, G, B, V
        K, X, J, Q, Z
        """

        chars = set([char for data in dataset for char in data.text])

        self.c2n_map.c2n = {}
        self.n2c = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
        }
        for char in chars:
            match char:
                case "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9":
                    self.c2n_map.c2n[char] = 0
                    self.n2c[0].append(char)
                case "e" | "s" | "h":
                    self.c2n_map.c2n[char] = 1
                    self.n2c[1].append(char)
                case "t" | "d" | "p":
                    self.c2n_map.c2n[char] = 2
                    self.n2c[2].append(char)
                case "a" | "l" | "f":
                    self.c2n_map.c2n[char] = 3
                    self.n2c[3].append(char)
                case "o" | "u" | "y":
                    self.c2n_map.c2n[char] = 4
                    self.n2c[4].append(char)
                case "n" | "c" | "w":
                    self.c2n_map.c2n[char] = 5
                    self.n2c[5].append(char)
                case "i" | "m" | "g":
                    self.c2n_map.c2n[char] = 6
                    self.n2c[6].append(char)
                case "r" | "b" | "v":
                    self.c2n_map.c2n[char] = 7
                    self.n2c[7].append(char)
                case "s" | "h" | "z":
                    self.c2n_map.c2n[char] = 8
                    self.n2c[8].append(char)
                case "k" | "x" | "j" | "q":
                    self.c2n_map.c2n[char] = 9
                    self.n2c[9].append(char)
                case _:
                    self.c2n_map.c2n[char] = 0
                    self.n2c[0].append(char)

        print(self.c2n_map.c2n)

    def obtain_words(self, dataset: list[Data]) -> None:
        words = set([word for data in dataset for word in data.text.split()])

        self.words = {}
        for word in words:
            if len(word) not in self.words:
                self.words[len(word)] = []
            self.words[len(word)].append(word)

        print(f"Obtained {len(words)} words")

    def __search_and_fix(self, word: str) -> str:
        """
        self.wordsに存在する単語のうち、最もwordと近いものを返す
        """
        candidate_chars = [self.n2c[int(char)] for char in word]

        for candidate in self.words[len(word)]:
            for char in candidate:
                if char not in candidate_chars:
                    break
            else:
                return candidate

        return word

    def perform(self, text: str) -> str:
        result = []
        for word in text.split():
            result.append(self.__search_and_fix(word))
        return " ".join(result)


if __name__ == "__main__":
    dataset = KP20KDataset()

    model = SearchDecipher(c2n_map=C2NMap(c2n={}))
    model.create_map(dataset.train_dataset)
    model.obtain_words(dataset.train_dataset)

    evaluate(model, dataset.test_dataset)
