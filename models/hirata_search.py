from copy import deepcopy

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
        words = [word for data in dataset for word in data.text.split()]
        word_count = {}
        for word in tqdm(words):
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1

        word_count_count = {}
        for count in word_count.values():
            if count not in word_count_count:
                word_count_count[count] = 0
            word_count_count[count] += 1

        print(sorted(word_count_count.items(), key=lambda x: x[0]))

        words = set(words)
        self.words = {}
        obtained_words_count = 0
        for word in tqdm(words):
            if len(word) not in self.words:
                self.words[len(word)] = {}

            current_dict = self.words[len(word)]
            for i, char in enumerate(word):
                if char not in current_dict:
                    current_dict[char] = {}

                if i == len(word) - 1:
                    if word_count[word] > 3:
                        current_dict[char] = word_count[word]
                        obtained_words_count += 1
                    else:  # 出現回数が3回以下の単語は省く
                        assert isinstance(current_dict, dict)
                        current_dict.pop(char)
                    break

                current_dict = current_dict[char]

        print(f"Obtained {obtained_words_count} words")

    @staticmethod
    def __search(current_dict: dict, current_word: str, candidate_chars_list: list[list[str]]) -> list[str]:
        words = []
        for char in candidate_chars_list[0]:
            if char not in current_dict:
                continue
            if isinstance(current_dict[char], int):
                words.append(current_word + char)
                continue
            words.extend(SearchDecipher.__search(current_dict[char], current_word + char, candidate_chars_list[1:]))

        return words

    def __search_and_fix(self, word: str) -> str:
        """
        self.wordsに存在する単語のうち、wordから複合されうるものを返す
        """
        if len(word) not in self.words:
            return word

        candidate_chars_list = [self.n2c[int(char)] for char in word]

        words = self.__search(self.words[len(word)], "", candidate_chars_list)

        if not words:
            return word

        return words[0]

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
