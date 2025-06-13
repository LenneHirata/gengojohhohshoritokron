from tqdm import tqdm

from utils.c2n_map import C2NMap
from utils.evaluate import evaluate
from utils.data import Data, KP20KDataset

from .utils.interface import DecipherBase


class SearchDecipher(DecipherBase):
    def __init__(self, dataset: list[Data]):
        c2n = {}
        chars = set([char for data in dataset for char in data.text])

        for char in chars:
            match char:
                case "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9":
                    c2n[char] = 0
                case "e" | "s" | "h":
                    c2n[char] = 1
                case "t" | "d" | "p":
                    c2n[char] = 2
                case "a" | "l" | "f":
                    c2n[char] = 3
                case "o" | "u" | "y":
                    c2n[char] = 4
                case "n" | "c" | "w":
                    c2n[char] = 5
                case "i" | "m" | "g":
                    c2n[char] = 6
                case "r" | "b" | "v":
                    c2n[char] = 7
                case "s" | "h" | "z":
                    c2n[char] = 8
                case "k" | "x" | "j" | "q":
                    c2n[char] = 9
                case _:
                    c2n[char] = 0

        super().__init__(c2n_map=C2NMap(c2n=c2n))

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

        candidate_chars_list = [self.c2n_map.n2c[int(char)] for char in word]

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

    model = SearchDecipher(dataset.train_dataset)
    model.obtain_words(dataset.train_dataset)

    evaluate(model, dataset.test_dataset)
