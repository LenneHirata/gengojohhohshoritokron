from utils.c2n_map import C2NMap
from utils.evaluate import evaluate
from utils.data import KP20KDataset

from .utils.interface import DecipherBase


class DecipherModel(DecipherBase):
    def perform(self, text: str) -> str:
        return "a" * len(text)


if __name__ == "__main__":
    dataset = KP20KDataset()

    chars = set([char for data in dataset.train_dataset for char in data.text])
    print(chars)

    model = DecipherModel(c2n_map=C2NMap(c2n={char: 0 for char in chars}))

    evaluate(model, dataset.test_dataset)
