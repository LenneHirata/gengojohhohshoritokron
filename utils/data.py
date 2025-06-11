from pydantic import BaseModel

from datasets import Dataset, DatasetDict, load_dataset


class Data(BaseModel):
    id: str
    text: str


class KP20KDataset:
    def __init__(self):
        dataset = load_dataset(path="taln-ls2n/kp20k", trust_remote_code=True)
        assert isinstance(dataset, DatasetDict)

        self.train_dataset = self.__preprocess(dataset["train"])
        self.valid_dataset = self.__preprocess(dataset["validation"])
        self.test_dataset = self.__preprocess(dataset["test"])

    def __preprocess(self, dataset: Dataset) -> list[Data]:
        return [
            Data(id=id, text=text.lower())
            for id, text in zip(dataset["id"], dataset["abstract"], strict=True)
        ]


if __name__ == "__main__":
    _ = KP20KDataset()
