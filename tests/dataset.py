from utils.data import KP20KDataset


def test_count() -> None:
    datasets = [KP20KDataset()]
    counts = [(530809, 20000, 20000)]

    for dataset, count in zip(datasets, counts):
        assert len(dataset.train_dataset) == count[0]
        assert len(dataset.valid_dataset) == count[1]
        assert len(dataset.test_dataset) == count[2]
