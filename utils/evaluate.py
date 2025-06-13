from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import pandas
from tqdm import tqdm

from models.utils.interface import DecipherBase
from utils.data import Data


class Output(Data):
    encoded: str
    output: str
    correct_count: int
    length: int


def process_data(data: Data, model: DecipherBase) -> Output:
    input_sequence = model.c2n_map(data.text)
    output_text = model.perform(input_sequence)
    correct_count = sum([char1 == char2 for char1, char2 in zip(data.text, output_text)])

    return Output(
        id=data.id,
        text=data.text,
        encoded=input_sequence,
        output=output_text,
        correct_count=correct_count,
        length=len(data.text),
    )


def evaluate(
    model: DecipherBase, dataset: list[Data], save: bool = True, workers: int = cpu_count(), data_per_process: int = 512
) -> float:
    outputs = []

    if len(dataset) < data_per_process * workers:
        for data in tqdm(dataset, desc="Evaluating", leave=save):
            outputs.append(process_data(data, model))
    else:
        with Pool(processes=workers) as pool:
            process_func = partial(process_data, model=model)
            for result in tqdm(
                pool.imap_unordered(process_func, dataset, chunksize=data_per_process),
                total=len(dataset),
                desc="Evaluating",
                leave=save,
            ):
                outputs.append(result)

    accuracy = sum([output.correct_count for output in outputs]) / sum([output.length for output in outputs])

    if save:
        print(f"Accuracy: {accuracy}")
        pandas.DataFrame([output.model_dump() for output in outputs]).to_csv(
            Path(__file__).parent.parent / "outputs" / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
            index=False,
        )

    return accuracy
