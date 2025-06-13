from datetime import datetime
from pathlib import Path

import pandas
from tqdm import tqdm

from models.utils.interface import DecipherBase
from utils.data import Data


class Output(Data):
    encoded: str
    output: str
    correct_count: int
    length: int


def evaluate(model: DecipherBase, dataset: list[Data]) -> None:
    outputs = []
    for data in tqdm(dataset):
        input_sequence = model.c2n_map(data.text)

        output_text = model.perform(input_sequence)

        correct_count = sum(
            [char1 == char2 for char1, char2 in zip(data.text, output_text)]
        )

        outputs.append(
            Output(
                id=data.id,
                text=data.text,
                encoded=input_sequence,
                output=output_text,
                correct_count=correct_count,
                length=len(data.text),
            )
        )

    print(
        f"Accuracy: {sum([output.correct_count for output in outputs]) / sum([output.length for output in outputs])}"
    )

    pandas.DataFrame([output.model_dump() for output in outputs]).to_csv(
        Path(__file__).parent.parent
        / "outputs"
        / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
        index=False,
    )
