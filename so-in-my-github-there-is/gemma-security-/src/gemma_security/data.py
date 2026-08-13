"""Data loading and prompt preparation for the Gemma Security project."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from .config import LABEL_TO_ID, RANDOM_SEED, RAW_DATA_PATH, TEST_SIZE


@dataclass(frozen=True)
class SecurityExample:
    """One labeled security example."""

    text: str
    label: str


def build_security_prompt(text: str) -> str:
    """Wrap raw security text in a consistent instruction prompt."""

    return (
        "Classify the following cybersecurity event into one security label.\n\n"
        f"Event:\n{text}\n\n"
        "Label:"
    )


def load_security_dataframe(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load a CSV dataset and validate the required columns."""

    dataframe = pd.read_csv(path)
    required_columns = {"text", "label"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    unknown_labels = sorted(set(dataframe["label"]) - set(LABEL_TO_ID))
    if unknown_labels:
        allowed = ", ".join(LABEL_TO_ID)
        found = ", ".join(unknown_labels)
        raise ValueError(f"Unknown label(s): {found}. Allowed labels: {allowed}")

    dataframe = dataframe[["text", "label"]].dropna().reset_index(drop=True)
    dataframe["prompt"] = dataframe["text"].map(build_security_prompt)
    dataframe["label_id"] = dataframe["label"].map(LABEL_TO_ID)
    return dataframe


def build_dataset_dict(path=RAW_DATA_PATH) -> DatasetDict:
    """Split the CSV into Hugging Face train and test datasets."""

    dataframe = load_security_dataframe(path)
    label_counts = dataframe["label"].value_counts()
    can_stratify = dataframe["label"].nunique() > 1 and label_counts.min() >= 2

    train_df, test_df = train_test_split(
        dataframe,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=dataframe["label"] if can_stratify else None,
    )

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
        }
    )
