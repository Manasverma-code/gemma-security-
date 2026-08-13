"""Central project settings.

Keeping settings in one file makes the notebook and scripts easier to read.
When you change the model, labels, or paths, this is the first place to look.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "security_examples.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_NAME = "google/gemma-2-2b"
OUTPUT_MODEL_DIR = PROJECT_ROOT / "models" / "gemma-security-classifier"

LABELS = [
    "phishing",
    "malware",
    "vulnerability",
    "bruteforce",
    "benign",
]

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

MAX_LENGTH = 512
TEST_SIZE = 0.2
RANDOM_SEED = 42

TRAINING_ARGS = {
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "num_train_epochs": 1,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "save_steps": 100,
}
