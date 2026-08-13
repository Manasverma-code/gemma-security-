"""Model and tokenizer loading helpers."""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ID_TO_LABEL, LABEL_TO_ID, MODEL_NAME


def load_tokenizer(model_name: str = MODEL_NAME):
    """Load the Gemma tokenizer and ensure padding works for batches."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_classifier_model(model_name: str = MODEL_NAME):
    """Load Gemma 2 with a classification head."""

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model
