"""Training entry point for the Gemma Security classifier."""

from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from .config import MAX_LENGTH, OUTPUT_MODEL_DIR, TRAINING_ARGS
from .data import build_dataset_dict
from .model import load_classifier_model, load_tokenizer


def tokenize_dataset(dataset_dict, tokenizer):
    """Tokenize prompts and rename label_id into the Trainer's expected labels field."""

    def tokenize_batch(batch):
        return tokenizer(batch["prompt"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset_dict.map(tokenize_batch, batched=True)
    tokenized = tokenized.rename_column("label_id", "labels")
    return tokenized.remove_columns(["text", "label", "prompt"])


def train():
    """Train Gemma 2 on the project dataset and save the result."""

    tokenizer = load_tokenizer()
    model = load_classifier_model()
    dataset_dict = build_dataset_dict()
    tokenized_dataset = tokenize_dataset(dataset_dict, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        **TRAINING_ARGS,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_MODEL_DIR))
    tokenizer.save_pretrained(str(OUTPUT_MODEL_DIR))


if __name__ == "__main__":
    train()
