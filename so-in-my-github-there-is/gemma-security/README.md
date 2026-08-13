# Gemma Security

Gemma Security is a starter project for building a small security-focused AI workflow with Gemma 2.

The first goal is simple: take a security-related text input, prepare it in a consistent format, fine-tune or run a Gemma 2 model, and return a useful security classification or explanation.

## Project Map

```text
gemma-security/
├── notebooks/
│   └── 01_gemma2_security_project.ipynb
├── src/
│   └── gemma_security/
│       ├── config.py
│       ├── data.py
│       ├── inference.py
│       ├── model.py
│       └── train.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
├── requirements.txt
└── README.md
```

## What Each Part Does

`notebooks/01_gemma2_security_project.ipynb` is the main learning notebook. Start there. It explains the code, project flow, dataset shape, training idea, and inference path.

`src/gemma_security/config.py` stores project settings such as the model name, label list, file paths, and training defaults.

`src/gemma_security/data.py` loads data and converts rows into prompts that Gemma can understand.

`src/gemma_security/model.py` loads the tokenizer and Gemma model.

`src/gemma_security/train.py` contains the training entry point.

`src/gemma_security/inference.py` contains a small prediction helper for trying the model on new security text.

## Setup

Create a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Gemma models are hosted on Hugging Face and may require accepting Google’s model terms. After accepting access, sign in:

```bash
huggingface-cli login
```

## Dataset Format

Start with a CSV file at:

```text
data/raw/security_examples.csv
```

Use this shape:

```csv
text,label
"User reports a phishing email with a suspicious login link.",phishing
"Server logs show repeated failed SSH login attempts.",bruteforce
"A package dependency has a critical CVE.",vulnerability
```

The starter labels are:

```text
phishing
malware
vulnerability
bruteforce
benign
```

You can change them in `src/gemma_security/config.py`.

## Run The Notebook

Open:

```text
notebooks/01_gemma2_security_project.ipynb
```

Run the cells from top to bottom. The notebook is written as project documentation, so it explains why each step exists before showing the code.

## Run Training Later

After you have real data:

```bash
PYTHONPATH=src python -m gemma_security.train
```

## Try Inference Later

After you have a trained model saved in `models/gemma-security-classifier`, use:

```bash
PYTHONPATH=src python -m gemma_security.inference "Suspicious email asks the user to reset their bank password."
```

## Current Status

This is the first project version. It gives you the notebook, structure, and starter code needed to begin building. The next important step is adding a real security dataset.
