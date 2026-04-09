# A Robust Diagnostic System Leveraging Explicit Domain Knowledge and Learned Data Patterns

A hybrid AI diagnostic system that combines **OWL ontology + SWRL rules** with **machine learning classifiers** for medical disease diagnosis, achieving a **25% accuracy improvement** over standalone approaches.

## Overview

Traditional diagnostic systems rely on either expert-crafted rules or data-driven ML — each with limitations. This project fuses both:

1. **Rule-Based Inference** — OWL ontology models encode structured medical knowledge; SWRL rules detect symptom patterns and trigger deterministic diagnoses.
2. **ML Fallback** — When rule-based inference is inconclusive, a trained `DecisionTreeClassifier` provides probabilistic predictions.

This hybrid approach ensures high confidence on well-known cases while gracefully handling ambiguous inputs.

## Key Results

- **25% higher accuracy** compared to pure ML or pure rule-based baselines
- Robust handling of ambiguous/incomplete symptom sets
- Interpretable rule-based outputs with ML fallback for edge cases

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Ontology | OWL (Web Ontology Language) |
| Rules | SWRL (Semantic Web Rule Language) |
| Ontology Editor | Protégé |
| ML Model | Scikit-learn (DecisionTreeClassifier) |
| Language | Python |
| Dataset | Kaggle Medical Diagnosis dataset |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the ML Model

```bash
python ml/train_model.py
```

This trains a `DecisionTreeClassifier` on `Medical_diagnosis.csv` and saves `ml/model.joblib`.

### Run Diagnosis

```bash
python diagnosis/diagnose.py --symptoms Fever,Cough --tests blood_sugar=140
```

The system first attempts rule-based diagnosis via the ontology. If no definitive match is found, it falls back to the ML classifier.

## Project Structure

```
├── ontology/           # OWL ontology files and SWRL rules
├── ml/
│   ├── train_model.py  # Model training script
│   └── model.joblib    # Trained classifier
├── diagnosis/
│   └── diagnose.py     # CLI diagnostic tool
└── requirements.txt
```

## Authors

Akhil Puttabanthi — CVR College of Engineering
Rohan Mukka — CVR College of Engineering
Shaik Samad Rizwan — CVR College of Engineering

