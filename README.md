# XAI Strategy 2a: Explainable AI on RDF Dataset

## Project Overview
This project implements Strategy 2a for explainable AI using RDF datasets by:
1. Converting RDF triples into a structured tabular format
2. Training classical machine learning models
3. Explaining model predictions using SHAP

## Dataset
- **Dataset**: AM (Algebraic Molecular) dataset
- **Source**: [AM RDF Dataset from DGL](https://data.dgl.ai/dataset/rdf/am-hetero.zip)
- **Task**: Multi-class classification
- **Format**: RDF triples converted to tabular format

## Installation

### ✅ Recommended: Run on Google Colab

This project was developed entirely in **Google Colab**, and we recommend running it there for the smoothest experience.

Launch the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sabin360/AM-xai/blob/main/Mini_project_AM.ipynb)

Once open in Colab:
- Upload the RDF `.nt` files manually, **or**
- Mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 🖥️ Optional: Run Locally (Advanced)

#### Prerequisites
- Python 3.8+
- pip

#### Steps
1. Clone this repository:
```bash
git clone https://github.com/Sabin360/AM-xai.git
cd AM-xai
```

2. (Optional) Create and activate a virtual environment:
```bash
python -m venv xai_env
# Windows:
xai_env\Scripts\activate
# Linux/Mac:
source xai_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:
```bash
pip install pandas rdflib scikit-learn shap lime matplotlib seaborn
```

4. Run the notebook:
```bash
jupyter notebook Mini_project_AM.ipynb
```

---

## Project Structure

```
AM-xai/
├── data/                        # (Optional) RDF data
├── results/                     # Output files
│   ├── converted_tabular_data.csv
│   ├── X_train.npy / X_test.npy
│   ├── y_train.npy / y_test.npy with two hop neighbourhood
│   ├── best_model.pkl
│   ├── feature_names.pkl
│   ├── label_encoder.pkl
│   ├── shap_values.pkl
│   ├── confusion_matrix.png
│   └── shap_summary.png
├── Mini_project_AM.ipynb        # Main notebook (Colab)
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## Methodology

### 1. RDF to Tabular Conversion
- Each RDF **subject** becomes a row
- Each **predicate** becomes a column
- Each **object** becomes a cell value
- Missing values handled
- One-hot encoding for categorical features

### 2. Machine Learning Pipeline
- Feature selection using variance threshold
- Model training:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Evaluation using accuracy, confusion matrix, classification report

### 3. Explainability (XAI)
- **SHAP**: Global and local explanations
- Feature importance plots
- SHAP summary and individual-level insights

---

## Results

The notebook outputs:
- Model performance comparison
- Confusion matrix visualizations
- Feature importance (SHAP)
- SHAP summary plots
- Model explanations per class/prediction

---

## Key Features

### ✅ Strategy 2a Implemented
- RDF data to tabular conversion
- Classical ML pipeline
- Model evaluation & explanation with SHAP
- Clean modular pipeline using pandas, scikit-learn, SHAP

### 🔍 Explainability Highlights
- SHAP global feature rankings
- Instance-level SHAP values
- Visualizations with seaborn/matplotlib

---

## Dependencies

See `requirements.txt`. Key libraries:
- `rdflib`: RDF parsing
- `pandas`: Tabular manipulation
- `scikit-learn`: ML models
- `shap`: Explainability
- `matplotlib`, `seaborn`: Plotting
- `lime` (optional): Additional explanation support

---

## Troubleshooting

### Common Issues
- **Memory Errors**: Use smaller `.nt` files or filter large RDF graphs
- **No output in Colab**: Ensure you run all code cells in sequence
- **SHAP errors**: Install full plotting libraries: `matplotlib`, `seaborn`
- **File Not Found**: Use Colab file upload or `drive.mount()` properly

---

## Contributing

1. Fork this repo
2. Create a feature branch
3. Push changes and submit a Pull Request

---

## License

This project is for academic use as part of the Explainable AI (XAI) mini project.

---

## Contact

For questions or academic queries, contact:

- 📧 Dr. Stefan Heindorf *(Professor)*
  📧 Sabin Pandey *(Student)* 
- 📧 Nithya Susan George *(Student)*

---

## Acknowledgments

- Dataset by [DGL (Deep Graph Library)](https://www.dgl.ai/)
- [SHAP](https://github.com/slundberg/shap) library for explanations
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
