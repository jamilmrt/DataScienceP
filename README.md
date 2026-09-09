# CWHDS: Data Science Learning Repository

CWHDS is a hands-on Python data science repository containing Jupyter notebooks, practice datasets, and small scripts. The material progresses from Python data handling and cleaning to visualization, feature engineering, machine learning, and introductory web scraping.

The repository is intended for learning, experimentation, and portfolio development. Most work is notebook-based and can be run interactively in Jupyter or VS Code.

## Contents

- NumPy fundamentals and array operations
- Pandas data manipulation, cleaning, transformation, grouping, and SQL workflows
- Matplotlib and Seaborn visualization techniques
- Web scraping with Beautiful Soup
- Exploratory data analysis and missing-data practice
- Scikit-learn preprocessing, pipelines, feature scaling, and model training
- Beginner machine learning projects using datasets such as Iris, Titanic, housing, placement, and crime data
- Recommendation-style exercises using JSON data

## Project Directory

```text
CWHDS/
├── 01Introduction.ipynb              # Introduction to the repository
├── 02data_Cleaning.ipynb              # Data cleaning practice
├── 03_people_you_may_know.ipynb       # Recommendation-style exercise
├── 04_pages_you_might_like.ipynb      # Recommendation-style exercise
├── Handling Missing Data.ipynb        # Missing-data practice
├── prk.ipynb                          # Additional practice notebook
├── 100_Days_ML/                       # Exploratory data analysis exercises
├── CoderOfBangalore/                  # Coder of Bangalore analysis
├── Data_Collection_Technique/         # Web scraping and HTML examples
├── Data_Visualisation/                # Matplotlib and Seaborn notebooks
├── Matplotlib/                        # Matplotlib practice materials
├── Messy Crime Dataset/               # Crime data cleaning and analysis
├── Numpy/                             # NumPy fundamentals
├── Pandas/                            # Pandas analysis and data cleaning
├── Scikit_Learn/                      # Machine learning and preprocessing projects
├── ScikitLearn/                       # Additional preprocessing and pipeline practice
├── *.json, *.txt                      # Sample structured and text data
└── *.csv                              # Practice datasets
```

### Directory Highlights

| Directory | Focus |
| --- | --- |
| `Numpy/` | Arrays, indexing, data types, and broadcasting |
| `Pandas/` | Selection, cleaning, transformation, reshaping, aggregation, merging, and SQL |
| `Data_Visualisation/` | Bar charts, pie charts, histograms, scatter plots, subplots, and Seaborn |
| `Data_Collection_Technique/` | HTML parsing and Beautiful Soup web scraping |
| `Scikit_Learn/` | Imputation, categorical data, feature scaling, pipelines, and ML projects |
| `ScikitLearn/` | Additional transformers, scaling methods, pipelines, and perceptron exercises |
| `Messy Crime Dataset/` | Cleaning and exploring incident-level crime data |
| `100_Days_ML/` | Univariate, bivariate, and multivariate exploratory analysis |

## Recommended Learning Path

1. Start with `01Introduction.ipynb` and `02data_Cleaning.ipynb`.
2. Study NumPy fundamentals in `Numpy/`.
3. Continue with the core Pandas notebooks in `Pandas/`.
4. Practice charts and visual analysis in `Data_Visualisation/` and `Matplotlib/`.
5. Explore web scraping examples in `Data_Collection_Technique/`.
6. Learn preprocessing and pipelines in `Scikit_Learn/` and `ScikitLearn/`.
7. Finish with the applied analysis and machine learning projects.

## Getting Started

### Prerequisites

- Python 3.9 or newer
- Jupyter Notebook, JupyterLab, or VS Code with the Jupyter extension

### Installation

Create and activate a virtual environment, then install the commonly used packages:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy matplotlib seaborn scikit-learn beautifulsoup4 openpyxl
```

### Run the Notebooks

From the repository root, start Jupyter:

```bash
jupyter notebook
```

Open a notebook, run the cells in order, and adjust the input paths when working from a different directory.

## Data and Generated Files

The repository includes CSV, JSON, TXT, XLSX, and serialized model files for practice. Some notebooks create derived datasets or trained models such as `model.pkl`. Generated outputs should be treated as learning artifacts rather than production assets.

The `.venv/`, `.ipynb_checkpoints/`, and IDE metadata directories are local development files and are not part of the learning content.

## Project Status

This is an evolving personal learning repository. File names and notebook organization reflect the progression of experiments, so occasional duplicates and unfinished practice files are expected.

## Future Improvements

- Add a consistent environment file such as `requirements.txt`.
- Standardize notebook and dataset naming.
- Add reusable Python modules for repeated data preparation steps.
- Include clearer conclusions and visual summaries in analysis notebooks.
- Add tests for reusable preprocessing and machine learning utilities.
