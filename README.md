# Personalized Low-Invasive Evaluation of Chronic Endometritis

## Project Description

This project aims to develop machine learning-based predictive models for the low-invasive diagnosis of Chronic Endometritis (CE) in premenopausal women. The dataset used includes a comprehensive set of clinical, instrumental, and laboratory parameters associated with CE, collected from 108 non-obese women. The purpose of this research is to provide an alternative to traditional invasive diagnostic methods, offering a more accessible approach for early detection and prevention of CE.

### Key Features:
- Machine learning models trained using data from clinical history, ultrasound, and laboratory tests.
- Focus on low-invasive parameters such as adiponectin, SHBG, FSH, and other relevant biomarkers.
- Development of five predictive models with varying degrees of complexity and accuracy.

For more details on the scientific context and findings, please refer to our publication: [Link to MDPI article].

## Data

This project utilizes the following dataset:

1. **Dataset Description**: The dataset includes clinical, instrumental, and laboratory data for 108 women, of which 44 were diagnosed with Chronic Endometritis (CE) and 64 were healthy controls. The dataset includes:
   - Clinical data (e.g., menstrual cycle, spontaneous abortions)
   - Laboratory results (e.g., CRP, IL-1, IL-6, adiponectin)
   - Ultrasound results (e.g., endometrial thickness, uterine fibroids)
   - Hormonal profile (e.g., FSH, LH, estradiol, testosterone)
   
2. **Data Source**: The dataset is publicly available in the IEEE DataPort repository:
   - [IEEE DataPort Repository Link](https://www.ieee-dataport.org) [IEEE DataPort Repository Link]
   
3. **Data Access**: The dataset can be downloaded as a `.csv` file directly from the repository. Please refer to the specific dataset entry on IEEE DataPort for further instructions.

## Requirements

To run the project and reproduce the results, the following dependencies are required:

- **Programming Language**: Python 3.12.2
- **Runtime environment**: Jupyter Notebook
- **Libraries used**:
  - Pandas 2.3.2 — for structured data analysis and manipulation (Data-Frame operations)
  - NumPy 2.2.0 — for multi-dimensional array computations and mathe-matical operations
  - Missingno 0.5.2 — for visualizing missing data patterns in datasets
  - Matplotlib 3.10.5 — for creating basic static visualizations
  - Seaborn 0.13.2 — for advanced statistical plotting
  - SciPy 1.16.1 — for statistical analysis and mathematical functions
  - Miceforest 6.0.3 — for multiple imputation by chained equations (MICE) to handle missing values
  - Imbalanced-learn 0.14.0 — for handling class imbalance using SMOTENC algorithm
  - Scikit-learn 1.7.1 — for data preprocessing, model building, and valida-tion
  - SHAP 0.48.0 — for model interpretability and explanation of predictions
  - Mlxtend 0.23.4 — for machine learning tools, including feature selection
  - XGBoost 3.0.4 — for gradient boosting machine learning models
  - MLstatkit 0.1.9 — for performing DeLong’s test to compare ROC-AUC of two models

## Installation

Follow the steps below to set up and run the project locally:

1. Clone the repository:

   ```bash
   git clone https://github.com/baintuyevtimur/Chronic-Endometritis-ML-Modeling.git
   cd Chronic-Endometritis-ML-Modeling
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook `notebook_CE.ipynb` to start using the machine learning models:

   ```bash
   jupyter notebook notebook_CE.ipynb
   ```

## Project Structure

The project is organized as follows:

```
Chronic-Endometritis-ML-Modeling
│
├── notebook_CE.ipynb          # Jupyter notebook for model training and evaluation
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
└───images                    # Visualizations
    ├── confusion_matrices.png
    ├── model_comparison.png
    ├── pr_aucs.png
    ├── roc_aucs.png
    ├── shap_plots_m1_bar.png
    ├── shap_plots_m1_beeswarm.png
    ├── shap_plots_m2_bar.png
    ├── shap_plots_m2_beeswarm.png
    ├── shap_plots_m3_bar.png
    ├── shap_plots_m3_beeswarm.png
    ├── shap_plots_m4_bar.png
    ├── shap_plots_m4_beeswarm.png
    ├── shap_plots_m5_bar.png
    └── shap_plots_m5_beeswarm.png
```

## Usage

To use the models, load the dataset and execute the model-building and evaluation process in the `notebook_CE.ipynb`. The notebook guides you through the necessary steps:

1. **Data Preprocessing**: Missing data imputation, feature scaling, and class balancing using SMOTE.
2. **Model Training**: Train gradient boosting models on the dataset using the scikit-learn and XGBoost libraries.
3. **Model Evaluation**: Evaluate models based on metrics such as ROC-AUC, Precision, Recall, and SHAP feature importance.
