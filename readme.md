# Student Performance Analysis

## Description
Student Performance Analysis is an end-to-end machine learning and data science project designed to analyze and predict student performance. This project includes a comprehensive workflow encompassing data ingestion, transformation, model training, analysis, and deployment.

## Features
- **Custom Exceptions & Logging**: Implementation of custom exceptions and logging mechanisms for better traceability.
- **Data Ingestion & Transformation**:
  - The `src/component` folder contains the core code for data processing.
  - Utilizes a pipeline approach for data transformation, including imputation, scaling, and encoding.
  - The preprocessor uses `ColumnTransformer` to combine numerical and categorical data processing pipelines.
  - The transformed data is saved along with the preprocessor for further use.
- **Model Training**:
  - Tests a variety of machine learning algorithms from scikit-learn, along with hyperparameter tuning.
  - Performance evaluation based on R2 score, with the best model being saved as a `.pkl` file in `artifact/`.
- **Data Analysis & Model Training Comprehension**:
  - Detailed data analysis and exploration conducted in Jupyter notebooks within the `notebook/` folder.
  - Includes a notebook demonstrating the model training process.
- **Prediction Pipeline**:
  - Automated prediction script (`predict_pipeline.py`) under `src/pipeline` for efficient model application.
- **Deployment**:
  - The model is deployed using Flask in `application.py`.
  - Includes basic HTML files for the backend setup.

## Technologies Used
- Python Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `catboost`, `xgboost`, `dill`, `flask`, `pickle`.

## Acknowledgements
Special thanks to [Krish Naik](https://www.youtube.com/@krishnaik06) for his invaluable tutorial, which inspired and guided the development of this project.
