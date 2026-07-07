# Titanic Survival Prediction - Machine Learning
<img width="950" height="200" alt="image" src="https://github.com/user-attachments/assets/e7c7389a-2c48-485c-81cc-9b99c78faa0d" />

## Context
- From [Kaggle](https://www.kaggle.com/competitions/titanic/overview)
- Dataset: test.csv, train.csv

## Objective
The goal of this project is to build a predictive model that answers the famous question: "What sorts of people were more likely to survive the Titanic disaster?" Using passenger data such as age, sex, class, fare, and family size, the model identifies key survival factors and generates predictions for the Kaggle test set.

## Notebook Structure & Workflow
* **Data Loading & Exploration:** Load train and test datasets, inspect structure, and identify missing values.
* **Exploratory Data Analysis (EDA):** Visualize relationships between features and survival (e.g., survival by sex and age).
* **Data Cleaning & Preprocessing:** Handle missing values (Age, Fare, Embarked), drop uninformative columns (Name, Ticket, Cabin), and encode categorical variables (Sex, Embarked).
* **Model Building:** Train a Random Forest Classifier and evaluate on a validation split.
* **Feature Importance Analysis:** Identify which features contributed most to predictions.
* **Submission Generation:** Predict on the test set and create the Kaggle submission file.

## Key Insights
* **Sex:** Females had a dramatically higher survival rate than males, consistent with the "women and children first" protocol.
* **Age:** Children (especially under 10) had better survival odds compared to adults. Young adults (18–30) formed a large portion of the casualties.
* **Socio-economic Status:** Higher fare and lower passenger class (Pclass 1) were strongly associated with higher survival probability.
* **Model Performance:** The Random Forest achieved approximately **82.7%** accuracy on the validation set, providing a solid baseline for this classic dataset.

## Feature Importance
* **Sex:** Most influential feature.
* **Fare / Pclass:** Strong indicators of socio-economic status and access to lifeboats.
* **Age:** Relevant, particularly distinguishing children from adults.

## Model Insights & Key Findings
* **Women, Children, and the Wealthy** were significantly more likely to survive.
* The analysis confirms historical accounts: survival was heavily influenced by **gender, age, and class** rather than random chance.
* The Random Forest model effectively captured these patterns, with clear interpretability through feature importance.

## Tech Stack & Libraries

* **Language:** ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

* **Data Manipulation:** ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

* **Visualization:** ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-5B8DB8?style=flat)

* **Machine Learning:** ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) (`RandomForestClassifier`, `train_test_split`, `accuracy_score`)

* **Environment:** ![Jupyter Notebook](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=flat&logo=jupyter&logoColor=white) ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

*To explore the detailed code, feel free to download the notebook file on this repo, or check out my [Kaggle](https://www.kaggle.com/code/maximendacleu/titanic-analysis]).*
