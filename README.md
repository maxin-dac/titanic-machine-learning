
# 🚢 Titanic: Survival Dynamics & Predictive Modeling

## 📖 Context
- **Source:** [Kaggle](https://www.kaggle.com/competitions/titanic/overview)
- **Datasets:** `train.csv` (891 passengers) & `test.csv` (418 passengers)

## 🎯 Objective
The goal of this project is twofold:
1. Perform a deep Exploratory Data Analysis (EDA) to transform historical passenger records into actionable operational intelligence. We identify and quantify the core drivers of survival (gender, class, family dynamics) and translate them into passenger personas and risk scores.
2. Build a robust, top-tier predictive model using advanced Feature Engineering and Ensemble Learning to predict survival on the unseen test set.

## 🔄 Notebook Structure & Workflow

| Phase | Description |
| :--- | :--- |
| **1. Data Quality & Imputation** | Strategic missing-value handling (e.g., imputing `Age` using median by `Pclass` & `Title`). |
| **2. Deep EDA & BI** | Univariate, Bivariate, and Multivariate analysis. Statistical validation (Chi-square, ANOVA) to prove feature significance. |
| **3. Advanced Feature Engineering** | Extracted latent variables: `Title` (from Name), `Deck` (from Cabin), `FamilySize`, `IsAlone`, `Ticket_Freq`, and `FarePerPerson`. |
| **4. Predictive Modeling** | Implemented an **Ensemble Voting Classifier** combining Random Forest, Gradient Boosting, SVM, and Logistic Regression. |
| **5. Validation & Submission** | Stratified 10-Fold Cross-Validation and generation of the final `submission.csv`. |

## 📊 Key Insights
* **The Gender & Class Interaction:** The "women and children first" protocol was applied asymmetrically. 1st/2nd-class females had near-universal survival (>90%), while 3rd-class females dropped to 50%. Class overrode gender for poorer women.
* **The "Goldilocks" Family Effect:** A clear inverted-U relationship. Solo travelers (30.4% survival overall) and large families ≥5 (22.3% survival) fared worst. Families of 2-4 had the highest survival rates (61.83% overall).
* **Deck Proximity:** Decks B-D (1st class, boat-deck adjacent) showed >75% survival. Deck G (lowest, 3rd class) dropped to 28%. The engineered `Deck` feature proved highly significant.
* **Fare as a Continuous Proxy:** Survival rises monotonically with fare quintile (Q1 = 21.8% → Q5 = 64.2%), capturing within-class hierarchy better than `Pclass` alone.

## 🧠 Machine Learning Approach
This project uses a **Soft Voting Ensemble Classifier** to maximize generalization and minimize variance:
* **Random Forest** (Handles non-linearities well)
* **Gradient Boosting** (State-of-the-art for tabular data)
* **Support Vector Machine** (SVM with RBF kernel, scaled)
* **Logistic Regression** (Linear baseline with regularization)

*The model weighs predictions, giving higher importance to Gradient Boosting and Random Forest, achieving a cross-validated accuracy of **76.79%**.*

## 🛠️ Tech Stack & Libraries

* **Language:** ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
* **Data Manipulation:** ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
* **Visualization:** ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-5B8DB8?style=flat)
* **Machine Learning:** ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) (`VotingClassifier`, `RandomForest`, `GradientBoosting`, `SVC`, `StratifiedKFold`)
* **Environment:** ![Jupyter Notebook](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=flat&logo=jupyter&logoColor=white) ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

🚀 Explore the Notebook
Want to see the code behind these insights?

* Clone this repo and run the local Jupyter Notebook.
* Download the pdf file on this repo.
* Check out the interactive, up-to-date version directly on [Kaggle](https://www.kaggle.com/code/maximendacleu/titanic-analysis).
