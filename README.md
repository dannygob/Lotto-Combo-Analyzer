# Lotto-Combo-Analyzer

🧠 Lotto Combo Analyzer — Version 1.0

A modular tool for analyzing and predicting lottery number combinations using historical data, machine learning models, and association rule mining. The project is structured for clarity, reproducibility, and future expansion.

---

📦 Project Structure

Below is the point-by-point structure, with a description of each module and its purpose.

1. Loading and Preprocessing  
Objective: Read the CSV file, extract the number columns, sort the combinations, and prepare the data for analysis.

✓ Load the lotto_history.csv file (format: Sorteo, numero 1, numero 2, numero 3, numero 4, numero 5, numero 6)  
✓ Extract columns of type numero 1, numero 2, etc.  
✓ Sort each combination  
✓ Convert to numeric types  

2. Statistical Analysis  
Objective: Explore the frequency of numbers, total sum per draw, and variations between draws.

✓ Frequency of appearance of each number  
✓ Total sum per draw  
✓ Differences between consecutive draws  
✓ Classification of changes: up, down, equal  
✓ Descriptive statistics  
✓ Visualizations: histograms, boxplots, time series  

3. Clustering of Combinations  
Objective: Group similar combinations using KMeans to detect patterns.

✓ Application of KMeans  
✓ Elbow method to determine the optimal number of clusters  
✓ Visualization of clusters  

4. Preparing Sequences for Models  
Objective: Transform data into sequences for training sequential models such as LSTM.

✓ Converting combinations to sequences  
✓ Filtering out-of-range values (1–52)  
✓ Multiclass coding with to_categorical  

5. Modeling with LSTM  
Objective: Train an LSTM neural network to predict the next probable number.

✓ LSTM Model Definition  
✓ Hyperparameter Optimization with Optuna  
✓ Loss Function: Cross Entropy  
✓ K-Fold Cross-Validation  
✓ Metrics: MAE, MSE, RMSE, R², Log Loss, Top-k Accuracy, MAPE  

6. Comparison with Other Models  
Objective: Evaluate the performance of classic models such as Random Forest and XGBoost.

✓ Training Alternative Models  
✓ Metric Comparison  
✓ Results Visualization  

7. Full-Group Evaluation  
Objective: Calculate the percentage of correct guesses per full combination (group of 6 numbers).

✓ Comparison between predicted and actual combinations  
✓ Calculate exact and partial guesses  
✓ Metrics per group  

8. Multiple Simulations  
Objective: Generate several alternative combinations for a future lottery.

✓ Multiple Prediction Generation  
✓ Repeated Combination Filtering  
✓ Most Likely or Diverse Combination Selection  

9. Association Rules (Apriori)  
Objective: Discover frequent relationships between drawn numbers.

✓ Transaction Coding  
✓ Application of Apriori  
✓ Rule Generation with Support, Confidence, and Lift  
✓ Lift Distribution Visualization  

10. Custom Forecast  
Objective: Allow the user to generate a combination for a specific draw.

✓ Prompt the user to enter a draw number  
✓ Generate a unique combination for that draw  
✓ Check the generated combination against historical data  
✓ Display the forecasted combination and its evaluation  

---

📊 Dataset Format

The dataset must be a CSV file with the following columns:  
Sorteo, numero 1, numero 2, numero 3, numero 4, numero 5, numero 6

Each row represents a historical lottery draw with six numbers and a draw identifier.

---

📈 Evaluation Metrics

• MAE, MSE, RMSE, R²  
• Log Loss (multiclass)  
• Top-k Accuracy  
• MAPE (Mean Absolute Percentage Error)  
• Accuracy by full group (6/6, 5/6, 4/6)  
• Lift, support, confidence (Apriori rules)

---

🛠 Improvements Implemented

• Version control and logging added for traceability (`logs/run.log`)  
• Top-k Accuracy and MAPE included in model evaluation  
• Clear separation between analysis and prediction in the notebook  
• Shared helper functions grouped in `src/utils.py`  
• Interactive forecast feature: user can input a draw number to receive a personalized prediction

---

🚀 Getting Started

1. Clone the repository:
   git clone https://github.com/dannygob/Lotto-Combo-Analyzer.git  
   cd Lotto-Combo-Analyzer

2. Install dependencies:
   pip install -r requirements.txt

3. Run the notebook:
   jupyter notebook notebooks/Lotto_Combo_Analyzer.ipynb

---

📌 Version

Current version: 1.0  
See CHANGELOG.txt for details.

## 📁 Folder Structure

lotto-combo-analyzer/
├── data/
│ └── lotto_history.csv # Original or example dataset
├── notebooks/
│ └── Lotto_Combo_Analyzer.ipynb # Main notebook (version 1.0)
├── src/
│ ├── __init__.py
│ ├── preprocessing.py # Loading and cleaning functions
│ ├── analysis.py # Statistics and visualizations
│ ├── clustering.py # KMeans and cluster visualization
│ ├── modeling.py # LSTM + Optuna + metrics
│ ├── comparison.py # Random Forest / XGBoost
│ ├── evaluation.py # Metrics for the entire group
│ ├── simulation.py # Multiple simulations
│ └── association.py # Apriori and association rules
├── results/
│ └── predictions.csv # Generated combinations
├── README.md # Project description
├── requirements.txt # Required libraries
└── .gitignore # Files to exclude from version control



