# Lotto-Combo-Analyzer
Lottery Evaluation
🧠 Lotto Combo Analyzer
Below is the point-by-point structure, with a description of each module and its purpose.

📦 Project Structure
1. Loading and Preprocessing
Objective: Read the CSV file, extract the number columns, sort the combinations, and prepare the data for analysis.

Load the lotto_history.csv file (format: draw, number 1, number 2, number 3, number 4, number 5, number 6)

Extract columns of type number 1, number 2, etc.

Order each combination

Convert to numeric types

2. Statistical Analysis
Objective: Explore the frequency of numbers, total sum per draw, and variations between draws.

Frequency of appearance of each number

Total sum per draw

Differences between consecutive draws

Classification of changes: up, down, equal

Descriptive statistics

Visualizations: histograms, boxplots, time series

3. Clustering of combinations
Objective: Group similar combinations using KMeans to detect patterns.

Application of KMeans

Elbow method to determine the optimal number of clusters

Visualization of clusters

4. Preparing sequences for models
Objective: Transform data into sequences for training sequential models such as LSTM.

Converting combinations to sequences

Filtering out-of-range values ​​(1–52)

Multiclass coding with to_categorical

5. Modeling with LSTM
Objective: Train an LSTM neural network to predict the next probable number.

LSTM Model Definition

Hyperparameter Optimization with Optuna

Loss Function: Cross Entropy

K-Fold Cross-Validation

Metrics: MAE, MSE, RMSE, R², Log Loss

6. Comparison with Other Models
Objective: Evaluate the performance of classic models such as Random Forest and XGBoost.

Training Alternative Models

Metric Comparison

Results Visualization

7. Full-Group Evaluation
Objective: Calculate the percentage of correct guesses per full combination (group of 6 numbers).

Comparison between predicted and actual combinations

Calculate exact and partial guesses

Metrics per group

8. Multiple Simulations
Objective: Generate several alternative combinations for a future lottery.

Multiple Prediction Generation

Repeated Combination Filtering

Most Likely or Diverse Combination Selection

9. Association Rules (Apriori)
Goal: Discover frequent relationships between drawn numbers.

Transaction Coding

Application of Apriori

Rule Generation with Support, Confidence, and Lift

Lift Distribution Visualization

10. Custom Forecast
Goal: Allow the user to generate a combination for a specific draw.

Interactive Draw Number Entry

Unique Combination Generation

Check Against History

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



## 🚀 Getting Started

1. Clone the repository:
   git clone https://github.com/your-username/lotto-combo-analyzer.git
   cd lotto-combo-analyzer

2. Install dependencies:
   pip install -r requirements.txt

3. Run the main notebook or script:
   jupyter notebook notebooks/Lotto_Combo_Analyzer.ipynb

## 📊 Dataset
Place your historical lottery dataset in the `data/` folder. The file should be named `lotto_history.csv` and contain columns like `numero 1`, `numero 2`, ..., `numero 6`.

## 🧠 Models Used

- LSTM (Long Short-Term Memory)
- Random Forest Classifier
- XGBoost Classifier

## 📈 Metrics

- MAE, MSE, RMSE, R²
- Log Loss
- Top-k Accuracy
- Group-level prediction accuracy

## 📄 License

opensource
