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
