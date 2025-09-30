**Stock Price Prediction with GA-Optimized LSTM**  

This repository implements stock price prediction using classical models, deep learning, and a Genetic Algorithm (GA)-tuned LSTM.
The GA is used to optimize LSTM hyperparameters for improved forecasting performance.  

**Project Overview**  

**Goal**  
Predict future stock prices using machine learning and deep learning.  

**Models Implemented**  
Random Forest (baseline)  
Simple RNN  
LSTM (default configuration)  
LSTM (GA-optimized hyperparameters)  

**Key Idea**  
While LSTMs are powerful for sequential data, their performance depends heavily on hyperparameters.  
A Genetic Algorithm (GA) is used here to automatically tune:  
1.Number of units  
2.Number of layers  
3.Dropout rate  
4.Learning rate  
5.Batch size  
6.Sequence length  

**Methodology**  
The implementation begins with data preprocessing, where stock price data (Close prices) is loaded, normalized using MinMaxScaler, and transformed into sequences of historical prices suitable for supervised learning. The dataset is then split into training (70%), validation (15%), and test sets (15%) to ensure proper evaluation. For baseline models, Linear Regression and Random Forest are trained on flattened sequences, while Simple RNN and a default LSTM are trained using Keras with early stopping to prevent overfitting. The GA-LSTM model incorporates a Genetic Algorithm to optimize hyperparameters, using validation RMSE as the fitness metric. The GA employs tournament selection to choose parents, uniform crossover to mix hyperparameters, and several mutation strategies, including small Gaussian steps, heavy-tailed Cauchy jumps, and rare “super jumps” for exploration. Elitism preserves the best-performing individuals, and random injection maintains diversity in the population. Finally, all models are evaluated on the test set using RMSE and MAE, with predictions inverse-scaled to the original price units, and performance is visualized by comparing predicted versus actual prices.  

**Results**  
| Model               | Test RMSE | Test MAE  |
| ------------------- | --------- | --------- |
| Random Forest       | 32.359    | 30.768    |
| Simple RNN          | 10.474    | 9.426     |
| LSTM (default)      | 5.621     | 4.489     |
| **LSTM (GA-tuned)** | **2.983** | **2.320** |  
  
**Findings and Observations**  
The experimental results show that the GA-optimized LSTM outperforms all other models, achieving the lowest RMSE and MAE, which demonstrates the effectiveness of hyperparameter tuning via a genetic algorithm. The default LSTM performs better than the Simple RNN and classical models such as Linear Regression, indicating the advantage of LSTM’s ability to capture temporal dependencies in sequential data. In contrast, Random Forest performs poorly on this task, likely because it cannot inherently model sequential relationships. Overall, the findings confirm that GA-based optimization significantly enhances LSTM’s prediction accuracy and robustness compared to untuned deep learning models and traditional approaches.  
  
**Conclusion**  
The study concludes that combining LSTM with GA-driven hyperparameter optimization provides a highly effective approach for time series forecasting. GA-LSTM consistently delivers superior predictive performance, reducing both RMSE and MAE compared to baseline models. While default LSTM already captures temporal patterns better than classical models, the additional GA tuning further improves accuracy and generalization. These results highlight the value of metaheuristic optimization in enhancing deep learning models for sequential data prediction tasks.
