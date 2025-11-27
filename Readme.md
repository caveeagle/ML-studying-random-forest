# Introduction

In this project, I test various families of models on a dataset, aiming to predict real estate prices.

## Project objective

To select the best-performing model for this task and determine its optimal hyperparameters.

## Additional information
In the file **`Technical Results.md`** you can find technical results of models training.
in the file ***`Files.md`*** you can find information about structure of directories, required Python dependencies.

# Approach

### Data selection: 
I prepared several datasets for different types of models. However, the best results were obtained with the dataset provided by Amin, which I use in all my models.

### Tested models: 
The following models were evaluated: KNN (as a baseline check), Random Forest, classical Gradient Boosting, boosted XGBoost (Extreme Gradient Boosting.), CatBoost model (boosting, optimized for categories), and simple neural networks (FNN) with embedding.

Models with the best results: **Classical Gradient Boosting**, **XGBoost**, and **FNN**.

For these models, parameter tuning was performed to achieve optimal results. They were also examined for overfitting. The difference between val_loss and train_loss remained within 10–20%.

# Conclusion
On this dataset, classical **Gradient Boosting** (GradientBoostingRegressor) provides the best performance and is recommended for implementation.

**XGBoost** produces slightly lower accuracy, while **neural networks** also perform well and can be used to enhance the model through an ensemble approach.

### **Gradient Boosting**  results:
The mean accuracy metric R2 are:
**0.810**

### With the given parameters:
`n_estimators=1200`
`max_depth=5`
`learning_rate=0.05`
`subsample=0.9`

### Next steps:
Using Gradient Boosting and neural networks, I plan to improve the model by introducing error modeling.



