# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:52:21 2023

@author: nichi
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.graphics.gofplots import qqplot


# Disable warning related to chaining assignments
pd.options.mode.chained_assignment = None

# Path of dataset
gameKaggle = pd.read_csv(r"C:\GroupProject\Dataset\games.csv")

# Assign each column in dataset a new header value
gameKaggle.columns = ["Primary Key", "Title", "Release Date", "Team", "Rating", "Times Listed", "Number of Reviews", 
                      "Genres", "Summary", "Reviews", "Plays", "Playing", "Backlogs", "Wishlist"]


# Ignore columns containing null values
noNullgameKaggle = gameKaggle.dropna()

# Convert values with 'K' to floats
noNullgameKaggle.loc[:, 'Times Listed'] = noNullgameKaggle['Times Listed'].str.replace('K', '').astype('float')*1000
noNullgameKaggle.loc[:, 'Number of Reviews'] = noNullgameKaggle['Number of Reviews'].str.replace('K', '').astype('float')*1000
noNullgameKaggle.loc[:, 'Plays'] = noNullgameKaggle['Plays'].str.replace('K', '').astype('float')*1000
noNullgameKaggle.loc[:, 'Playing'] = noNullgameKaggle['Playing'].str.replace('K', '').astype('float')*1000
noNullgameKaggle.loc[:, 'Backlogs'] = noNullgameKaggle['Backlogs'].str.replace('K', '').astype('float')*1000
noNullgameKaggle.loc[:, 'Wishlist'] = noNullgameKaggle['Wishlist'].str.replace('K', '').astype('float')*1000


# Assign variables for Rating
X_Rating = noNullgameKaggle.drop(['Primary Key', 'Title', 'Release Date', 'Team', 'Rating', 'Times Listed', 'Number of Reviews', 
                     'Genres', 'Summary', 'Reviews', 'Plays'], axis = 1)

Y_Rating = noNullgameKaggle[["Rating"]].astype('int')

# Split dataset rating
X_Rating_train_valid, X_Rating_test, Y_Rating_train_valid, Y_Rating_test = train_test_split(X_Rating, Y_Rating, test_size=0.25, random_state=42)
X_Rating_train, X_Rating_valid, Y_Rating_train, Y_Rating_valid = train_test_split(X_Rating_train_valid, Y_Rating_train_valid, test_size=0.25, random_state=42)

# Reshape target
Y_Rating_train = np.ravel(Y_Rating_train)

    

def RandomForest():
    
    # Create a Random Forest Regression model
    # Number of decision trees in the random forest
    n_estimators = 100  
    
    # Maximum depth of each decision tree (None means unlimited)
    max_depth = None    
    
    # Random seed for reproducibility
    random_state = 42   
    
    # Generate model
    rf_regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        
    # Train the model
    rf_regressor.fit(X_Rating_train, Y_Rating_train)
    
    # Make predictions on the test set
    y_pred = rf_regressor.predict(X_Rating_test)
    
    
    # Display
    print("Random Forest")
    
    # Calculate the mean squared error
    mse = mean_squared_error(Y_Rating_test, y_pred)
    print("Mean Squared Error:", mse)        
    
    # Calculate the root mean squared error
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error:", rmse)
    
    # Calculate the mean absolute error
    mae = mean_absolute_error(Y_Rating_test, y_pred)
    print("Mean Absolute Error:", mae)
    
    # Calculate the r-squared value
    r_squared = r2_score(Y_Rating_test, y_pred)
    print("R-Squared: ", r_squared, "\n")


    # Create a DataFrame with actual and predicted values
    df = pd.DataFrame({'Actual': Y_Rating_test.values.flatten(), 'Predicted': y_pred})
    
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Create a heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap (Random Forest)')
    plt.show()
    


def LogisticReg(X_train, Y_train, X_test, Y_test, solver='liblinear', max_iter=100):
    
    # Create regressor object
    log_regressor = LogisticRegression(solver=solver, max_iter=max_iter)

    # Train the model
    log_regressor.fit(X_train, Y_train)
    
    # Make predictions on the test set
    y_pred = log_regressor.predict(X_test)
    
    
    # Display
    print("Logistic Regression")
    
    # Calculate the mean squared error
    mse = mean_squared_error(Y_test, y_pred)
    print("Mean Squared Error:", mse)        
    
    # Calculate the root mean squared error
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error:", rmse)
    
    # Calculate the mean absolute error
    mae = mean_absolute_error(Y_Rating_test, y_pred)
    print("Mean Absolute Error:", mae)
    
    # Calculate the r-squared value
    r_squared = r2_score(Y_test, y_pred)
    print("R-Squared: ", r_squared, "\n")
    
    
    # Create a DataFrame with actual and predicted values
    df = pd.DataFrame({'Actual': Y_Rating_test.values.flatten(), 'Predicted': y_pred})
    
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Create a heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap (Logistic Regression)')
    plt.show()
    
    
    
# Linear Regression
def LinearReg(X_train, Y_train, X_test, Y_test):
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Feature Variables
    X = np.random.rand(100, 1) 
    
    # Target Variable
    y = 2 + 3 * X + np.random.randn(100, 1)
    
    
    # Create a LinearRegression object
    lin_regressor = LinearRegression()
    
    # Train the linear regression model
    lin_regressor.fit(X_train, Y_train)
    
    # Make predictions on the test set
    y_pred = lin_regressor.predict(X_test)
    
    
    # Display
    print("Linear Regression")
    
    # Print the learned intercept and slope
    print("Intercept:", lin_regressor.intercept_)
    print("Slope:", lin_regressor.coef_[0], "\n")
    
    # Calculate the mean squared error
    mse = mean_squared_error(Y_test, y_pred)
    print("Mean Squared Error:", mse)        
    
    # Calculate the root mean squared error
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error:", rmse)
    
    # Calculate the mean absolute error
    mae = mean_absolute_error(Y_Rating_test, y_pred)
    print("Mean Absolute Error:", mae)
    
    # Calculate the r-squared value
    r_squared = r2_score(Y_test, y_pred)
    print("R-Squared: ", r_squared)
    
    
    # Create a DataFrame with actual and predicted values
    df = pd.DataFrame({'Actual': Y_Rating_test.values.flatten(), 'Predicted': y_pred})
    
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Create a heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap (Linear Regression)')
    plt.show()
    
    
    
# Q-Q Plot
def QQPlot(Y_test):
    
    # Generate plot
    qqplot(Y_test, line = 's')
    plt.title("Q-Q Plot")
    plt.show()
    
    
    
    
# Main 
def main():
    RandomForest()
    LogisticReg(X_Rating_train, Y_Rating_train, X_Rating_test, Y_Rating_test, solver='liblinear', max_iter=100)
    LinearReg(X_Rating_train, Y_Rating_train, X_Rating_test, Y_Rating_test)
    QQPlot(Y_Rating_test)
    
main()