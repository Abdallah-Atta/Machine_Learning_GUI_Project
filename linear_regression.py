import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from tkinter import messagebox

def Linear_Regression_function(df,targetCoulmn):

    # Assuming 'X' contains features and 'y' contains the target variable
    X = df.drop(targetCoulmn, axis=1)
    
    scaler = MinMaxScaler()
    df[[targetCoulmn]] = scaler.fit_transform(df[[targetCoulmn]])

    y = df[[targetCoulmn]]

    # Split the df into training and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)


    # Evaluate the model using Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    messagebox.showinfo("Mean Squared Error:", f"MSE: {mse:.2f}")

    # # Print the coefficients and intercept of the model
    # print("Coefficients:", model.coef_)
    # print("Intercept:", model.intercept_)
