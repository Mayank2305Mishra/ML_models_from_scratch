# import all of the libraries needed for training the models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from ml_metrics_eval import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
# Creating a class for Linear Regression

class LinearRegression:
    def __init__(self, alpha=0.01, epochs=1000):
        self.alpha = alpha
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.J_history = []
    def predict(self, x):
        return np.dot(x, self.weights) + self.bias
    def cost_function(self, x , y):
        m = x.shape[0]
        cost = 0
        for i in range(m):
            err = (np.dot(x[i], self.weights) + self.bias) - y[i]
            cost += err**2
        return cost/(2*m)
    
    def gradient_calculation( self , x , y):
        m = x.shape[0]
        dj_dw = np.zeros(self.weights.shape)
        dj_db = 0
        for i in range(m):
            err = (np.dot(x[i], self.weights) + self.bias) - y[i]
            for j in range(len(self.weights)):
                dj_dw[j] += err * x[i][j]
            dj_db += err
        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db
    def gradient_descent(self, x, y):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        for epoch in range(self.epochs):
            dj_dw, dj_db = self.gradient_calculation(x, y)
            self.weights -= self.alpha * dj_dw
            self.bias -= self.alpha * dj_db
            cost = self.cost_function(x, y)
            self.J_history.append(cost)
            if epoch% math.ceil(self.epochs / 10) == 0 :
                print(f"Epoch {epoch}: Cost {cost}")
        return self.weights, self.bias
    def fit(self, x, y):
        self.weights, self.bias = self.gradient_descent(x, y)
        return self.weights, self.bias
    def plot_cost(self):
        plt.plot(range(self.epochs), self.J_history)
        plt.xlabel("No. of iterations (Epochs)")
        plt.ylabel("Cost")
        plt.title("Cost vs iterations")
        plt.show()
    def plot_regression_line(self, x, y):
        plt.scatter(y, y, color='blue', label='Data points')
        plt.scatter(y, self.predict(x), color='red', label='Regression line')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Linear Regression Fit")
        plt.legend()
        plt.show()
    def evaluation_metrics(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"R2 Score: {r2}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Mean Absolute Error: {mae}")
        return r2, mse, rmse, mae
    def _model_summary(self):
        print("Model Summary:")
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.bias}")
        print(f"Learning Rate: {self.alpha}")
        print(f"Epochs: {self.epochs}")
        
class PolynomialRegression(LinearRegression):
    def __init__(self, degree=2, alpha=0.01, epochs=1000):
        super().__init__(alpha, epochs)
        self.degree = degree
    def polynomial_features_multi(self,x, degree):
        n_samples, n_features = x.shape
        x_poly = [np.ones(n_samples)]
        for d in range(1, degree + 1):
            for feature_indices in self.generate_combinations(n_features, d):
                feature = np.prod(x[:, feature_indices], axis=1)
                x_poly.append(feature)
        return np.column_stack(x_poly)
    
    def generate_combinations(self,n_features, degree):
        if degree == 1:
            return [[i] for i in range(n_features)]
        combinations = []
        for i in range(n_features):
            for sub_combination in self.generate_combinations(n_features, degree - 1):
                if i <= sub_combination[0]:
                    combinations.append([i] + sub_combination)
        return combinations
    
    def fit(self, x, y):
        x_poly = self.polynomial_features_multi(x, self.degree)
        self.weights, self.bias = self.gradient_descent(x_poly, y)
        return self.weights, self.bias
    def predict(self, x):
        x_poly = self.polynomial_features_multi(x, self.degree)
        return np.dot(x_poly, self.weights) + self.bias
    def plot_cost(self):
        return super().plot_cost()
    def plot_regression_line(self, x, y):
        plt.scatter(y, y, color='blue', label='Data points')
        plt.scatter(y, self.predict(x), color='red', label='Regression line')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Polynomial Regression Fit")
        plt.legend()
        plt.show()
    def evaluation_metrics(self, y_true, y_pred):
        return super().evaluation_metrics(y_true, y_pred)