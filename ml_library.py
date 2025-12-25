# import all of the libraries needed for training the models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from ml_metrics_eval import *
# Creating a class for Linear Regression

class LinearRegression:
    def __init__(self, alpha=0.01, iters=1000):
        self.alpha = alpha
        self.iters = iters
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
        for iter in range(self.iters):
            dj_dw, dj_db = self.gradient_calculation(x, y)
            self.weights -= self.alpha * dj_dw
            self.bias -= self.alpha * dj_db
            cost = self.cost_function(x, y)
            self.J_history.append(cost)
            if iter% math.ceil(self.iters / 10) == 0 :
                print(f"Epoch {iter}: Cost {cost}")
        return self.weights, self.bias
    def fit(self, x, y):
        self.weights, self.bias = self.gradient_descent(x, y)
        return self.weights, self.bias
    def plot_cost(self):
        plt.plot(self.J_history)
        plt.xlabel("No. of iterations")
        plt.ylabel("Cost")
        plt.title("Cost vs iterations")
        plt.show()
    def plot_regression_line(self, x, y):
        plt.scatter(x[:,0], y, color='blue', label='Data points')
        plt.scatter(x[:,0], self.predict(x), color='red', label='Regression Model')
        plt.xlabel("X-1st Column")
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
        print(f"Iterations: {self.iters}")
        
class PolynomialRegression(LinearRegression):
    def __init__(self, degree=2, alpha=0.01, iters=1000):
        super().__init__(alpha, iters)
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

class LogisticRegression:
    def __init__(self, alpha=0.01, iters=1000):
        self.alpha = alpha
        self.iters = iters
        self.weights = None
        self.bias = None
        self.J_history = []
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def cost_function(self, x, y):
        m = len(x)
        cost = 0
        for i in range(m):
            z = np.dot(x[i], self.weights) + self.bias
            f_wb = self.sigmoid(z)
            cost += - (y[i] * np.log(f_wb) + (1 - y[i]) * np.log(1 - f_wb))
        return cost/m
    def gradient_calculation(self, x, y):
        m = len(x)
        dj_dw = np.zeros(self.weights.shape)
        dj_db = 0
        for i in range(m):
            z = np.dot(x[i], self.weights) + self.bias
            f_wb = self.sigmoid(z)
            err = f_wb - y[i]
            for j in range(len(self.weights)):
                dj_dw[j] += err * x[i][j]
            dj_db += err
        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db
    def gradient_descent(self, x, y):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        for iter in range(self.iters):
            dj_dw, dj_db = self.gradient_calculation(x, y)
            self.weights -= self.alpha * dj_dw
            self.bias -= self.alpha * dj_db
            cost = self.cost_function(x, y)
            self.J_history.append(cost)
            if iter % math.ceil(self.iters / 10) == 0:
                print(f"Iteration {iter}: Cost {cost}")
        return self.weights, self.bias
    def fit(self, x, y):
        self.weights, self.bias = self.gradient_descent(x, y)
        return self.weights, self.bias
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred_proba = self.sigmoid(z)
        return (y_pred_proba >= 0.5).astype(int)
    def probability(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    def evaluation_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Confusion Matrix:\n{cm}")
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xlabel("Predicted") 
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        return accuracy, precision, recall, f1, cm
    def plot_cost(self):
        plt.plot(self.J_history)
        plt.xlabel("No. of iterations")
        plt.ylabel("Cost")
        plt.title("Cost vs iterations")
        plt.show()
    def plot_regression_line(self, x, y):
        plt.scatter(y, y, color='blue', label='Data points')
        plt.scatter(y, self.predict(x), color='red', label='Regression line')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Polynomial Regression Fit")
        plt.legend()
        plt.show()

class NeuralNetwork:
    def __init__(self ,learning_rate=0.01 ,epochs=1000 , loss_function='mse'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_function = loss_function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def sigmoid_derivative(self, z):
        a = self.sigmoid(z)
        return a * (1 - a)  
    def relu(self, z):
        return np.maximum(0, z)
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)
    def linear(self, z):
        return z    
    def linear_derivative(self, z):
        return np.ones_like(z)
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    def softmax_derivative(self, z):
        s = self.softmax(z)
        return s * (1 - s)
    def Sequential(self,*layers):
        return {
            'layers': layers,
            'parameters': {},
        }
    def Dense(self, units, activation='relu'):
        return {
            'units': units,
            'activation': activation
        }
    def initialize_parameters(self, model , input_dim):
        parameters = {}
        layer_dims = [input_dim]
        for layer in model['layers']:
            layer_dims.append(layer['units'])
        L = len(layer_dims)
        for l in range(1, L):
            parameters['w' + str(l)] = np.random.randn(layer_dims[l-1], layer_dims[l]) * 0.01
            parameters['b' + str(l)] = np.zeros((1, layer_dims[l]))
        model['parameters'] = parameters
        return model
    def forward_propagation(self, X, model):
        parameters = model['parameters']
        caches = {'a0': X}
        L = len(model['layers'])
        for l in range(1, L + 1):
            z = np.dot(caches['a' + str(l-1)], parameters['w' + str(l)]) + parameters['b' + str(l)]
            if model['layers'][l-1]['activation'] == 'relu':
                a = self.relu(z)
            elif model['layers'][l-1]['activation'] == 'sigmoid':
                a = self.sigmoid(z)
            elif model['layers'][l-1]['activation'] == 'linear':
                a = self.linear(z)
            elif model['layers'][l-1]['activation'] == 'softmax':
                a = self.softmax(z)
            caches['z' + str(l)] = z
            caches['a' + str(l)] = a
        return caches[f'a{L}'], caches  
    def backward_propagation(self, X, y, model, caches):
        m = X.shape[0]
        parameters = model['parameters']
        grads = {}
        L = len(model['layers'])
        aL = caches['a' + str(L)]
        if self.loss_function == 'categorical_cross_entropy' and model['layers'][L-1]['activation'] == 'softmax':
            dz = (aL - y) / m
        elif self.loss_function == 'binary_cross_entropy' and model['layers'][L-1]['activation'] == 'sigmoid':
            dz = (aL - y) / m
        elif self.loss_function == 'mse':
            dz = 2 * (aL - y) / m
            if model['layers'][L-1]['activation'] == 'sigmoid':
                dz = dz * self.sigmoid_derivative(caches['z' + str(L)])
            elif model['layers'][L-1]['activation'] == 'relu':
                dz = dz * self.relu_derivative(caches['z' + str(L)])
        else:
            dz = (aL - y) / m
            if model['layers'][L-1]['activation'] == 'sigmoid':
                dz = dz * self.sigmoid_derivative(caches['z' + str(L)])
            elif model['layers'][L-1]['activation'] == 'relu':
                dz = dz * self.relu_derivative(caches['z' + str(L)])
        for l in reversed(range(1, L + 1)):
            a_prev = caches['a' + str(l-1)]
            w = parameters['w' + str(l)]
            
            grads['dw' + str(l)] = np.dot(a_prev.T, dz)
            grads['db' + str(l)] = np.sum(dz, axis=0, keepdims=True)
            
            if l > 1:
                da_prev = np.dot(dz, w.T)
                z_prev = caches['z' + str(l-1)]
                
                if model['layers'][l-2]['activation'] == 'relu':
                    dz = da_prev * self.relu_derivative(z_prev)
                elif model['layers'][l-2]['activation'] == 'sigmoid':
                    dz = da_prev * self.sigmoid_derivative(z_prev)
                elif model['layers'][l-2]['activation'] == 'linear':
                    dz = da_prev
                elif model['layers'][l-2]['activation'] == 'softmax':
                    dz = da_prev * self.softmax_derivative(z_prev)
        
        return grads

    def update_parameters(self, model, grads):
        parameters = model['parameters']
        L = len(model['layers'])
        for l in range(1, L + 1):
            parameters['w' + str(l)] -= self.learning_rate * grads['dw' + str(l)]
            parameters['b' + str(l)] -= self.learning_rate * grads['db' + str(l)]
        model['parameters'] = parameters
        return model
    def fit(self, X, y, model):
        input_dim = X.shape[1]
        model = self.initialize_parameters(model, input_dim)
        for epoch in range(self.epochs):
            y_pred, caches = self.forward_propagation(X, model)
            grads = self.backward_propagation(X, y, model, caches)
            model = self.update_parameters(model, grads)
            if epoch % math.ceil(self.epochs / 10) == 0:
                if self.loss_function == 'mse':
                    loss = mean_squared_error(y, y_pred)
                elif self.loss_function == 'binary_cross_entropy':
                    loss = cross_entropy_loss(y, y_pred)
                elif self.loss_function == 'categorical_cross_entropy':
                    loss = categorical_cross_entropy_loss(y, y_pred)
                print(f"Epoch {epoch}: Loss {loss}")
        return model
    def predict(self, X, model):
        y_pred, _ = self.forward_propagation(X, model)
        return y_pred
    def evaluation_metrics(self, y_true, y_pred):
        if self.loss_function == 'mse':
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            print(f"R2 Score: {r2}")
            print(f"Mean Squared Error: {mse}")
            print(f"Root Mean Squared Error: {rmse}")
            print(f"Mean Absolute Error: {mae}")
            return mse, rmse, mae, r2
        elif self.loss_function in ['binary_cross_entropy', 'categorical_cross_entropy']:
            accuracy = accuracy_score(y_true, np.round(y_pred))
            precision = precision_score(y_true, np.round(y_pred), average='weighted')
            recall = recall_score(y_true, np.round(y_pred), average='weighted')
            f1 = f1_score(y_true, np.round(y_pred), average='weighted')
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            return accuracy, precision, recall, f1
def train_test_split(X, y, test_size=0.2):
    n_samples = len(X)
    n_test = int(test_size * n_samples)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def z_score_normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=0) #ddof=0 for population std , used in SKlearn
    X_normalized = (X - mean) / std
    return X_normalized