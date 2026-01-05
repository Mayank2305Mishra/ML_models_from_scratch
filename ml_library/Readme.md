# Winter of Code - 2025/26

> Cyberlabs - Machine Learning Division ( IIT (ISM) Dhanbad )

### About
The winter of code 2025/26 was an online hackathon where we were supposed to make machine learning alogrithms from scratch, we could use libraries like `Numpy` , `Pandas` and `Matplotlib`.

> This `Readme.md` contains information about the alogrithms implemented during the WOC -2025/26

### Algorithms and models

The alogrithms and models build were -

- [ ] Linear Regression
- [ ] Logistic Regression
- [ ] Polynomial Regression
- [ ] N - layer Neural Network
- [ ] K - Nearest Neighbours
- [ ] K Means Clustering 
- [ ] Initilizers - He, Xavier ( NN )
- [ ] Optimizers - Gradient Descent and Adam ( NN )

All of these alogrithms were build from scratch 


---

### Folder and File structure

- `./datasets` - conatains all of the datasets (train + test) provided for the alogrithms.
-  `./outputs` - contains all of the outputs of test dataset after the predictions were made from the models.
- `ml_library.py` - python library which have all of the alogrithms and important function, all coded from scratch.

- `ml_metrics_eval.py` - python library which contains the functions for evaluation metrics of models.

- `*.ipynb` - python notebook where the all of the models were trained and tested on the dataset provided.
---

```
All of the alogrithms were implemented using the class method 
from Object Oriented Programming (OOPs).
```
---
### Models and Algorithms 

1. <details>
   <summary><b>Linear Regression</b></summary>

   <br/>

   Linear Regression is one of the most fundamental machine learning models.  
   It tries to fit the maximum number of data points using a straight line,
   plane, or hyperplane based on the shape of the input features.
   > Here we create a Linear Regression class which have hyper-parameter like alpha (learning rate , number of iterations).

   ```python
   class LinearRegression:
       def __init__(self, alpha=0.01, iters=1000):
           self.alpha = alpha
           self.iters = iters
           self.weights = None
           self.bias = None
           self.J_history = []
    ```
    > 1. Cost Function : This function calculates the cost/ loss function for the model. Our goal is to minimize this function by changing values of w and b.
    
    `J(w,b) = 1/m * sum(x.w + b - y)**2 `

    ```python
    def cost_function(self, x , y):
        m = x.shape[0]
        cost = 0
        for i in range(m):
            err = (np.dot(x[i], self.weights) + self.bias) - y[i]
            cost += err**2
        return cost/(2*m)
    ```
    

</details>

