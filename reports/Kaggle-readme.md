# Winter of Code 2025/26

This file contains the report of kaggle competition, Binary Buzz , which was a classification problem.

----
### Understanding the datasets

> The competition provided 2 files : `train.csv` and `test.csv`

#### train.csv
This file had a shape of (2000, 3074) :

`2000` - 2000 rows or datasamples were provides

`3074` - 3074 columns which inducled the training features, id column and the target column.

##### Key Highlights about the file (All of the features are collapsible and have code and a small summary) :

- <details> <summary><b> Loading the data </b></summary>
    Loading the `train.csv` dataset using pandas as storing it as a dataframe.


    ```python
        import pandas as pd
        df = pd.read_csv('train.csv')
    ```
    The dataset is stored as a dataframe using a vriable df.
   </details>

- <details> <summary><b>Duplicate columns</b></summary>   
     Checking for any duplicate columns , if any ( would drop them)

    ``` python
        df.duplicated().sum()
        #> 0
    ```
    No duplicate columns were found
  </details>

- <details> <summary><b>Splitting the dataset</b></summary>   
     Splitting the dataset into features and target variable / independent and dependedt variable

    ``` python
        X = df.drop(columns=['id','target'], axis = 1)
        y = df['target']
        # axis = 1 , operates horizontally (on columns)
        X.shape , y.shape
        #> (2000, 3072) (2000,)
    ```
    X - features variable and y - target variable.
  </details>
- <details> <summary><b>Train Test Split</b></summary>   
     Splitting the data into training and testing data chunks to avoid data leakage in the model which evaluating the models or potential overfitting of the model.

    ``` python
        from sklearn.model_selection import train_test_split
        X_train , X_val , y_train , y_val = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)

        X_train.shape , X_val.shape , y_train.shape , y_val.shape
        #> ((1600, 3072), (400, 3072), (1600,), (400,))
    ```
    Used sci-kit learn library's model_selection's feature train_test_split , to split the data.
  </details>

- <details> <summary><b>Correlation Matrix</b></summary>   
     Checking for highly correlated matrix and dropping them.

    ``` python
        corr = X_train.corr().abs() # Returns a square matrix with corr of each column with other columns
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)) # Calculating the upper half of the matrix
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)] # Droping columns with corr higher than 0.95 / 95%
        len(to_drop) #> 303
        X_train = X_train.drop(columns = to_drop, axis=1)
        X_val = X_val.drop(columns = to_drop, axis=1)
    ```
    Dropped almost 303 columns , which had correlation higher than 0.95.
  </details>

- <details> <summary><b>Outlier Detection</b></summary>   
     Outliers are datapoints which are very different from rest of the data / they don't fit the normal patterns. Using the InterQuartile Range (IQR) 

    ``` python
        Q1 = X.quantile(0.25) # 25 pecentile
        Q3 = X.quantile(0.75) # 75 percentile
        IQR = Q3 - Q1 

        outliers_count = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1).sum() # Outlier exists if the data point is beyond |IQR * 1.5|
        print("Outliers:", outliers_count)
        #>
    ```
    Did not remove the outliers because they dont affect tree models , which are distance independent. (eg: XGBoost, LightGBM, RandomForest, etc..)
  </details>

- <details> <summary><b>Standard Scaler</b></summary>   
    Using Standard Scaler to rescale the data such that all features lies in similar ranges and have mean 0 and standard deviation as 1.

    ``` python
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    ```
  </details>

--- 
#### Experiments and Trials

Most of my experiments were done to get rid of the high dimensionality of the data (Curse of Dimensionality).

> All of the below points are collapsible and can be opened by clicking them.

- <details> <summary><b>ANOVA</b></summary>   
    ANOVA stands for Analysis of Variance , it used to check how well a feature seprates a class , so we can select the k best features (based on best split) out of 3072

    ``` python
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=1000) # k could be varied
        X_train = selector.fit_transform(X_train, y_train)
        X_val = selector.transform(X_val)
    ```
    Tried to play around with different values of k and different models.
  </details>

- <details> <summary><b>PCAs - Principal Component Analysis</b></summary>   
    PCA combines many related features into fewer new features that capture most of the data’s variation. This was implemented to 

    ``` python
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.85) 
        X_train = pca.fit_transform(X_scaled)
        X_val = pca.transform(X_val)
    ```
    Tried PCA with a variance of 0.85 , did not get much better results with them.
  </details>

- <details> <summary><b>RFE-Recursive Feature Elimination</b></summary>   
    RFE is a feature selection method that repeatedly trains a model, removes the least important features, and keeps the best ones.

    ``` python
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000) # Even tried with RandomForest and XGBoost

        rfe = RFE(estimator=model, n_features_to_select=10)
        pca = PCA(n_components=0.85) 
        X_train = rfe.fit_transform(X_train, y_train)
        X_val = rfe.transform(X_val)
    ```
    Gave few good results , but tooke a lot of time to complete (~35-40 mins) , so couldn't experiment much.
  </details>

The dataset had heavy class imbalance 
`class(0) : class(1) - 9:1 ` , to counter this did a few experiments :

- <details> <summary><b>Sampling - SMOTE</b></summary>   
    SMOTE (Synthetic Minority Over-sampling Technique) ,it is an oversampling techinque which creates synthetic data of minority class.

    ``` python
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)
        y_train.value_counts()
        #> 0 : 1440 , 1: 1440 (prev. 160)
    ```
    Did not improve the results much (~ - 0.2- 0.3) , felt like the models were overfitting sometimes.
  </details>

- <details> <summary><b>Class weights</b></summary>   
    Most of the models had the feature of adding class weights , which enables the learning of models to pay more attention to minority classes 

    ``` python
        #Logistic Regression
        model = LogisticRegression(class_weight={0: 1, 1: 5})
        #RandomForest
        model = RandomForestClassifier(class_weight="balanced")
        #XGBoost
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(scale_pos_weight=scale_pos_weight)
        #LightGBM
        model = LGBMClassifier(class_weight="balanced")
        #CatBoost
        model = CatBoostClassifier(auto_class_weights="Balanced")

    ```
    Imporved the balanced accuracy a lot , kept implemeting over the course of competition.
  </details> 

I even tried to tune my threshold value , which actually boosed my balanced accuracy score (learning from this I even implemeted a .probabilty in my ml_model from scratch).

- <details> <summary><b>Threshold tuning (Best Balanced accuracy)</b></summary>   
    SMOTE (Synthetic Minority Over-sampling Technique) ,it is an oversampling techinque which creates synthetic data of minority class.

    ``` python
        from sklearn.metrics import balanced_accuracy_score
        import numpy as np

        thresholds = np.arange(0.0, 1.01, 0.01)

        best_threshold = 0
        best_bal_acc = 0

        for t in thresholds:
            y_pred = (y_proba > t).astype(int)
            bal_acc = balanced_accuracy_score(y_val, y_pred)
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_threshold = t

       print("Best threshold:", best_threshold)
       print("Best balanced accuracy:", best_bal_acc)
       #> Best threshold: 0.26
       #> Best balanced accuracy : 0.6944444444444444
    ```
    Helped me increase the balanced accuracy from the one which had a default threshold as 0.5.
  </details>

I tried hyperparameter tuning , which helped me figure out the best hyperparameters of my model:

- <details> <summary><b>GridSearchCV</b></summary>   
    This tries out each and every hyperparamter on the model and returns the best set of hyperparameters , it does all the permuatation and combinations of hyperparameters (mostly for small datasets).

    ``` python
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
    )

    param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.8]
    }

    grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=1
    )

    grid.fit(X_train, y_train)

    ```
    Took a lot of time to implement , could not even complete the hyperparameter tuning using this , as I had a very large dataset to handle.
  </details>

- <details> <summary><b>RandomSearchCV</b></summary>   
    RandomSearchCV randomly picks few combinations and retuns the best hyperparameters from the combinations picked.

    ``` python
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV
        
        rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
        )

        param_dist = {
    "n_estimators": randint(200, 800),
    "max_depth": [None] + list(range(3, 15)),
    "max_features": ["sqrt", "log2", 0.6, 0.8]
        }

        rand_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,                 # increase if time allows
    cv=5,
    scoring="balanced_accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=1
        )

    rand_search.fit(X_train, y_train)
    ```
    Was way faster than GridSearchCV and imporved the results significantly.
  </details>

- <details> <summary><b>Optuna</b></summary>   
    Used Optuna library for hyperparameter tuning which use bayesian statistics , which is basicly an intelligent way (rather than random) to pick combinations . Felt this a bit too new and complex , did try to implement but was running into errors, but it is a faster and more efficient method.
  </details>

Tried experimenting with the a new way of validating the models performance 

- <details> <summary><b>K-Fold Cross-Validation</b></summary>   
    K-Fold Cross-Validation is a method to check how well your model really performs by testing it on different parts of the data, not just one split. It basically breaks the data into K sets , where K-1 sets go to training and the remaining one is used for evaluating the model.
    
    ```python
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="balanced_accuracy"
    )

    print("CV scores:", scores)
    print("Mean CV score:", scores.mean())

    ```
 </details>

---
#### 'ID' Feature 

So I found a strong correlation between the 'id' column and the target column , which boosted my balanced accuract , f1 , recall , precision. But eventually it was not a correct method or way (so dropped it).

---
#### Final Submissions and python notebooks 

`XGB_HP_ANOVA.ipynb`

Model Used : XGBoost 

```python
model = XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        scale_pos_weight= 9 * 1.2,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=1.5,
        tree_method='hist',
        random_state=42 + 2,
        n_jobs=-1
    )

```
Threshold tuning : Best Threshold - 0.06

```python
from sklearn.metrics import balanced_accuracy_score
import numpy as np

thresholds = np.arange(0.0, 1.01, 0.01)

best_threshold = 0
best_bal_acc = 0

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_threshold = t

print("Best threshold:", best_threshold)
print("Best balanced accuracy:", best_bal_acc)

```


| Metric | Score |
|--------|--------|
| Balanced Accuracy | 0.6805555555555556 |
| F1 score | 0.32857142857142857 |
|  Precision | 0.23  |
|Recall | 0.575 |
| Public Score | 1.0 |
| Private Score | 0.50087|


....


`LGBM_HP_ANOVA.ipynb`

Model used : LightGBM

```python
model = LGBMClassifier(random_state=42)
```

ANOVA : Used (k = 1000)

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=1000)
X_train = selector.fit_transform(X_train, y_train)
X_val = selector.transform(X_val)
X_test = selector.transform(X_test)
```

HyperParameter tuning : RandomSearchCV

```python 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

param_dist = {
    'n_estimators': [200, 350],
    'num_leaves': [31, 63],
    'learning_rate': [0.05, 0.1],
    'class_weight': ['balanced']
}
scorer = make_scorer(balanced_accuracy_score)
search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10,
    scoring=scorer,
    cv=3,
    n_jobs=-1,
    verbose=1
)


search.fit(X_train, y_train)

print("Best Params:", search.best_params_)
print("Best Balanced Accuracy:", search.best_score_)

```

| Metric | Score |
|--------|--------|
| Balanced Accuracy | 0.6513888888888889 |
| F1 score | 0.3053435114503817 |
|  Precision | 0.21978021978021978  |
|Recall | 0.5 |
| Public Score | 1.0 |
| Private Score | 0.5002|

> Added Even the best model on the private dataset

`RF_HP_ANOVA.ipynb`

Model Used : Random Forest 

```python
model = RandomForestClassifier(n_estimators=1000, random_state=42)
```

ANOVA : K = 300

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=300)
X_train = selector.fit_transform(X_train, y_train)
X_val = selector.transform(X_val)
X_test = selector.transform(X_test)
```
Threshold tuning : Best Threshold : 0.1

```python
from sklearn.metrics import balanced_accuracy_score
import numpy as np

thresholds = np.arange(0.0, 1.01, 0.01)

best_threshold = 0
best_bal_acc = 0

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_threshold = t

```

HyperParameter tuning : RandomSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

param_dist = {
    'n_estimators': [100, 200, 1000],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}
scorer = make_scorer(balanced_accuracy_score)
search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10,
    scoring=scorer,
    cv=3,
    n_jobs=-1,
    verbose=1
)


search.fit(X_train, y_train)

```

| Metric | Score |
|--------|--------|
| Balanced Accuracy | 0.6944444444444444 |
| F1 score | 0.2714 |
|  Precision |  0.15833333333333333 |
|Recall | 0.95 |
| Public Score | 1.0 |
| Private Score | 0.7099|

---
#### Metrics of other models and values I tried

| Model         | k_selectKBest | Corr Threshold | SMOTE | Balanced Accuracy | Precision | Recall | F1 Score | Leaderboard Score | Hyperparameter Tuning |
|--------------|---------------|----------------|-------|-------------------|-----------|--------|----------|-------------------|-----------------------|
| Random Forest | 300           | 0.95           | No    | 0.7166666667      | 0.2012987013 | 0.775  | 0.3195876289 | 0.99              | HP                    |
| XGBoost       | 1000          | 0.95           | Yes   | 0.6847222222      | 0.1904761905 | 0.70   | 0.2994652406 | 0.87              | HP                    |
| Random Forest | 1000          | 0.90           | Yes   | 0.6763888889      | 0.1693989071 | 0.775  | 0.2780269058 |                   | HP                    |
| Random Forest | 1000          | 0.95           | No    | 0.6750000000      | 0.1562500000 | 0.875  | 0.2651515152 |                   | HP                    |
| Random Forest | 1000          | 0.95           | Yes   | 0.6611111111      | 0.1439393939 | 0.95   | 0.2500000000 | 0.93              | HP                    |
| XGBoost       | 300           | 0.95           | Yes   | 0.6500000000      | 0.1956521739 | 0.45   | 0.2727272727 |                   | HP                    |
| Random Forest | 500           | 0.95           | No    | 0.7083333333      | 0.1923076923 | 0.75   | 0.3061224490 |                   | HP                    |
| Random Forest | 300           | 0.90           | No    | 0.7097222222      | 0.1967213115 | 0.75   | 0.3114754098 |                   | HP                    |
| LightGBM      | 500           | 0.95           | No    | 0.7222222222      | 0.2153846154 | 0.75   | 0.3349514563 | 1.02              | HP                    |                 |
| LightGBM      | 1000          | 0.90           | Yes   | 0.7013888889      | 0.1984126984 | 0.775  | 0.3162393162 |                   | HP                    |
| LightGBM      | 300           | 0.95           | Yes   | 0.6958333333      | 0.2058823529 | 0.70   | 0.3181818182 |                   | HP                    |                 |
| CatBoost      | 1000          | 0.90           | Yes   | 0.7125000000      | 0.2191780822 | 0.775  | 0.3417721519 |                   | HP                    |

