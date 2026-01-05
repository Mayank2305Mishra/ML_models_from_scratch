# Winter of Code 2025/26

This file contains the report of kaggle competition, Binary Buzz , which was a classification problem.

----
### Understanding the datasets

> The competition provided 2 files : `train.csv` and `test.csv`

#### train.csv
This file had a shape of (2000, 3074) :

`2000` - 2000 rows or datasamples were provides

`3074` - 3074 columns which inducled the training features, id column and the target column.

##### Key Highlights about the file :

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
        X_test = scaler.transform(X_test)
    ```
  </details>

--- 
#### Experiments and Trials

Most of my experiments were done to get rid of the high dimensionality of the data (Curse of Dimensionality).

- <details> <summary><b>ANOVA</b></summary>   
    ANOVA stands for Analysis of Variance , it used to check how well a feature seprates a class , so we can select the k best features (based on best split) out of 3072

    ``` python
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=1000) # k could be varied
        X_train = selector.fit_transform(X_train, y_train)
        X_val = selector.transform(X_val)
        X_test = selector.transform(X_test)
    ```
    Tried to play around with different values of k and different models.
  </details>