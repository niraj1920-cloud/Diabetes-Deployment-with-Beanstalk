
# Diabetes Prediction System



## About the Project
The Diabetes Prediction System project is focused on predicting the likelihood of diabetes in patients based on medical attributes like glucose levels, blood pressure, insulin, BMI, and age. This classification model serves as a diagnostic aid for healthcare providers and patients.


## About Dataset

The dataset used in this project is a medical dataset containing various health metrics to predict diabetes outcomes in patients. Each row represents an individual patient's health data, with a total of 8 features and 1 target variable:

Features:

Input Variables:

- Pregnancies: Number of times the patient has been pregnant.

- Glucose: Plasma glucose concentration (mg/dL), measured 2 hours after a glucose tolerance test.

- BloodPressure: Diastolic blood pressure (mm Hg).

- SkinThickness: Skin fold thickness (mm) as a measure of subcutaneous fat.

- Insulin: Serum insulin (mu U/ml), which helps gauge insulin levels in the body.
- BMI: Body mass index, a ratio of weight to height (kg/m²), indicating body fat.
- DiabetesPedigreeFunction: A score (0-2.5) estimating genetic predisposition to diabetes, based on family history.
- Age: Age of the patient in years.

Target variable:

- Outcome: A binary variable where 1 indicates the presence of diabetes, and 0 indicates no diabetes.
## Tech Stack
 - Python
 - Pandas
 - Numpy
 - Scikit-Learn
 - Flask
 - AWS Beanstalk
## Project Structure

```bash
  Diabetes_Prediction_System/
├── .ebextensions/
│   └── python.config                 # AWS Elastic Beanstalk configuration for deployment.
│
├── Dataset/
│   └── diabetes.csv                  # Dataset containing medical data for diabetes prediction.
│
├── Model/
│   ├── modelForPrediction.pkl        # Saved logistic regression model for making predictions.
│   └── standardScalar.pkl            # Saved StandardScaler object to standardize input features.
│
├── Notebook/
│   └── Logistic_Regression.ipynb     # Jupyter notebook with data preprocessing, EDA, model training, and evaluation.
│
├── templates/
│   ├── home.html                     # Home page template for web interface.
│   ├── index.html                    # Main page for user interaction (data input, predictions).
│   └── single_prediction.html        # Template to display individual prediction results.
│
├── application.py                    # Flask application file to run the web app and serve predictions.
│
└── requirements.txt                  # File listing all Python dependencies for the project.

```

## Workflow

1. Data collection and loading:
 - The first step is to load the diabetes dataset from a CSV file. This dataset contains multiple features related to diabetes, such as Glucose, Blood Pressure, BMI, Age, and the target variable "Outcome" which indicates whether a patient has diabetes (1) or not (0).

2. Handling Missing or Incorrect Data:
 - In this step, certain features (like BMI, Blood Pressure, Glucose, etc.) are checked for zero values, as zeros are considered incorrect in medical contexts. These zero values are replaced with the mean of the respective columns to prevent skewed results.

    ```bash
    here few misconception is there like BMI can not be zero, BP can't be zero, glucose, insuline can't be zero so lets try to fix it
    # now replacing zero values with the mean of the column
    data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())
    data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
    data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
    data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
    data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())
    ```

3. Data Preprocessing:
- Dependent and Dependent Variables: The dataset is divided into features (independent variables, X) and the target variable (dependent variable, y).
- Scaling: The features are scaled using StandardScaler to ensure that they all have a mean of 0 and a standard deviation of 1, making it easier for the model to learn from the data.
    ```bash
  ##standard Scaling- Standardization
    def scaler_standard(X_train, X_test):
        #scaling the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    ```
- Saving the Scaler: The scaler used for standardization is saved using pickle, so it can be used later to scale new data consistently.

    ```bash
  #saving the model
    file = open('C:/Users/Niraj/Documents/Logistic Regression Project/Model/standardScalar.pkl','wb')
    pickle.dump(scaler,file)
    file.close()
    ```

4. Model Building:
- Logistic Regression: A logistic regression model is initialized. This model is suitable for binary classification tasks like predicting whether a person has diabetes or not.
- Training the Model: The model is trained on the scaled training data (X_train_scaled and y_train).

    ```bash
  log_reg = LogisticRegression()

    log_reg.fit(X_train_scaled,y_train)
    ```

5. Hyperparameter Tuning (Grid Search):
- GridSearchCV is used to fine-tune the hyperparameters of the logistic regression model. The parameter grid includes various values for penalties, regularization strength (C), and solvers. This helps improve the model's performance by finding the best combination of these hyperparameters.

    ```bash
  ## Hyperparameter Tuning
    ## GridSearch CV
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    # parameter grid
    parameters = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-3,3,7),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
    }
    ```

6. Model Evaluation:
- Predictions: After training, the model is used to make predictions on the test dataset (X_test).
- Confusion Matrix: A confusion matrix is generated to evaluate the performance of the model. It provides insight into how many true positives, false positives, true negatives, and false negatives were identified by the model.

    ```bash
    conf_mat = confusion_matrix(y_test,y_pred)
    ```
    ```bash
   true_positive = conf_mat[0][0]
   false_positive = conf_mat[0][1]
   false_negative = conf_mat[1][0]
   true_negative = conf_mat[1][1]
    ```

- Performance Metrics:
    - Accuracy: Measures the percentage of correct predictions.

        ```bash
        Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
        0.796875
        ```
    - Precision: Indicates the proportion of positive predictions that are actually correct.
        ```bash
       Precision = true_positive / (true_positive + false_positive)
        0.9
        ```
    - Recall: Measures the model’s ability to correctly identify all actual positive cases.
        ```bash
       Recall = true_positive / (true_positive + false_negative)
        0.8181818181818182
        ```

    - F1 Score: Provides a balanced measure of Precision and Recall, useful when there’s an uneven class distribution.
        ```bash
      F1_Score = 2*(Recall * Precision) / (Recall + Precision)
      0.8571428571428572
        ```

7. Saving the Model:
- Once the model is trained and evaluated, it is saved using pickle as a .pkl file. This allows for easy loading of the model later without the need to retrain it.
- The saved model is stored as modelForPrediction.pkl, which can then be used to make predictions on new data.

8. Developing a Flask Application:
- A Flask web application is developed to provide an interactive user interface where users can input their data (e.g., glucose, BMI, etc.) and get predictions on whether they are likely to have diabetes or not.
- The application loads the saved model and scaler, processes user inputs, and provides real-time predictions.

9. Deploying the Application to AWS Elastic Beanstalk:
- The Flask application is packaged with all necessary dependencies and configuration files, such as requirements.txt, application.py, and any environment-specific configurations (e.g., .ebextensions).
- The application is deployed to AWS Elastic Beanstalk, a cloud platform for deploying and managing applications. AWS Elastic Beanstalk automatically handles the infrastructure, load balancing, scaling, and application monitoring.
  
## Screenshots

![image](https://github.com/user-attachments/assets/e7bfbd60-cf29-46c5-ae88-fe4a3f417483)

![image](https://github.com/user-attachments/assets/7a2cccf9-36b1-4c9f-9a7a-20aa9b52aa02)


## Contributions
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request
## Deployment

To deploy this project run

```bash
  npm run deploy
```

