# I'm putting all code we've seen before here

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from df_after_transform import df_after_transform
from sklearn import set_config
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import (
    RFECV,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    f_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.svm import LinearSVC
import streamlit as st


set_config(display="diagram")  # display='text' is the default

pd.set_option(
    "display.max_colwidth", 1000, "display.max_rows", 50, "display.max_columns", None
)

# load data

loans = pd.read_csv("inputs/2013_subsample.zip")

# drop some bad columns here, or in the pipeline

# loans = loans.drop(
#     ["member_id", "id", "desc", "earliest_cr_line", "emp_title", "issue_d"], axis=1
# )

# create holdout sample

y = loans.loan_status == "Charged Off"
y.value_counts()
loans = loans.drop("loan_status", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    loans, y, stratify=y, test_size=0.2, random_state=0
)  # (stratify will make sure that test/train both have equal fractions of outcome)

# define the profit function


def custom_prof_score(y, y_pred, roa=0.02, haircut=0.20):
    """
    Firm profit is this times the average loan size. We can
    ignore that term for the purposes of maximization. 
    """
    TN = sum((y_pred == 0) & (y == 0))  # count loans made and actually paid back
    FN = sum((y_pred == 0) & (y == 1))  # count loans made and actually defaulting
    return TN * roa - FN * haircut


# so that we can use the fcn in sklearn, "make a scorer" out of that function

prof_score = make_scorer(custom_prof_score)

# list of all num vars:
num_pipe_features = X_train.select_dtypes(include="number").columns

# List of all categorical variables
cat_pipe_features = X_train.select_dtypes(include='object').columns  # all: X_train.select_dtypes(include='object').columns


##################################################
# Function to create a pipeline based on user-selected model and features
def create_pipeline(model_name, num_pipe_features, cat_pipe_features):
    if model_name == 'Logistic Regression':
        clf = LogisticRegression(class_weight='balanced')
    elif model_name == 'Random Forest':
        clf = RandomForestClassifier(class_weight='balanced')
    # Add more elif statements for other models
    elif model_name == 'Lasso':
        clf = Lasso(alpha = 0.3)
    elif model_name == 'Ridge':
        clf = Ridge()
    # Preprocessing pipelines for numerical and categorical features
    numer_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())

    cat_pipe = make_pipeline(OneHotEncoder())
    
    # Preprocessing pipeline for the entire dataset
    # didn't use make_column_transformer; wanted to name steps
    preproc_pipe = make_column_transformer(
    (numer_pipe, num_pipe_features), 
    (cat_pipe, cat_pipe_features), 
    remainder="drop",
    )

    # I used "Pipeline" not "make_pipeline" bc I wanted to name the steps
    pipe = Pipeline([('columntransformer',preproc_pipe),
                 ('feature_create','passthrough'), 
                 ('feature_select','passthrough'), 
                 ('clf', clf)
                ])

    return pipe



'''
hi
'''

##################################################
# begin : user choices
st.title("Choose Model and Features, and Display Pipeline")
# num_pipe_features =  .... st.menu(list of choices or something);

# Checkbox to select numerical features
selected_num_features = st.multiselect("Select numerical features:", num_pipe_features)

# Checkbox to select categorical features
selected_cat_features = st.multiselect("Select categorical features:", cat_pipe_features)
    
# Dropdown menu to choose the model
model_name = st.selectbox("Choose Model:", ['Logistic Regression', 'Random Forest'])
st.write("Selected Model:", model_name)

# Create the pipeline based on the selected model and features
pipe = create_pipeline(model_name, selected_num_features, selected_cat_features)

# end: user choices
##################################################

# pipe.set_param() # replace the vars eith the vars they want)
# pipe.set_param() # replace the model with the model they choose

pipe

 # why isn't thisprinting in streamlit
