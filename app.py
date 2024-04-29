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
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
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

dont_use = ["member_id", "id", "desc", "earliest_cr_line", "emp_title", "issue_d","title"]

# list of all num vars:
num_pipe_features = X_train.select_dtypes(include="number").columns

# exclude any bad features:
num_pipe_features = [e for e in num_pipe_features if e not in dont_use]

cat_pipe_features = ["grade"]  # all: X_train.select_dtypes(include='object').columns

##################################################

numer_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())

cat_pipe = make_pipeline(OneHotEncoder())

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
                 ('clf', LogisticRegression(class_weight='balanced'))
                ])
'''
hi
'''

##################################################
# begin : user choices

# num_pipe_features =  .... st.menu(list of choices or something)

# end: user choices
##################################################

# pipe.set_param() # replace the vars eith the vars they want)
# pipe.set_param() # replace the model with the model they choose

# Setting the configuration to 'diagram' mode for visualization
set_config(display='diagram')

# Plotting the pipeline
st.write(pipe)

# Saving the pipeline diagram to a file
pipeline_diagram = "pipeline_diagram.png"

# Fit the pipeline on your data
pipe.fit(X_train)

# Plotting the decision tree
plt.figure(figsize=(20, 10))
plot_tree(pipe.named_steps['clf'], filled=True, feature_names=X_train.columns, class_names=["0", "1"])
plt.savefig(pipeline_diagram)

# Displaying the pipeline diagram
st.image(pipeline_diagram, caption='Pipeline Diagram') # why isn't thisprinting in streamlit
