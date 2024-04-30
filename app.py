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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
    MinMaxScaler,
    KBinsDiscretizer,
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
def create_pipeline(model_name, feature_select, feature_create, num_pipe_features, cat_pipe_features, degree = None):
    if model_name == 'Logistic Regression':
        clf = LogisticRegression(class_weight='balanced')
    elif model_name == 'Random Forest':
        clf = RandomForestClassifier(class_weight='balanced')
    # Add more elif statements for other models
    elif model_name == 'Lasso':
        clf = Lasso(alpha = 0.3)
    elif model_name == 'Ridge':
        clf = Ridge()
    elif model_name == 'Linear SVC':
        clf = LinearSVC(class_weight='balanced')
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

    # Feature selection transformer based on user choice
    if feature_select == 'passthrough':
        feature_selector = 'passthrough'
    elif feature_select.startswith('PCA'):
        n_components = int(feature_select.split('(')[1].split(')')[0])
        feature_selector = PCA(n_components=n_components)
    elif feature_select.startswith('SelectKBest'):
        k = int(feature_select.split(',')[1].split('=')[1].strip(')'))
        feature_selector = SelectKBest(k=k)
    elif feature_select.startswith('SelectFromModel'):
        if 'LassoCV' in feature_select:
            model = LassoCV()
        elif 'LinearSVC' in feature_select:
            model = LinearSVC(penalty="l1", dual=False, class_weight='balanced')
        feature_selector = SelectFromModel(model, threshold='median')
    elif feature_select.startswith('RFECV'):
        model = None
        if 'LinearSVC' in feature_select:
            cv_index = feature_select.index('cv=')
            cv_value = int(feature_select[cv_index:].split(',')[0].split('=')[1])
            model = LinearSVC(penalty="l1", dual=False, class_weight='balanced')
        elif 'LogisticRegression' in feature_select:
            cv_index = feature_select.index('cv=')
            cv_value = int(feature_select[cv_index:].split(',')[0].split('=')[1])
            model = LogisticRegression(class_weight='balanced')
        feature_selector = RFECV(model, cv=cv_value, scoring=prof_score)
    elif feature_select.startswith('SequentialFeatureSelector'):
        model = None
        if 'LogisticRegression' in feature_select:
            model = LogisticRegression(class_weight='balanced')
        scoring = prof_score
        n_features_to_select = int(feature_select.split(',')[2].split('=')[1])
        cv = int(feature_select.split(',')[3].split('=')[1].strip(')'))
        feature_selector = SequentialFeatureSelector(model, scoring=scoring, n_features_to_select=n_features_to_select, cv=cv)
    else:
        st.error("Invalid feature selection method!")
        return None

    # Define the feature creation transformer based on the selected method
    if feature_create == 'passthrough':
        feature_creator = 'passthrough'
    elif feature_create.startswith('PolynomialFeatures'):
        interaction_only = 'interaction_only' in feature_create
        feature_creator = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
    elif feature_create == 'Binning':
        feature_creator = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    elif feature_create == 'Feature Scaling':
        feature_creator = MinMaxScaler()
        
    # I used "Pipeline" not "make_pipeline" bc I wanted to name the steps
    pipe = Pipeline([('columntransformer',preproc_pipe),
                 ('feature_create', feature_creator), 
                 ('feature_select', feature_selector), 
                 ('clf', clf)
                ])

    return pipe



'''
hi
'''

##################################################
# begin : user choices
st.title("Choose Model, Feature Selection Method, Feature Creation Method, Features, and Display Pipeline")
# num_pipe_features =  .... st.menu(list of choices or something);

# Checkbox to select numerical features
selected_num_features = st.multiselect("Select Numerical Features:", num_pipe_features)

# Checkbox to select categorical features
selected_cat_features = st.multiselect("Select Categorical Features:", cat_pipe_features)
    
# Dropdown menu to choose the model
model_name = st.selectbox("Choose Model:", ['Logistic Regression', 'Random Forest', 'Lasso', 'Ridge', 'Linear SVC'])
st.write("Selected Model:", model_name)

# Dropdown menu to choose the feature selection method
feature_select_method = st.selectbox("Choose Feature Selection Method:", ['passthrough', 'PCA(5)', 'PCA(10)', 'PCA(15)',
                                                                             'SelectKBest(f_classif,k=5)', 'SelectKBest(f_classif,k=10)', 'SelectKBest(f_classif,k=15)',
                                                                             'SelectFromModel(LassoCV())', 'SelectFromModel(LinearSVC(penalty="l1", dual=False, class_weight="balanced"), threshold="median")',
                                                                             'RFECV(LinearSVC(penalty="l1", dual=False, class_weight="balanced"), cv=2, scoring=prof_score)',
                                                                             'RFECV(LogisticRegression(class_weight="balanced"), cv=2, scoring=prof_score)',
                                                                             'SequentialFeatureSelector(LogisticRegression(class_weight="balanced"), scoring=prof_score, n_features_to_select=5, cv=2)',
                                                                             'SequentialFeatureSelector(LogisticRegression(class_weight="balanced"), scoring=prof_score, n_features_to_select=10, cv=2)',
                                                                             'SequentialFeatureSelector(LogisticRegression(class_weight="balanced"), scoring=prof_score, n_features_to_select=15, cv=2)'])

# Dropdown menu to choose the feature creation method
feature_create_method = st.selectbox("Choose Feature Creation Method:", ['passthrough', 'PolynomialFeatures', 'Binning', 'Feature Scaling'])

# If PolynomialFeatures is selected, provide an input field to specify the degree
if feature_create_method == 'PolynomialFeatures':
    degree = st.number_input("Enter the degree for PolynomialFeatures", min_value=1, max_value=5, value=2)
else:
    degree = None

# Dropdown menu to choose the cross-validation strategy
cv = st.number_input("Enter the number of folds for cross-validation", min_value=2, max_value=10, value=5)

# Create the pipeline based on the selected model and features
pipe = create_pipeline(model_name, feature_select_method, feature_create_method, selected_num_features, selected_cat_features, degree)

# end: user choices
##################################################

# pipe.set_param() # replace the vars eith the vars they want)
# pipe.set_param() # replace the model with the model they choose

pipe

 # why isn't thisprinting in streamlit
