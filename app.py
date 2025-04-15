import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import (
    RFECV,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    f_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve,
    classification_report,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
    cross_val_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
)
from sklearn.svm import LinearSVC
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Custom import for TunedThresholdClassifierCV
from sklearn.model_selection import TunedThresholdClassifierCV

################################ formatting #############################

set_config(display="diagram")  # display='text' is the default

# Page config
st.set_page_config(
    "Machine Learning to Create Custom Predictions for Loan Defaults",
    "📈",
    initial_sidebar_state="expanded",
    layout="wide",
)

pd.set_option(
    "display.max_colwidth", 1000, "display.max_rows", 50, "display.max_columns", None
)

################################################ sidebar #############################################
with st.sidebar:
    if 'current_section' not in st.session_state:
        st.session_state['current_section'] = 'Overview'

    with st.sidebar:
        st.write("# Menu:")

        menu_options = {
            "Overview, Objectives, Process, and Results": "Overview",
            "Custom Machine Learning Model Builder": "Custom Model Builder",
            "Dictionary For Variables Used": "Dictionary"
        }

        # Use buttons with space padding for alignment
        max_length = max(len(option) for option in menu_options.keys())
        for text, section in menu_options.items():
            padded_text = text.ljust(max_length)  # Padding text to make uniform
            if st.button(padded_text):
                st.session_state['current_section'] = section

############################################# load data, scoring, features ################################################

# load data
loans = pd.read_csv("inputs/final_2013_subsample.csv")

# Drop unnecessary columns
loans = loans.drop(["id", "member_id", "desc", "earliest_cr_line", "emp_title", "issue_d"], axis=1)

# create target variable and drop from features
y = loans.loan_status == "Charged Off"
loans = loans.drop("loan_status", axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    loans, y, stratify=y, test_size=0.2, random_state=0
)  # (stratify will make sure that test/train both have equal fractions of outcome)

# Define the profit function
def custom_prof_score(y, y_pred, roa=0.02, haircut=0.20):
    """
    Firm profit is this times the average loan size. We can
    ignore that term for the purposes of maximization. 
    
    Parameters:
    -----------
    y : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    roa : float, default=0.02
        Return on assets for loans paid back.
    haircut : float, default=0.20
        Loss on defaulted loans.
        
    Returns:
    --------
    float
        Profitability score (higher is better).
    """
    TN = sum((y_pred == 0) & (y == 0))  # count loans made and actually paid back
    FN = sum((y_pred == 0) & (y == 1))  # count loans made and actually defaulting
    return TN * roa - FN * haircut

# Create a scorer that can be used in sklearn
prof_score = make_scorer(custom_prof_score)

# List of all numerical variables
num_pipe_features = X_train.select_dtypes(include="float64").columns

# List of all categorical variables
cat_pipe_features = X_train.select_dtypes(include='object').columns

################################################## custom model code #################################################

# Function to create a pipeline based on user-selected model and features
def create_pipeline(model_name, feature_select, feature_create, num_pipe_features, cat_pipe_features, degree=None):
    """
    Creates a machine learning pipeline with user-selected components.
    
    Parameters:
    -----------
    model_name : str
        Name of the classification model to use.
    feature_select : str
        Feature selection method to use.
    feature_create : str
        Feature creation method to use.
    num_pipe_features : list
        List of numerical feature column names.
    cat_pipe_features : list
        List of categorical feature column names.
    degree : int, optional
        Degree for polynomial features if applicable.
        
    Returns:
    --------
    Pipeline
        Configured scikit-learn pipeline.
    """
    # Select base classifier
    if model_name == 'Logistic Regression':
        clf = LogisticRegression(class_weight='balanced', penalty='l2')
    elif model_name == 'Linear SVC':
        clf = LinearSVC(class_weight='balanced', penalty='l2')
    elif model_name == 'K-Nearest Neighbors':
        clf = KNeighborsClassifier(weights='uniform')
    elif model_name == 'Decision Tree':
        clf = DecisionTreeClassifier(class_weight='balanced')
        
    # Preprocessing pipelines for numerical and categorical features
    numer_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
    cat_pipe = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
    
    # Preprocessing pipeline for the entire dataset
    preproc_pipe = make_column_transformer(
        (numer_pipe, num_pipe_features), 
        (cat_pipe, cat_pipe_features), 
        remainder="drop",
    )

    # Define the feature selection transformer
    if feature_select == 'passthrough':
        feature_selector = 'passthrough'
    elif feature_select.startswith('PCA'):
        n_components = int(feature_select.split('(')[1].split(')')[0])
        feature_selector = TruncatedSVD(n_components=n_components)
    elif feature_select.startswith('SelectKBest'):
        feature_selector = SelectKBest(score_func=f_classif)
    elif feature_select.startswith('SelectFromModel'):
        if 'LinearSVC' in feature_select:
            model = LinearSVC(penalty="l2", dual=False, class_weight='balanced')
            feature_selector = SelectFromModel(model)
    elif feature_select.startswith('RFECV'):    
        if 'LogisticRegression' in feature_select:
            model = LogisticRegression(class_weight='balanced')
            feature_selector = RFECV(model, cv=5, scoring=prof_score)
    elif feature_select.startswith('SequentialFeatureSelector'):
        if 'LogisticRegression' in feature_select:
            model = LogisticRegression(class_weight='balanced')
            feature_selector = SequentialFeatureSelector(model, scoring=prof_score, n_features_to_select=2, cv=5)
    else:
        st.error("Invalid feature selection method!")
        return None

    # Define the feature creation transformer
    if feature_create == 'passthrough':
        feature_creator = 'passthrough'
    elif feature_create.startswith('PolynomialFeatures'):
        interaction_only = 'interaction_only' in feature_create
        feature_creator = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
    elif feature_create == 'MinMaxScaler':
        feature_creator = MinMaxScaler()
    elif feature_create == 'MaxAbsScaler':
        feature_creator = MaxAbsScaler()
    
    # Create pipeline with TunedThresholdClassifierCV
    pipe = Pipeline([
        ('columntransformer', preproc_pipe),
        ('feature_create', feature_creator), 
        ('feature_select', feature_selector), 
        ('clf', TunedThresholdClassifierCV(clf, scoring=prof_score, cv=5))
    ])

    return pipe

################################################### Overview ########################################################

if st.session_state['current_section'] == 'Overview':

    st.markdown("<h1 style='text-align: center;'>Overview</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .centered-text {
        text-align: center;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Center and style the paragraph text
    st.markdown("""
    <div class="centered-text">
    When a loan is taken out, the lender assumes the risk that the borrower will default. This application helps predict loan defaults based on various attributes, optimizing lender profitability by balancing the return on successful loans against the losses from defaults. Our custom machine learning models aim to identify the most reliable predictors of loan repayment behavior.
    </div>""", unsafe_allow_html=True)
    
    st.write("\n" * 2)

    st.subheader("Project Overview")
    st.write("""This dashboard enables users to build custom prediction pipelines by selecting predictor variables, model types, and feature engineering techniques. The application focuses on maximizing profitability rather than just predictive accuracy, reflecting real-world lending considerations.""")
    
    st.write("\n" * 2)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Type")
        st.write("Binary classification (default vs. non-default)")
        
        st.subheader("Available Models")
        st.write("• Logistic Regression\n• Linear SVC\n• K-Nearest Neighbors\n• Decision Tree")
        
        st.subheader("Data Source")
        st.write("""The dataset contains 134,804 loan records from 2013, with 113,780 fully paid loans and 21,024 charged-off loans (defaults). Each record includes 33 variables covering borrower characteristics and loan terms.""")
    
    with col2:
        st.subheader("Key Objective")
        st.write("""Optimize model performance using the custom profit function, which balances returns from successful loans against losses from defaults.""")
        
        st.subheader("Key Hypothesis")
        st.write("""Interest rate is hypothesized to be the most significant predictor of loan defaults compared to other common indicators.""")
        
        st.subheader("Sample Period")
        st.write("January 2013 – December 2013")

    st.write("\n" * 2)

    st.subheader("Methodology")
    st.write("""
    Our approach involves:
    
    1. Data preparation with an 80-20 train-test split, stratified by the target variable
    2. Feature engineering with categorical encoding and numerical scaling
    3. Feature selection to identify the most predictive variables
    4. Model training with threshold optimization using TunedThresholdClassifierCV
    5. Evaluation based on both traditional metrics and custom profitability score
    
    The profitability score is calculated as: TN * ROA - FN * haircut, where TN represents true negatives (predicted and actual non-defaults), FN represents false negatives (predicted non-defaults that actually defaulted), ROA is the return on assets (2%), and haircut is the loss on defaulted loans (20%).
    """)

    st.write("\n" * 2)

    st.subheader("Key Performance Indicators")
    st.write("""
    Models are evaluated using:
    
    1. Custom profit score - our primary optimization target
    2. Precision - the proportion of predicted non-defaults that actually paid back the loan
    3. Recall - the proportion of actual non-defaults correctly identified by the model
    4. F1-score - the harmonic mean of precision and recall
    """)

################################################### Custom Model Builder ########################################################

elif st.session_state['current_section'] == 'Custom Model Builder':
    
    st.markdown("<h1 style='text-align: center;'>Custom Loan Default Prediction Model</h1>", unsafe_allow_html=True)
    
    st.write("""
    This tool allows you to build a custom machine learning pipeline to predict loan defaults.
    Select your preferred features, model, and feature engineering techniques to optimize the profit score.
    """)
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Model Configuration", "Results & Evaluation"])
    
    with tab1:
        # Feature Selection UI
        st.subheader("Step 1: Select Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Features**")
            selected_num_features = st.multiselect(
                "Select numerical variables to include:", 
                num_pipe_features, 
                default=list(num_pipe_features)[:5]  # Default to first 5 numerical features
            )
            
        with col2:
            st.write("**Categorical Features**")
            selected_cat_features = st.multiselect(
                "Select categorical variables to include:", 
                cat_pipe_features, 
                default=list(cat_pipe_features)[:3]  # Default to first 3 categorical features
            )
        
        # Model Selection UI
        st.subheader("Step 2: Choose Model Type")
        model_options = ['Logistic Regression', 'Linear SVC', 'K-Nearest Neighbors', 'Decision Tree']
        model_name = st.selectbox("Select classification model:", model_options)
        
        # Feature Engineering UI
        st.subheader("Step 3: Select Feature Engineering Methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Feature Selection Method**")
            feature_select_options = [
                'passthrough', 
                'PCA(10)', 
                'SelectKBest(f_classif)', 
                'SelectFromModel(LinearSVC(penalty="l2", dual=False))', 
                'RFECV(LogisticRegression, scoring=prof_score)', 
                'SequentialFeatureSelector(LogisticRegression, scoring=prof_score)'
            ]
            feature_select_method = st.selectbox(
                "Choose feature selection technique:", 
                feature_select_options
            )
            
        with col2:
            st.write("**Feature Creation Method**")
            feature_create_options = [
                'passthrough', 
                'PolynomialFeatures', 
                'MinMaxScaler', 
                'MaxAbsScaler'
            ]
            feature_create_method = st.selectbox(
                "Choose feature creation technique:", 
                feature_create_options
            )
        
        # Additional parameters based on feature creation method
        if feature_create_method == 'PolynomialFeatures':
            col1, col2 = st.columns(2)
            with col1:
                degree = st.slider(
                    "Polynomial degree:", 
                    min_value=2, 
                    max_value=3, 
                    value=2,
                    help="Higher values create more complex interactions but risk overfitting"
                )
            with col2:
                interaction_only = st.checkbox(
                    "Interaction terms only (no polynomial terms)", 
                    value=True,
                    help="If selected, only interaction terms will be created, not pure polynomial terms"
                )
                feature_create_method = "PolynomialFeatures(interaction_only)" if interaction_only else "PolynomialFeatures"
        else:
            degree = None
        
        # Hyperparameter Tuning UI
        st.subheader("Step 4: Configure Hyperparameter Tuning")
        
        hyperparameter_ranges = {}
        
        # Model-specific hyperparameters
        if model_name in ['Linear SVC', 'Logistic Regression']:
            col1, col2 = st.columns(2)
            with col1:
                C_min = st.slider(
                    'Regularization strength (C) - Min', 
                    min_value=0.01, 
                    max_value=10.0, 
                    value=0.1,
                    format="%.2f",
                    help="Lower values increase regularization"
                )
            with col2:
                C_max = st.slider(
                    'Regularization strength (C) - Max', 
                    min_value=0.01, 
                    max_value=10.0, 
                    value=1.0,
                    format="%.2f"
                )
            hyperparameter_ranges['C'] = np.logspace(np.log10(C_min), np.log10(C_max), num=5)
            
        elif model_name == 'K-Nearest Neighbors':
            col1, col2 = st.columns(2)
            with col1:
                n_neighbors_min = st.slider(
                    'Number of Neighbors - Min', 
                    min_value=1, 
                    max_value=20, 
                    value=3
                )
            with col2:
                n_neighbors_max = st.slider(
                    'Number of Neighbors - Max', 
                    min_value=1, 
                    max_value=20, 
                    value=7
                )
            hyperparameter_ranges['n_neighbors'] = list(range(n_neighbors_min, n_neighbors_max + 1))
            
        elif model_name == 'Decision Tree':
            col1, col2 = st.columns(2)
            with col1:
                min_split_min = st.slider(
                    'Min Samples Split - Min', 
                    min_value=2, 
                    max_value=20, 
                    value=2,
                    help="Minimum samples required to split an internal node"
                )
            with col2:
                min_split_max = st.slider(
                    'Min Samples Split - Max', 
                    min_value=2, 
                    max_value=20, 
                    value=10
                )
            hyperparameter_ranges['min_samples_split'] = list(range(min_split_min, min_split_max + 1))
            
            col1, col2 = st.columns(2)
            with col1:
                max_depth_min = st.slider(
                    'Max Depth - Min', 
                    min_value=1, 
                    max_value=20, 
                    value=3,
                    help="Maximum depth of the tree"
                )
            with col2:
                max_depth_max = st.slider(
                    'Max Depth - Max', 
                    min_value=1, 
                    max_value=20, 
                    value=10
                )
            hyperparameter_ranges['max_depth'] = list(range(max_depth_min, max_depth_max + 1))
        
        # Feature selection-specific hyperparameters
        if feature_select_method.startswith('SelectKBest'):
            col1, col2 = st.columns(2)
            with col1:
                k_min = st.slider(
                    'Number of Features (k) - Min', 
                    min_value=1, 
                    max_value=30, 
                    value=5
                )
            with col2:
                k_max = st.slider(
                    'Number of Features (k) - Max', 
                    min_value=1, 
                    max_value=30, 
                    value=15
                )
            hyperparameter_ranges['feature_select__k'] = list(range(k_min, k_max + 1, 2))
            
        elif feature_select_method.startswith('PCA'):
            col1, col2 = st.columns(2)
            with col1:
                n_components_min = st.slider(
                    'Number of Components - Min', 
                    min_value=1, 
                    max_value=20, 
                    value=5
                )
            with col2:
                n_components_max = st.slider(
                    'Number of Components - Max', 
                    min_value=1, 
                    max_value=20, 
                    value=10
                )
            hyperparameter_ranges['feature_select__n_components'] = list(range(n_components_min, n_components_max + 1))
        
        # Cross-validation configuration
        st.subheader("Step 5: Configure Cross-Validation")
        num_folds = st.slider(
            "Number of cross-validation folds:", 
            min_value=3, 
            max_value=10, 
            value=5,
            help="Higher values give more robust estimates but take longer to run"
        )
        cv = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        # Train model button
        train_model = st.button("Train Model", type="primary")
    
    # Create the pipeline based on user selections
    if selected_num_features or selected_cat_features:
        pipe = create_pipeline(
            model_name, 
            feature_select_method, 
            feature_create_method, 
            selected_num_features, 
            selected_cat_features, 
            degree
        )
    else:
        st.warning("Please select at least one feature to continue.")
        pipe = None
    
    # Function to construct parameter grid based on user inputs
    def construct_param_grid(model_name, feature_select_method, hyperparameter_ranges):
        param_grid = {}
        
        # Add model-specific parameters
        if model_name in ['Logistic Regression', 'Linear SVC'] and 'C' in hyperparameter_ranges:
            param_grid['clf__estimator__C'] = hyperparameter_ranges['C']
            
        elif model_name == 'K-Nearest Neighbors' and 'n_neighbors' in hyperparameter_ranges:
            param_grid['clf__estimator__n_neighbors'] = hyperparameter_ranges['n_neighbors']
            
        elif model_name == 'Decision Tree':
            if 'min_samples_split' in hyperparameter_ranges:
                param_grid['clf__estimator__min_samples_split'] = hyperparameter_ranges['min_samples_split']
            if 'max_depth' in hyperparameter_ranges:
                param_grid['clf__estimator__max_depth'] = hyperparameter_ranges['max_depth']
        
        # Add feature selection parameters
        if feature_select_method.startswith('SelectKBest') and 'feature_select__k' in hyperparameter_ranges:
            param_grid['feature_select__k'] = hyperparameter_ranges['feature_select__k']
            
        elif feature_select_method.startswith('PCA') and 'feature_select__n_components' in hyperparameter_ranges:
            param_grid['feature_select__n_components'] = hyperparameter_ranges['feature_select__n_components']
        
        return param_grid
    
    with tab2:
        if train_model and pipe is not None:
            with st.spinner('Training model and performing cross-validation...'):
                # Update parameter grid with user-defined hyperparameter ranges
                param_grid = construct_param_grid(model_name, feature_select_method, hyperparameter_ranges)
                
                if not param_grid:
                    st.warning("No hyperparameters to tune. Using default values.")
                
                # Display pipeline and parameters
                st.subheader("Pipeline Configuration")
                st.write("**Selected Model Pipeline:**")
                st.write(pipe)
                
                st.write("**Hyperparameter Search Space:**")
                st.json(param_grid)
                
                # Create GridSearchCV
                grid_search = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    cv=cv,
                    scoring=prof_score,
                    n_jobs=-1,  # Use all available cores
                    verbose=1,
                    return_train_score=True
                )
                
                try:
                    # Fit the grid search
                    with st.spinner('Performing grid search...'):
                        results = grid_search.fit(X_train, y_train)
                    
                    # Display model performance
                    st.subheader("Model Performance")
                    
                    # Best parameters
                    st.write("**Best Hyperparameters:**")
                    st.json(results.best_params_)
                    
                    # Best score
                    st.write(f"**Best Cross-Validated Profit Score:** {results.best_score_:.4f}")
                    
                    # Test set evaluation
                    y_pred_test = results.predict(X_test)
                    test_profit = custom_prof_score(y_test, y_pred_test)
                    st.write(f"**Test Set Profit Score:** {test_profit:.4f}")
                    
                    # Classification report on test set
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred_test, output_dict=True)
                    
                    # Create a styled table for the classification report
                    report_df = pd.DataFrame({
                        "Precision": [report["0"]["precision"], report["1"]["precision"]],
                        "Recall": [report["0"]["recall"], report["1"]["recall"]],
                        "F1-Score": [report["0"]["f1-score"], report["1"]["f1-score"]],
                        "Support": [report["0"]["support"], report["1"]["support"]]
                    }, index=["Non-Default", "Default"])
                    
                    # Add accuracy row
                    accuracy_df = pd.DataFrame({
                        "Precision": [np.nan],
                        "Recall": [np.nan],
                        "F1-Score": [report["accuracy"]],
                        "Support": [report["0"]["support"] + report["1"]["support"]]
                    }, index=["Accuracy"])
                    
                    report_df = pd.concat([report_df, accuracy_df])
                    st.table(report_df)
                    
                    # Feature importance analysis
                    st.subheader("Feature Importance Analysis")
                    
                    try:
                        # Extract feature names after preprocessing
                        if hasattr(results.best_estimator_[-1].estimator, 'coef_'):
                            # Get feature names after feature selection
                            if hasattr(results.best_estimator_[:-1], 'get_feature_names_out'):
                                try:
                                    feature_names = results.best_estimator_[:-1].get_feature_names_out()
                                except:
                                    feature_names = [f"Feature {i}" for i in range(results.best_estimator_[-1].estimator.coef_.shape[1])]
                            else:
                                feature_names = [f"Feature {i}" for i in range(results.best_estimator_[-1].estimator.coef_.shape[1])]
                            
                            # Extract coefficients
                            if hasattr(results.best_estimator_[-1].estimator, 'coef_'):
                                coefs = results.best_estimator_[-1].estimator.coef_[0]
                                
                                # Create DataFrame for coefficients
                                coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
                                coef_df = coef_df.sort_values('Coefficient', ascending=False)
                                
                                # Plot feature importance
                                fig, ax = plt.subplots(figsize=(10, 8))
                                plt.barh(coef_df['Feature'].head(15), coef_df['Coefficient'].head(15), color='skyblue')
                                plt.xlabel('Coefficient Value')
                                plt.ylabel('Feature')
                                plt.title('Top 15 Feature Importances')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Show coefficient table
                                st.write("**Feature Coefficients:**")
                                st.dataframe(coef_df.head(20))
                        else:
                            st.info("Feature importance visualization is only available for linear models.")
                    except Exception as e:
                        st.warning(f"Could not extract feature importances: {str(e)}")
                    
                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred_test)
                    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Default", "Default"])
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cm_display.plot(ax=ax, cmap='Blues', values_format='d')
                    plt.title('Confusion Matrix on Test Data')
                    st.pyplot(fig)
                    
                    # Profit analysis
                    st.subheader("Profitability Analysis")
                    
                    # Calculate profit metrics
                    TN = sum((y_pred_test == 0) & (y_test == 0))  # Correctly predicted non-defaults
                    FN = sum((y_pred_test == 0) & (y_test == 1))  # Defaults incorrectly predicted as non-defaults
                    TP = sum((y_pred_test == 1) & (y_test == 1))  # Correctly predicted defaults
                    FP = sum((y_pred_test == 1) & (y_test == 0))  # Non-defaults incorrectly predicted as defaults
                    
                    # Create profit metrics
                    profit_metrics = {
                        "ROA on Correct Non-Defaults (TN)": TN * 0.02,
                        "Loss on Missed Defaults (FN)": FN * 0.20,
                        "Net Profit": TN * 0.02 - FN * 0.20,
                        "Opportunity Cost (FP)": FP * 0.02,  # Revenue missed by rejecting good loans
                    }
                    
                    # Create profit metrics DataFrame
                    profit_df = pd.DataFrame({
                        "Metric": profit_metrics.keys(),
                        "Value": profit_metrics.values()
                    })
                    
                    # Display profit metrics
                    st.table(profit_df)
                    
                    # Precision-Recall curve
                    st.subheader("Precision-Recall Curve")
                    
                    # Check if the model has predict_proba or decision_function
                    if hasattr(results.best_estimator_, 'predict_proba'):
                        y_scores = results.best_estimator_.predict_proba(X_test)[:,1]
                    elif hasattr(results.best_estimator_, 'decision_function'):
                        y_scores = results.best_estimator_.decision_function(X_test)
                    else:
                        y_scores = None
                    
                    if y_scores is not None:
                        # Plot precision-recall curve
                        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
                        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        pr_display.plot(ax=ax)
                        plt.title('Precision-Recall Curve')
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"An error occurred during model training: {str(e)}")
                    st.exception(e)
        
        elif train_model and pipe is None:
            st.error("Please select at least one feature before training the model.")
        
        else:
            st.info("Configure your model on the left and click 'Train Model' to see results here.")

################################################### Dictionary Section ########################################################

elif st.session_state['current_section'] == 'Dictionary':
    st.markdown("<h1 style='text-align: center;'>Data Dictionary</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;'>Numerical Features</h2>", unsafe_allow_html=True)
    numerical = {
        "annual_inc": "The self-reported annual income provided by the borrower during registration.",
        "dti": "Debt-to-income ratio: monthly debt payments divided by self-reported monthly income, excluding mortgage and the requested loan.",
        "emp_length": "Employment length in years. Values range from 0 (less than one year) to 10 (ten or more years). Note: 4.4% of values are missing.",
        "fico_range_high": "The upper boundary of the borrower's FICO score range at loan origination.",
        "fico_range_low": "The lower boundary of the borrower's FICO score range at loan origination.",
        "installment": "The monthly payment amount if the loan originates.",
        "int_rate": "Interest rate on the loan (percentage).",
        "loan_amnt": "The listed amount applied for by the borrower. May be reduced by the credit department during underwriting.",
        "mort_acc": "Number of mortgage accounts in the borrower's credit file.",
        "open_acc": "Number of open credit lines in the borrower's credit file.",
        "pub_rec": "Number of derogatory public records on the borrower's credit report.",
        "pub_rec_bankruptcies": "Number of public record bankruptcies on the borrower's credit report.",
        "revol_bal": "Total revolving credit balance.",
        "revol_util": "Revolving line utilization rate: amount of credit used relative to all available revolving credit. Note: Some missing values present.",
        "total_acc": "Total number of credit lines in the borrower's credit file.",
    }
    
    # Create a DataFrame for better display
    num_df = pd.DataFrame({"Variable": numerical.keys(), "Description": numerical.values()})
    st.table(num_df)

    st.markdown("<h2 style='text-align: center;'>Categorical Features</h2>", unsafe_allow_html=True)
    categorical = {
        "addr_state": "The state provided by the borrower in the loan application (49 unique values).",
        "grade": "LC assigned loan grade (7 values: A, B, C, D, E, F, G).",
        "home_ownership": "The borrower's home ownership status (Values: RENT, OWN, MORTGAGE).",
        "initial_list_status": "The initial listing status of the loan (W: Whole, F: Fractional).",
        "purpose": "The borrower's stated reason for the loan (13 categories including debt_consolidation, credit_card, home_improvement, etc.).",
        "sub_grade": "LC assigned loan subgrade (35 values from A1 to G5).",
        "term": "Loan duration in months (36 or 60).",
        "verification_status": "Income verification status (Verified, Not Verified, Source Verified).",
        "zip_code": "The first 3 digits of the borrower's zip code.",
    }
    
    # Create a DataFrame for better display
    cat_df = pd.DataFrame({"Variable": categorical.keys(), "Description": categorical.values()})
    st.table(cat_df)

    st.markdown("<h2 style='text-align: center;'>Model Types</h2>", unsafe_allow_html=True)
    model = {
        "Logistic Regression": "A linear model that estimates the probability of a binary outcome. Effective for credit risk modeling due to its interpretability and regularization capabilities.",
        "Linear SVC": "Support Vector Classifier that finds a hyperplane to maximize the margin between classes. Well-suited for high-dimensional data and robust to outliers.",
        "K-Nearest Neighbors": "A non-parametric algorithm that classifies based on proximity to training examples. Useful when decision boundaries are irregular.",
        "Decision Tree": "A flowchart-like model that makes decisions based on feature thresholds. Captures non-linear relationships and interactions between features.",
    }
    
    # Create a DataFrame for better display
    model_df = pd.DataFrame({"Model": model.keys(), "Description": model.values()})
    st.table(model_df)

    st.markdown("<h2 style='text-align: center;'>Feature Selection Methods</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Techniques to select the most relevant subset of features for model construction</h4>", unsafe_allow_html=True)
    
    selection = {
        "Passthrough": "Bypasses feature selection, using all available features.",
        "PCA": "Principal Component Analysis: Reduces dimensionality by creating new features (components) that capture maximum variance in the data.",
        "SelectKBest(f_classif)": "Selects top k features based on ANOVA F-value between features and target.",
        "SelectFromModel(LinearSVC)": "Uses L1 regularization in a LinearSVC model to identify and select the most important features.",
        "RFECV(LogisticRegression)": "Recursive Feature Elimination with Cross-Validation: Iteratively removes least important features based on model weights, optimizing via cross-validation.",
        "SequentialFeatureSelector": "Sequentially adds or removes features to find the optimal feature subset based on cross-validated performance.",
    }
    
    # Create a DataFrame for better display
    select_df = pd.DataFrame({"Method": selection.keys(), "Description": selection.values()})
    st.table(select_df)

    st.markdown("<h2 style='text-align: center;'>Feature Creation Methods</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Techniques to transform or generate new features from existing ones</h4>", unsafe_allow_html=True)
    
    creation = {
        "Passthrough": "Bypasses feature creation, using original features without transformation.",
        "PolynomialFeatures": "Generates polynomial and interaction terms from original features, capturing non-linear relationships.",
        "MinMaxScaler": "Scales features to a specific range (typically [0,1]) by subtracting minimum and dividing by range.",
        "MaxAbsScaler": "Scales features by their maximum absolute value to range [-1,1], preserving sparsity and sign.",
    }
    
    # Create a DataFrame for better display
    create_df = pd.DataFrame({"Method": creation.keys(), "Description": creation.values()})
    st.table(create_df)

    st.markdown("<h2 style='text-align: center;'>Model Evaluation</h2>", unsafe_allow_html=True)
    evaluation = {
        "Profit Score": "Custom metric representing expected profit: (true negatives × ROA) - (false negatives × haircut). Primary optimization target.",
        "Precision": "Proportion of predicted non-defaults that actually paid back loans. Important for minimizing losses.",
        "Recall": "Proportion of actual non-defaults correctly identified by the model. Important for maximizing revenue.",
        "F1-Score": "Harmonic mean of precision and recall, providing a balance between the two metrics.",
        "Confusion Matrix": "Visual representation showing true positives, false positives, true negatives, and false negatives.",
        "Precision-Recall Curve": "Plot showing the trade-off between precision and recall at different classification thresholds."
    }
    
    # Create a DataFrame for better display
    eval_df = pd.DataFrame({"Metric": evaluation.keys(), "Description": evaluation.values()})
    st.table(eval_df)


