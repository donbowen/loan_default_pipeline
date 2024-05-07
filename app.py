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
from sklearn.ensemble import HistGradientBoostingClassifier, VotingRegressor
from sklearn.feature_selection import (
    RFECV,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    f_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve,
    RocCurveDisplay,
    classification_report,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
    cross_val_score,
    check_cv,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    KBinsDiscretizer,
)
from sklearn.svm import LinearSVC
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import seaborn as sns
from sklearn.experimental import enable_hist_gradient_boosting
from scipy.sparse import csr_matrix #delete later, replace with effective spare to dense import fix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import os


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
            "Leaderboard of Previous Custom Models": "Leaderboard",
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

# drop some bad columns here, or in the pipeline
loans = loans.drop("id", axis = 1)

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
num_pipe_features = X_train.select_dtypes(include="float64").columns

# List of all categorical variables
cat_pipe_features = X_train.select_dtypes(include='object').columns  # all: X_train.select_dtypes(include='object').columns

def load_leaderboard():
    if os.path.exists('leaderboard.csv'):
        return pd.read_csv('leaderboard.csv')
    else:
        return pd.DataFrame(columns=['User Name', 'Model Name', 'Numerical Features', 'Categorical Features', 'Feature Selection Method', 'Feature Creation Method', 'F1-score'])

# Load the leaderboard at the start of the app
if 'leaderboard' not in st.session_state:
    st.session_state['leaderboard'] = load_leaderboard()

################################################## custom model code #################################################

# Function to create a pipeline based on user-selected model and features
def create_pipeline(model_name, feature_select, feature_create, num_pipe_features, cat_pipe_features, degree = None):
    if model_name == 'Logistic Regression':
        clf = LogisticRegression(class_weight='balanced', penalty='l2')
    elif model_name == 'Linear SVC':
        clf = LinearSVC(class_weight='balanced', penalty='l2')
    elif model_name == 'K-Nearest Neighbors':
        clf = KNeighborsClassifier(weights='uniform')
    elif model_name == 'Decision Tree':
        clf = DecisionTreeClassifier(class_weight = 'balanced')
    # Preprocessing pipelines for numerical and categorical features
    numer_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())

    cat_pipe = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
    
    # Preprocessing pipeline for the entire dataset
    # didn't use make_column_transformer; wanted to name steps
    preproc_pipe = make_column_transformer(
    (numer_pipe, num_pipe_features), 
    (cat_pipe, cat_pipe_features), 
    remainder="drop",
    )

# Define the feature selection transformer based on the selected method
    if feature_select == 'passthrough':
        feature_selector = 'passthrough'
    elif feature_select.startswith('PCA'):
        n_components = int(feature_select.split('(')[1].split(')')[0])
        feature_selector = TruncatedSVD(n_components=n_components)
    elif feature_select.startswith('SelectKBest'):
        feature_selector = SelectKBest(score_func=f_classif)
    elif feature_select.startswith('SelectFromModel'):
        if 'LinearSVC' in feature_select:
            class_weight = st.selectbox("Select class weight for LinearSVC", ['balanced', None])
            model = LinearSVC(penalty="l2", dual=False, class_weight=class_weight)
            feature_selector = SelectFromModel(model)
    elif feature_select.startswith('RFECV'):    
        if 'LogisticRegression' in feature_select:
            class_weight = st.selectbox("Select class weight for LogisticRegression", ['balanced', None])
            model = LogisticRegression(class_weight=class_weight)
    
        feature_selector = RFECV(model, cv=5, scoring=prof_score)
    elif feature_select.startswith('SequentialFeatureSelector'):
        model = None
        if 'LogisticRegression' in feature_select:
            class_weight = st.selectbox("Select class weight for LogisticRegression", ['balanced', None])
            model = LogisticRegression(class_weight=class_weight)
            
        scoring = prof_score
        feature_selector = SequentialFeatureSelector(model, scoring=scoring, n_features_to_select= 2, cv= 5)
    else:
        st.error("Invalid feature selection method!")
        return None

    # Define the feature creation transformer based on the selected method
    if feature_create == 'passthrough':
        feature_creator = 'passthrough'
    elif feature_create.startswith('PolynomialFeatures'):
        interaction_only = 'interaction_only' in feature_create
        feature_creator = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
    elif feature_create == 'MinMaxScaler':
        feature_creator = MinMaxScaler()
    elif feature_create == 'MaxAbsScaler':
        feature_creator = MaxAbsScaler()
        
    # I used "Pipeline" not "make_pipeline" bc I wanted to name the steps
    pipe = Pipeline([('columntransformer',preproc_pipe),
                 ('feature_create', feature_creator), 
                 ('feature_select', feature_selector), 
                 ('clf', clf)
                ])

    return pipe

################################################### Overview ########################################################

if st.session_state['current_section'] == 'Overview':

    st.markdown("<h1 style='text-align: center;'>Overview</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .centered-text {
        text-align: center;
        font-size: 16px; /* This size is typical for default body text in Streamlit */
    }
    </style>
    """, unsafe_allow_html=True)

    # Center and style the paragraph text using Markdown with custom HTML and CSS
    st.markdown("""
    <div class="centered-text">
    When a loan is taken out the lender takes on the risk that the borrower will default on their loan. The bigger question our team is interested in addressing is how various attributes related to loans affect the likelihood of loan defaults. So overall, we want to learn how to predict loan defaults given specifications for many important variables. The goal of the project is to compare combinations of predictor variables and classification models to find the best ways of predicting which borrowers will default on their loans.
</div>""", unsafe_allow_html=True)
    st.write("\n" * 5)

    st.subheader("Our Project:")
    st.write("""In this project our team built a dashboard allowing the user to select which predictor variables they would like to use in their model and which type of model and features they would like to select and create. Essentially the user is able to build their own pipeline and compare its effectiveness against other models run on our dashboard.""")
    st.write("\n" * 5)

    st.subheader("Type of ML model:")
    st.write("Classification model.")
    st.write("\n" * 5)

    st.subheader("Models:")
    st.write("Logistic Regression, Linear SVC, K-Nearest Neighbors, and Decision Tree")
    st.write("\n" * 5)

    st.subheader("Hypothesis:")
    st.write("Our hypothesis is that interest rate has the most significant impact on loan defaults compared to other common leading indicators.")
    st.write("\n" * 5)
    
    st.subheader("Data:")
    st.write("""We used the 2013 subsample csv provided in the machine learning folder. We have 134,804 observations of loan data with 33 data points. According to the loan status variable, of those observations, 113,780 loans are fully paid, while the remaining 21,024 are charged off (loan default).""")
    st.write("\n" * 5)

    st.subheader("Observation:")
    st.write("An observation is the ID given that each value represents a unique person and their corresponding conditions.")
    st.write("\n" * 5)

    st.subheader("Sample Period:")
    st.write("January 2013 – December 2013")
    st.write("\n" * 5)

    st.subheader("Predictor variables:")
    st.write("Check the dictionary tab to view all the variable options for the model")
    st.write("\n" * 5)

    st.subheader("Process:")
    st.write(""" After loading the csv file, we dropped the unnecessary columns, which were variables that wouldn't have made sense to include in any of the ML models. (See finaldataframe.ipynb in the source repository). We then split the data into training and testing data using an 80-20 split. Then we created a pipeline. In this pipeline we split the predictor variables into numerical and categorical values. We used One Hot Encoder to transform the categorical variables into numerical variables, so that it can be fed into the ML models. Then based on the method the user selects, we define the feature selection and feature creation transformers for each of the possible models. We also defined the hyperparameters we would like to maximize depending on the classification model. Based on user input, we created a function to construct a parameter grid, updated it with new hyperparameter ranges, and fit the grid to search our data. Finally, we plotted our results.""")
    st.write("\n" * 5)

    st.subheader("Results:")
    st.write("""Since we are working with a classification model, we used a decision matrix as our primary way of visualizing and analyzing the results.""")

################################################### custom model builder ########################################################

elif st.session_state['current_section'] == 'Custom Model Builder':
    
    # begin : user choices
    st.markdown("<h1 style='text-align: center;'>Build Your Own Custom Model</h1>", unsafe_allow_html=True)
    # num_pipe_features =  .... st.menu(list of choices or something);

    user_name = st.text_input("Enter Your Name:", key='user_name')
    # Checkbox to select numerical features

    
    selected_num_features = st.multiselect("Select Numerical Features:", num_pipe_features, key='selected_num_features')
    
    # Checkbox to select categorical features
    selected_cat_features = st.multiselect("Select Categorical Features:", cat_pipe_features, key='selected_cat_features')
        
    # Dropdown menu to choose the model
    model_options = ['Logistic Regression', 'Linear SVC', 'K-Nearest Neighbors', 'Decision Tree']
    model_name = st.selectbox("Choose Model:", model_options, key='selected_model')

    # Dropdown menu to choose the feature selection method
    feature_select_options = ['passthrough', 'PCA', 'SelectKBest(f_classif)', 'SelectFromModel(LinearSVC(penalty="l1", dual=False))', 'RFECV(LogisticRegression, scoring=prof_score)', 'SequentialFeatureSelector(LogisticRegression, scoring=prof_score)',]
    feature_select_method = st.selectbox("Choose Feature Selection Method:", feature_select_options, key='selected_feature_selection')
    
    # Dropdown menu to choose the feature creation method
    feature_create_options = ['passthrough', 'PolynomialFeatures', 'MinMaxScaler', 'MaxAbsScaler']
    feature_create_method = st.selectbox("Choose Feature Creation Method:", feature_create_options, key='selected_feature_creation')
    
    # If PolynomialFeatures is selected, provide an input field to specify the degree
    if feature_create_method == 'PolynomialFeatures':
        degree = st.number_input("Enter the degree for PolynomialFeatures", min_value=1, max_value=5, value=2)
    else:
        degree = None

    hyperparameter_ranges = {}        

    if model_name in ['Linear SVC', 'Logistic Regression']:
        C_min = st.slider('C - Min Value', min_value=0.1, max_value=10.0, value=1.0)
        C_max = st.slider('C - Max Value', min_value=0.1, max_value=10.0, value=5.0)
        hyperparameter_ranges['C'] = np.linspace(C_min, C_max, num=10) 
    elif model_name == 'K-Nearest Neighbors':
        n_neighbors_min = st.slider('Number of Neighbors - Min Value', min_value=1, max_value=20, value=3)
        n_neighbors_max = st.slider('Number of Neighbors - Max Value', min_value=1, max_value=20, value=10)
        hyperparameter_ranges['n_neighbors'] = list(range(n_neighbors_min, n_neighbors_max + 1))
    elif model_name == 'Decision Tree':
        min_split_min = st.slider('Min Samples Split - Min Value', min_value=2, max_value=50, value=2)
        min_split_max = st.slider('Min Samples Split - Max Value', min_value=2, max_value=50, value=10)
        hyperparameter_ranges['min_samples_split'] = list(range(min_split_min, min_split_max + 1))
        
    if feature_select_method in ['SelectKBest(f_classif)']:
        selectkbest_k_min = st.slider('SelectKBest - Min K', min_value=1, max_value=50, value=5)
        selectkbest_k_max = st.slider('SelectKBest - Max K', min_value=1, max_value=50, value=25)
        selectkbest_k_step = st.slider('SelectKBest - Step Size', min_value=1, max_value=10, value=5)
        hyperparameter_ranges['k'] = np.arange(selectkbest_k_min, selectkbest_k_max + 1, selectkbest_k_step)    
    elif feature_select_method in ['PCA']:
        n_components_min = st.slider('PCA - Min Number of Components', min_value=1, max_value=100, value=5)
        n_components_max = st.slider('PCA - Max Number of Components', min_value=1, max_value=100, value=25)
        hyperparameter_ranges['n_components'] = np.arange(n_components_min, n_components_max + 1) 
    # elif feature_select_method == 'SelectFromModel(LinearSVC(penalty="l1", dual=False))':    
    elif feature_select_method in ['SequentialFeatureSelector(LogisticRegression, scoring=prof_score)']:
        n_features_min = st.slider('Minimum Number of Features for SequentialFeatureSelector', min_value=1, max_value=50, value=5)
        n_features_max = st.slider('Maximum Number of Features for SequentialFeatureSelector', min_value=1, max_value=50, value=25)
        hyperparameter_ranges['n_features_to_select'] = np.arange(n_features_min, n_features_max + 1)    
    elif feature_select_method in ['RFECV(LogisticRegression, scoring=prof_score)']:
        step_min = st.slider('RFECV Step - Min Value', min_value=1, max_value=10, value=1)
        step_max = st.slider('RFECV Step - Max Value', min_value=1, max_value=10, value=5)
        hyperparameter_ranges['step'] = np.arange(step_min, step_max + 1)
    else:
        hyperparameter_ranges = None
    
    # Create the pipeline based on the selected model and features
    pipe = create_pipeline(model_name, feature_select_method, feature_create_method, selected_num_features, selected_cat_features, degree)
    
    # Dropdown menu to choose the cross-validation strategy
    num_folds = st.number_input("Enter the number of folds for cross-validation", min_value=2, max_value=10, value=5)

    # Define your cross-validation strategy based on the user input
    cv = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # end: user choices
    ##################################################
    
    # User choice outputs
    
    pipe

    param_grid = {}

        # Function to construct parameter grid based on user input
    def construct_param_grid(feature_selection_method, model, hyperparameter_ranges):
        
        if feature_selection_method == 'SelectKBest(f_classif)':
            param_grid['feature_select__k'] = hyperparameter_ranges['k']
        elif feature_selection_method == 'PCA':
            param_grid['feature_select__n_components'] = hyperparameter_ranges['n_components']
        elif feature_selection_method == 'SequentialFeatureSelector(LogisticRegression, scoring=prof_score)':
            param_grid['feature_select__n_features_to_select'] = hyperparameter_ranges['n_features_to_select']
        elif feature_selection_method == 'RFECV(LogisticRegression, scoring=prof_score)':
            param_grid['feature_select__step'] = hyperparameter_ranges['step']
        
        if model in ['Logistic Regression', 'Linear SVC']:
            param_grid['clf__C'] = hyperparameter_ranges['C']
        elif model == 'K-Nearest Neighbors':
            param_grid['clf__n_neighbors'] = hyperparameter_ranges['n_neighbors']
        elif model == 'Decision Tree':
            param_grid['clf__min_samples_split'] = hyperparameter_ranges['min_samples_split']
        
        return param_grid
    
    
    # Update parameter grid with new hyperparameter ranges
    param_grid = construct_param_grid(feature_select_method, model_name, hyperparameter_ranges)

    st.write(param_grid)
    
    grid_search = GridSearchCV(estimator = pipe, 
                           param_grid = param_grid,
                           cv = cv,
                           scoring= prof_score, 
                           error_score="raise",
                           )

    # Fit the grid search to your data
    try:
        results = grid_search.fit(X_train, y_train)
    except Exception as e:
        # Report the resulting error traceback
        st.write("An error occurred during grid search fitting:")
        st.write(e)
        
    st.write("\n" * 5)
    st.markdown("<h1 style='text-align: center;'>Ranking CV Test Scores by Mean and SD </h1>", unsafe_allow_html=True)
    output_df = pd.DataFrame(results.cv_results_).set_index('params').fillna('')
    st.write(output_df)

    # Create a new figure and axis object using Matplotlib's object-oriented interface
    fig, ax = plt.subplots()
    
    # Plot the scatter plot
    scatter = ax.scatter(output_df['std_test_score'], output_df['mean_test_score'], color='blue')
    ax.scatter(output_df['std_test_score'][0], output_df['mean_test_score'][0], color='red')
    
    # Set the plot title and labels
    ax.set_title("Mean vs STD of CV Test Scores")
    ax.set_ylabel("Mean Test Score")
    ax.set_xlabel("STD Test Score")
    
    # Show the plot
    st.pyplot(fig)


    # Get the best estimator and predictions
    best_estimator = results.best_estimator_
    y_pred_train = results.predict(X_train)

    if model_name in ["Logistic Regression", "Linear SVC", "K-Nearest Neighbors", "Decision Tree"]:
        # Calculate classification report
        report = classification_report(y_train, y_pred_train, output_dict=True)
        
        # Create a formatted classification report string
        # classification_report_str = """

       #     |          | Precision | Recall | F1-Score | Support |
       #     |----------|-----------|--------|----------|---------|
       #     | False    |   {:.4f}  | {:.4f} |   {:.4f} |   {:<6} |         # Replaced with centered format below
       #     | True     |   {:.4f}  | {:.4f} |   {:.4f} |   {:<6} |
       #     | Accuracy |           |        |   {:.4f} |         |
      #  """.format(report["False"]["precision"], report["False"]["recall"], report["False"]["f1-score"], report["False"]["support"],
      #             report["True"]["precision"], report["True"]["recall"], report["True"]["f1-score"], report["True"]["support"],
       #            report["accuracy"])
        
        F1score = report['True']['f1-score']
        st.session_state['model_F1score'] = F1score
    

        classification_report_str = f"""
        <div style="text-align: center; width: 100%;">
            <table style="margin-left: auto; margin-right: auto;">
                <tr>
                    <th></th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
                <tr>
                    <td>False</td>
                    <td>{report["False"]["precision"]:.4f}</td>
                    <td>{report["False"]["recall"]:.4f}</td>
                    <td>{report["False"]["f1-score"]:.4f}</td>
                    <td>{report["False"]["support"]}</td>
                </tr>
                <tr>
                    <td>True</td>
                    <td>{report["True"]["precision"]:.4f}</td>
                    <td>{report["True"]["recall"]:.4f}</td>
                    <td>{report["True"]["f1-score"]:.4f}</td>
                    <td>{report["True"]["support"]}</td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td></td>
                    <td></td>
                    <td>{report["accuracy"]:.4f}</td>
                    <td></td>
                </tr>
            </table>
        </div>
    """
        
        # Display classification report
        st.write("\n" * 5)
        st.markdown("<h1 style='text-align: center;'>Classification Report</h1>", unsafe_allow_html=True)
        st.markdown(classification_report_str, unsafe_allow_html=True)

        # Assuming y_true_train and y_pred_train are your true and predicted labels for the training set
        precision, recall, _ = precision_recall_curve(y_train, y_pred_train)
        
        # Create a PrecisionRecallDisplay object
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
        
        # Create a new figure and axis object using Matplotlib's object-oriented interface
        fig, ax = plt.subplots()
        
        # Plot the Precision-Recall curve on the specified axis
        pr_display.plot(ax=ax)
        
        # Set labels and title
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        
        # Save the figure to a file
        plt.savefig('precision_recall_curve.png')
        
        # Display the plot in Streamlit
        st.write("\n" * 5)
        st.markdown("<h1 style='text-align: center;'>Precision Recall</h1>", unsafe_allow_html=True)
        st.image('precision_recall_curve.png')
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_train, y_pred_train)
        
        # Display confusion matrix
        st.write("\n" * 5)
        st.markdown("<h1 style='text-align: center;'>Confusion Matrix</h1>", unsafe_allow_html=True)
        confusion_matrix_chart = ConfusionMatrixDisplay(cm).plot()
        st.pyplot(confusion_matrix_chart.figure_)




    
    # Function to save model results and selections
    def run_model():
        user_name = st.session_state.get('user_name', 'Anonymous')
        model_name = st.session_state.get('selected_model', 'Default Model')
        numerical_features = ', '.join(st.session_state.get('selected_num_features', []))
        categorical_features = ', '.join(st.session_state.get('selected_cat_features', [])) 
        feature_select_method = st.session_state.get('selected_feature_selection', 'Default Selection')
        feature_create_method = st.session_state.get('selected_feature_creation', 'Default Creation')
        F1score = st.session_state.get('model_F1score', 0)  # Placeholder for where you calculate accuracy
        
        new_entry = pd.DataFrame([{
            'User Name': user_name,
            'Model Name': model_name,
            'Numerical Features': numerical_features,
            'Categorical Features': categorical_features,
            'Feature Selection Method': feature_select_method,
            'Feature Creation Method': feature_create_method,
            'F1-score': F1score
        }])

        if 'leaderboard' not in st.session_state:
            st.session_state['leaderboard'] = pd.DataFrame(columns=list(new_entry.keys()))
    
        st.session_state['leaderboard'] = pd.concat([st.session_state['leaderboard'], new_entry], ignore_index=True)
        st.session_state['leaderboard'].to_csv('leaderboard.csv', index=False)
        st.success('Model results saved to leaderboard.')
        
    if st.button('Done'):
        run_model()

################################################### Leaderboard ########################################################

elif st.session_state['current_section'] == 'Leaderboard':

    st.markdown("<h1 style='text-align: center;'>Leaderboard</h1>", unsafe_allow_html=True)
    st.header("Compare your model to previous ones ranked by their performance")
    
    if 'leaderboard' in st.session_state and not st.session_state.leaderboard.empty:
        sorted_leaderboard = st.session_state['leaderboard'].sort_values(by='F1-score', ascending=False).reset_index(drop=True)
        sorted_leaderboard.index = np.arange(1, len(sorted_leaderboard) + 1)
        st.dataframe(sorted_leaderboard)
    else:
        st.write("No leaderboard data available.")
    
################################################### Dictionary ########################################################

elif st.session_state['current_section'] == 'Dictionary':
    st.markdown("<h1 style='text-align: center;'>Dictionary</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;'>Numerical Features:</h2>", unsafe_allow_html=True)
    numerical = {
        "annual_inc": "The self-reported annual income provided by the borrower during registration.",
        "dti": "A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.",
        "earliest_cr_line": "The month the borrower's earliest reported credit line was opened",
        "emp_length": "Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. (WARNING: 5962 or 4.4227% of the fields are missing)",
        "fico_range_high": "The upper boundary range the borrower’s FICO at loan origination belongs to.",
        "fico_range_low": "The lower boundary range the borrower’s FICO at loan origination belongs to.",
        "installment": "The monthly payment owed by the borrower if the loan originates.",
        "int_rate": "Interest Rate on the loan.",
        "loan_amnt": "The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.",
        "mort_acc": "Number of mortgage accounts.",
        "open_acc": "The number of open credit lines in the borrower's credit file.",
        "pub_rec": "Number of derogatory public records",
        "pub_rec_bankruptcies": "Number of public record bankruptcies",
        "revol_bal": "Total credit revolving balance",
        "revol_util": "Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit. (WARNING: 78 fields with missing values)",
        "total_acc": "The total number of credit lines currently in the borrower's credit file",
    }
    for term, definition in numerical.items():
        col1, col2 = st.columns([1, 8])  # Adjust the ratio if needed to accommodate your content
        with col1:
            st.markdown(f"<div style='text-align: right; font-weight: bold;'>{term}</div>", unsafe_allow_html=True)
        with col2:
            st.write(definition)

    st.markdown("<h2 style='text-align: center;'>Categorical Features:</h2>", unsafe_allow_html=True)
    categorical = {
        "addr_state": "The state provided by the borrower in the loan application (49 values)",
        "grade": "LC assigned loan grade (7 values: A, B, C, D, E, F, G)",
        "home_ownership": "The home ownership status provided by the borrower during registration or obtained from the credit report. Values: RENT, OWN, MORTGAGE",
        "initial_list_status": "The initial listing status of the loan. Possible values are – W, F",
        "issue_d": "The month which the loan was funded (values include all 12 months)",
        "purpose": "A category provided by the borrower for the loan request (13 values: debt_consolidation, credit_card, home_improvement, other, major_purchase, small_business, car, medical, house, moving, wedding, vacation, renewable_energy)",
        "sub_grade": "LC assigned loan subgrade (35 values: A1, A2,...  …G3, G4, G5)",
        "term": "The number of payments on the loan. Values are in months and can be either 36 or 60. (36 months or 60 months)",
        "verification_status": "Indicates if income was verified by LC, not verified, or if the income source was verified (3 values: Verified, Not Verified, Source Verified)",
        "zip_code": "The first 3 numbers of the zip code provided by the borrower in the loan application. (834 values)",
    }
    for term, definition in categorical.items():
        col1, col2 = st.columns([1, 8])  # Adjust the ratio if needed to accommodate your content
        with col1:
            st.markdown(f"<div style='text-align: right; font-weight: bold;'>{term}</div>", unsafe_allow_html=True)
        with col2:
            st.write(definition)


    st.markdown("<h2 style='text-align: center;'>Model:</h2>", unsafe_allow_html=True)
    model = {
        "Logistic Regression": "A supervised machine learning algorithm for a binary classification problem that produces a probability that an instance belongs to a given class.",
        "Linear SVC": "A supervised machine learning algorithm that finds a hyperplane that maximally separates the different classes in the data.",
        "K-Nearest Neighbors": "A non-parametric supervised machine learning algorithm that finds a certain number of nearest points based on a distance metric, such as a Euclidean distance.",
        "Decision Tree": "A supervised machine learning algorithm that creates a flowchart-like tree structure where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label. It is constructed by recursively splitting the training data into subsets based on the values of the attributes until a stopping criterion is me",
    }
    for term, definition in model.items():
        col1, col2 = st.columns([1, 5])  # Adjust the ratio if needed to accommodate your content
        with col1:
            st.markdown(f"<div style='text-align: right; font-weight: bold;'>{term}</div>", unsafe_allow_html=True)
        with col2:
            st.write(definition)


    st.markdown("<h2 style='text-align: center;'>Feature Selection:</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Feature selection is the process of choosing a subset of relevant features (variables, predictors) for use in model construction</h4>", unsafe_allow_html=True)

    st.write("\n" * 5)
    
    selection = {
        "Passthrough": "Skips over the Feature Selection.",
        "PCA": "PCA stands for  Principal Component Analysis. It is used for dimensionality reduction, and indirectly performs feature selection by identifying the most important features (components) that capture the maximum variance in the data.",
        "SelectKBest(f_classif)": "A feature selection technique that selects the k best features based on a scoring function.",
        "SelectFromModel(LinearSVC...)": "A feature selection method that selects features based on the importance given by an underlying model. The penalty specifies the use of L1 regularization, which encourages sparsity in the feature weights. By setting dual = False it utilizes the primal optimization problem, which is preferred when the number of samples is smaller than the number of features",
        "RFECV(LogisticRegression, scoring=prof_score)": "The Recursive Feature Elimination with Cross-Validation method recursively removes the least important features and selects the optimal subset of features based on cross-validation performance. It evaluates the performance of the model with different subsets of features using cross-validation, selecting the subset that maximizes the specified scoring metric",
        "SequentialFeatureSelector(...)": "Selects features by iteratively adding or removing them based on their individual contribution to the model's performance. In each iteration, it evaluates the performance of the model with different subsets of features using cross-validation, selecting the subset that maximizes or minimizes the specified scoring metric, depending on whether it's performing forward or backward selection.",
    }
    for term, definition in selection.items():
        col1, col2 = st.columns([1, 5])  # Adjust the ratio if needed to accommodate your content
        with col1:
            st.markdown(f"<div style='text-align: right; font-weight: bold;'>{term}</div>", unsafe_allow_html=True)
        with col2:
            st.write(definition)

    st.markdown("<h2 style='text-align: center;'>Features Creation:</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Feature Creation is the process of transforming raw data into features that better represent the underlying problem to the predictive models, thus improving their performance</h4>", unsafe_allow_html=True)
    creation = {
        "Passthrough": "Skips over the Feature Creation",
        "PolynomialFeatures": "Transforms input features by generating polynomial combinations of them, up to a specified degree",
        "MinMaxScaler": "It scales and transforms the features such that they are mapped to a specified range, typically between 0 and 1. This scaling is achieved by subtracting the minimum value of each feature and then dividing by the range (maximum value minus minimum value) of that feature.",
        "MaxAbsScaler": "It scales and transforms the features such that the absolute values of each feature are mapped to the range [-1, 1]. It is a useful tool for ensuring that features are on a consistent scale, making it easier for machine learning models to learn from the data without being biased by the scale of the features. It's especially beneficial when dealing with sparse data or when you want to preserve the sign of the feature values.",
    }
    for term, definition in creation.items():
        col1, col2 = st.columns([1, 5])  # Adjust the ratio if needed to accommodate your content
        with col1:
            st.markdown(f"<div style='text-align: right; font-weight: bold;'>{term}</div>", unsafe_allow_html=True)
        with col2:
            st.write(definition)
