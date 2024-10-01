import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.combine import SMOTEENN

# Load dataset automatically
@st.cache_data
def load_data():
    df = pd.read_csv('Customer_Churn.csv')
    return df

# Preprocessing function as per your code
def preprocess_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['tenure_group'] = pd.cut(df.tenure, range(1,80,12), right=False, 
                                labels=["1-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
    df = df.drop(columns=['customerID', 'tenure'])
    df_copy = pd.get_dummies(df, drop_first=True)
    return df, df_copy

# Load and preprocess the dataset
df = load_data()
df, df_copy = preprocess_data(df)

# Sidebar options with checkboxes for sections
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a section", ["Overview", "Plots", "Models"])

# --- OVERVIEW ---
if option == "Overview":
    st.header("Customer Churn Prediction")

    # Checkboxes for various overview items
    if st.sidebar.checkbox("Display top 5 rows"):
        st.subheader("Top 5 Rows")
        st.write(df.head())
    
    if st.sidebar.checkbox("Columns in the dataset"):
        st.subheader("Columns in the Dataset")
        st.write(df.columns.tolist())
    
    if st.sidebar.checkbox("Dataframe shape"):
        st.subheader("Dataframe Shape")
        st.write(df.shape)
    
    if st.sidebar.checkbox("Describe dataframe"):
        st.subheader("Dataframe Description")
        st.write(df.describe())
    
    if st.sidebar.checkbox("Churn value counts"):
        st.subheader("Churn Value Counts")
        st.write(df['Churn'].value_counts())

# --- PLOTS ---
elif option == "Plots":
    st.header("Plots")

    # Checkboxes for various plots
    if st.sidebar.checkbox("Bar Plot of Churn Counts"):
        st.subheader("Bar Plot of Churn Counts")
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax)
        st.pyplot(fig)
    
    if st.sidebar.checkbox("Count plots for features"):
        features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
        for feature in features:
            st.subheader(f"Count Plot for {feature}")
            fig, ax = plt.subplots()
            sns.countplot(x=feature, data=df, hue='Churn', ax=ax)
            st.pyplot(fig)
    
    if st.sidebar.checkbox("lmplot between MonthlyCharges and TotalCharges"):
        st.subheader("lmplot between MonthlyCharges and TotalCharges")
        fig = sns.lmplot(x='MonthlyCharges', y='TotalCharges', data=df_copy, fit_reg=False)
        st.pyplot(fig)
    
    if st.sidebar.checkbox("KDE Plot for MonthlyCharges by Churn"):
        st.subheader("KDE Plot for MonthlyCharges by Churn")
        fig, ax = plt.subplots()
        sns.kdeplot(df_copy['MonthlyCharges'][df_copy['Churn'] == 0], color='Red', fill=True, ax=ax)
        sns.kdeplot(df_copy['MonthlyCharges'][df_copy['Churn'] == 1], color='Blue', fill=True, ax=ax)
        ax.legend(['No Churn', 'Churn'])
        st.pyplot(fig)
    
    if st.sidebar.checkbox("KDE Plot for TotalCharges by Churn"):
        st.subheader("KDE Plot for TotalCharges by Churn")
        fig, ax = plt.subplots()
        sns.kdeplot(df_copy['TotalCharges'][df_copy['Churn'] == 0], color='Red', fill=True, ax=ax)
        sns.kdeplot(df_copy['TotalCharges'][df_copy['Churn'] == 1], color='Blue', fill=True, ax=ax)
        ax.legend(['No Churn', 'Churn'])
        st.pyplot(fig)
    
    if st.sidebar.checkbox("Correlation with Churn"):
        st.subheader("Correlation with Churn")
        corr_with_churn = df_copy.corr()['Churn'].sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        corr_with_churn.plot(kind='bar', ax=ax)
        st.pyplot(fig)

# --- MODELS ---
# elif option == "Models":
#     st.header("Model Evaluation Metrics")
    
#     # Features and target for model training
#     X = df_copy.drop('Churn', axis=1)
#     y = df_copy['Churn']
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
#     # Store model results for the table
#     model_results = []

#     # Decision Tree
#     if st.sidebar.checkbox("Decision Tree Classifier"):
#         dec_tree = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=6, min_samples_leaf=8)
#         dec_tree.fit(X_train, y_train)
#         y_pred_dt = dec_tree.predict(X_test)
#         accuracy_dt = accuracy_score(y_test, y_pred_dt)
#         model_results.append(["Decision Tree", accuracy_dt * 100])
    
#     # K-Nearest Neighbors
#     if st.sidebar.checkbox("K-Nearest Neighbors Classifier"):
#         knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
#         knn.fit(X_train, y_train)
#         y_pred_knn = knn.predict(X_test)
#         accuracy_knn = accuracy_score(y_test, y_pred_knn)
#         model_results.append(["K-Nearest Neighbors", accuracy_knn * 100])
    
#     # Random Forest
#     if st.sidebar.checkbox("Random Forest Classifier"):
#         rfc = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0, max_depth=6, min_samples_leaf=8)
#         rfc.fit(X_train, y_train)
#         y_pred_rfc = rfc.predict(X_test)
#         accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
#         model_results.append(["Random Forest", accuracy_rfc * 100])

#     # If at least one model is checked, display the results in a table
#     if model_results:
#         st.subheader("Model Scores")
#         model_results_df = pd.DataFrame(model_results, columns=["Model", "Accuracy (%)"])
#         st.table(model_results_df)

elif option == "Models":
    st.header("Model Evaluation")
    
    # Features and target for model training
    X = df_copy.drop('Churn', axis=1)
    y = df_copy['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Function to train and evaluate a model
    def train_and_evaluate(model, X_train, y_train, X_test, y_test, name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        return {
            "name": name,
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": pd.DataFrame(class_report).transpose()
        }

    # Function to apply SMOTEENN and train/evaluate
    def apply_smoteenn_and_evaluate(model, X, y, name):
        sm = SMOTEENN()
        X_sampled, y_sampled = sm.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=0)
        return train_and_evaluate(model, X_train, y_train, X_test, y_test, f"{name} with SMOTEENN")

    # Model definitions
    models = {
        "Decision Tree (DT)": DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=6, min_samples_leaf=8),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
        "Random Forest (RF)": RandomForestClassifier(n_estimators=100, criterion='gini', random_state=1, max_depth=6, min_samples_leaf=8)
    }

    # Model selection
    selected_models = st.multiselect("Select models to evaluate", list(models.keys()))

    if st.button("Train and Evaluate"):
        for name in selected_models:
            model = models[name]
            st.subheader(f"{name} Classifier")

            # Original model
            result = train_and_evaluate(model, X_train, y_train, X_test, y_test, name)
            
            st.write(f"Score (Accuracy): {result['accuracy']:.4%}")
            
            st.write("Confusion Matrix:")
            st.write(pd.DataFrame(result['confusion_matrix'], 
                                  columns=['Predicted Negative', 'Predicted Positive'],
                                  index=['Actual Negative', 'Actual Positive']))
            
            st.write("Classification Report:")
            st.table(result['classification_report'].style.format("{:.2f}"))

            # SMOTEENN
            st.subheader(f"{name} with SMOTEENN")
            smoteenn_result = apply_smoteenn_and_evaluate(model, X, y, name)
            
            st.write(f"Score (Accuracy): {smoteenn_result['accuracy']:.4%}")
            
            st.write("Confusion Matrix:")
            st.write(pd.DataFrame(smoteenn_result['confusion_matrix'], 
                                  columns=['Predicted Negative', 'Predicted Positive'],
                                  index=['Actual Negative', 'Actual Positive']))
            
            st.write("Classification Report:")
            st.table(smoteenn_result['classification_report'].style.format("{:.2f}"))

        # Feature importance for Random Forest (if selected)
        if "Random Forest (RF)" in selected_models:
            st.subheader("Random Forest Feature Importance")
            rf_model = models["Random Forest (RF)"]
            rf_model.fit(X_train, y_train)
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
            plt.title("Top 10 Feature Importance")
            st.pyplot(fig)
