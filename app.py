import streamlit as st
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# imbalanced-learn methods (optional)
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    imblearn_available = True
except ImportError:
    imblearn_available = False

st.set_page_config(page_title="Dry Bean Type Classifier", layout='wide')

st.markdown("""
<style>
    body {background-color: #f4f4f4; color: #000;}
    .stApp {background-color: #d9d9d9;}
    .css-18e3th9 {background-color: #f2f2f2;}
    .css-ffhzg2 {color: #000;}
    .stButton>button {background-color: #4f4f4f; color: #fff;}
    .stTextInput>div>div>input {background-color: #eaeaea; color: #000;}
</style>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Beans_Multiclass_Classification.csv')
    return data

data = load_data()

# Display the first few rows
st.header("Dataset Head")
st.dataframe(data.head())

# Display info
st.header("Dataset Info")
buffer = io.StringIO()
data.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

# Display describe
st.header("Dataset Describe")
st.dataframe(data.describe())

# EDA Visualizations
st.header("Exploratory Data Analysis")

# Visualize distributions of features using histograms and boxplots
st.subheader("Feature Distributions: Histograms and Boxplots")
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Class' in numerical_cols:
    numerical_cols.remove('Class')  # Assuming Class is categorical

for col in numerical_cols[:5]:  # Limit to first 5 to avoid too many plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data[col], ax=ax1, kde=True)
    ax1.set_title(f'Histogram of {col}')
    sns.boxplot(x=data[col], ax=ax2)
    ax2.set_title(f'Boxplot of {col}')
    st.pyplot(fig)

# Analyze the class distribution (check for class imbalance)
st.subheader("Class Distribution")
if 'Class' in data.columns:
    fig, ax = plt.subplots()
    data['Class'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.write("Class counts:")
    st.write(data['Class'].value_counts())

# Plot feature correlations (heatmap)
st.subheader("Feature Correlations Heatmap")
corr = data[numerical_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

# Visualize multivariate relationships (pairplot)
st.subheader("Multivariate Relationships: Pairplot")
# Select a subset of features for pairplot to avoid performance issues
subset_cols = numerical_cols[:4] + ['Class'] if 'Class' in data.columns else numerical_cols[:5]
pairplot_data = data[subset_cols]
fig = sns.pairplot(pairplot_data, hue='Class' if 'Class' in data.columns else None)
st.pyplot(fig)

# Missing values check and handling
st.subheader("Missing Values & Handling")
missing_counts = data.isna().sum()
missing_counts = missing_counts[missing_counts > 0]
if missing_counts.empty:
    st.success("No missing values found in the dataset.")
else:
    st.warning("Missing values detected:")
    st.write(missing_counts)
    # Basic handling: numeric -> median, categorical -> mode
    data_cleaned = data.copy()
    for col in data_cleaned.columns:
        if data_cleaned[col].isna().any():
            if data_cleaned[col].dtype in ['float64', 'int64']:
                median_val = data_cleaned[col].median()
                data_cleaned[col].fillna(median_val, inplace=True)
            else:
                mode_val = data_cleaned[col].mode(dropna=True)
                if not mode_val.empty:
                    data_cleaned[col].fillna(mode_val[0], inplace=True)
    st.write("Missing values have been imputed (numerical=median, categorical=mode).")

# Outlier detection and treatment
st.subheader("Outlier Detection & Treatment")
if 'data_cleaned' not in locals():
    data_cleaned = data.copy()

# IQR-based detection
outlier_report = {}
for col in numerical_cols:
    q1 = data_cleaned[col].quantile(0.25)
    q3 = data_cleaned[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data_cleaned[(data_cleaned[col] < lower_bound) | (data_cleaned[col] > upper_bound)]
    outlier_report[col] = len(outliers)

st.write("Number of IQR outliers per numeric feature:")
st.write(outlier_report)

# Option: clip outliers by bounds
for col in numerical_cols:
    q1 = data_cleaned[col].quantile(0.25)
    q3 = data_cleaned[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data_cleaned[col] = data_cleaned[col].clip(lower_bound, upper_bound)

st.write("Outliers trimmed using IQR clipping for numerical features.")

# Boxplots after outlier treatment for first 5 features
st.subheader("Boxplots After IQR Outlier Clipping")
for col in numerical_cols[:5]:
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x=data_cleaned[col], ax=ax)
    ax.set_title(f'Boxplot of {col} (after clipping)')
    st.pyplot(fig)

# Feature Engineering & Preprocessing
st.header("Feature Engineering & Preprocessing")

# Identify categorical and numeric features
categorical_cols = data_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
if 'Class' in categorical_cols:
    categorical_cols.remove('Class')

st.subheader("Categorical Encoding")
if categorical_cols:
    st.write("Categorical columns:", categorical_cols)
    data_encoded = data_cleaned.copy()
    # One-hot encode all categorical features except target
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    ohe_df = pd.DataFrame(ohe.fit_transform(data_encoded[categorical_cols]),
                          columns=ohe.get_feature_names_out(categorical_cols),
                          index=data_encoded.index)
    data_encoded = pd.concat([data_encoded.drop(columns=categorical_cols), ohe_df], axis=1)
else:
    st.write("No categorical features to encode.")
    data_encoded = data_cleaned.copy()

# Encode target if needed
target_col = 'Class' if 'Class' in data_encoded.columns else None
if target_col and data_encoded[target_col].dtype == 'object':
    le = LabelEncoder()
    data_encoded[target_col] = le.fit_transform(data_encoded[target_col])
    st.write("Encoded target class labels.")

# Check and treat skewness
st.subheader("Skewness Check & Treatment")
skewed_feats = data_encoded[numerical_cols].skew().abs()
skewed_cols = skewed_feats[skewed_feats > 1].index.tolist()
if skewed_cols:
    st.write("Highly skewed numeric features:", skewed_cols)
    # apply log1p transform to skewed features (handle negatives by shift if needed)
    data_encoded[skewed_cols] = data_encoded[skewed_cols].apply(lambda x: np.log1p(x - x.min() + 1) if (x <= 0).any() else np.log1p(x))
    st.write("Applied log1p transform to skewed features.")
else:
    st.write("No highly skewed numeric features detected.")

# Scale numeric features
st.subheader("Scaling Numeric Features")
scaler = StandardScaler()
scaled_num = pd.DataFrame(scaler.fit_transform(data_encoded[numerical_cols]),
                          columns=numerical_cols,
                          index=data_encoded.index)

# Replace numeric cols with scaled values
data_scaled = data_encoded.copy()
data_scaled[numerical_cols] = scaled_num
st.write("StandardScaler applied to numeric features.")

# Train/test split with stratified sampling
st.subheader("Train/Test Split")
if target_col:
    X = data_scaled.drop(columns=[target_col])
    y = data_scaled[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)
    st.write("Train/Test split completed using stratified sampling on target.")
    st.write("X_train shape:", X_train.shape)
    st.write("X_test shape:", X_test.shape)

    # Model Training and Evaluation
    st.header("Model Building and Evaluation")
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = []
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred)

        test_prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        test_rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        gap = train_acc - test_acc
        results.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'F1 (weighted)': test_f1,
            'Accuracy Gap': gap,
            'Overfitting (Y/N)': 'Y' if gap > 0.05 else 'N',
            'Precision (weighted)': test_prec,
            'Recall (weighted)': test_rec,
            'CV Accuracy Mean': cv_scores.mean(),
            'CV Accuracy Std': cv_scores.std()
        })

    results_df = pd.DataFrame(results).sort_values(by='Test Accuracy', ascending=False)
    st.subheader("Models Performance")
    st.dataframe(results_df)

    # Overfitting check summary
    st.subheader("Overfitting Check")
    overfit_df = results_df[['Model','Train Accuracy','Test Accuracy','Accuracy Gap']]
    st.dataframe(overfit_df)

    # Confusion matrix for best model on test set
    best_model_name = results_df.iloc[0]['Model']
    best_model = classifiers[best_model_name]
    best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {best_model_name}')
    st.pyplot(fig)

    st.subheader("Best Model Classification Report")
    st.text(classification_report(y_test, best_pred, zero_division=0))

    # Hyperparameter Tuning for top models
    st.header("Hyperparameter Tuning")
    top_models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM (RBF Kernel)': SVC(probability=True, random_state=42)
    }

    tuning_results = []

    # RF GridSearch
    rf_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    gs_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    gs_rf.fit(X_train, y_train)
    rf_best = gs_rf.best_estimator_
    rf_best_score = gs_rf.best_score_
    y_pred_rf_best = rf_best.predict(X_test)
    rf_test_acc = accuracy_score(y_test, y_pred_rf_best)
    tuning_results.append({
        'Model': 'Random Forest',
        'Best Params': gs_rf.best_params_,
        'CV Best Score': rf_best_score,
        'Test Accuracy': rf_test_acc
    })

    # Gradient Boosting RandomizedSearch
    gb_param_dist = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    rs_gb = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), gb_param_dist, n_iter=6, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    rs_gb.fit(X_train, y_train)
    gb_best = rs_gb.best_estimator_
    gb_best_score = rs_gb.best_score_
    y_pred_gb_best = gb_best.predict(X_test)
    gb_test_acc = accuracy_score(y_test, y_pred_gb_best)
    tuning_results.append({
        'Model': 'Gradient Boosting',
        'Best Params': rs_gb.best_params_,
        'CV Best Score': gb_best_score,
        'Test Accuracy': gb_test_acc
    })

    # SVM RandomizedSearch
    svm_param_dist = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf']
    }
    rs_svm = RandomizedSearchCV(SVC(probability=True, random_state=42), svm_param_dist, n_iter=6, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    rs_svm.fit(X_train, y_train)
    svm_best = rs_svm.best_estimator_
    svm_best_score = rs_svm.best_score_
    y_pred_svm_best = svm_best.predict(X_test)
    svm_test_acc = accuracy_score(y_test, y_pred_svm_best)
    tuning_results.append({
        'Model': 'SVM (RBF)',
        'Best Params': rs_svm.best_params_,
        'CV Best Score': svm_best_score,
        'Test Accuracy': svm_test_acc
    })

    tuning_df = pd.DataFrame(tuning_results)
    st.subheader("Hyperparameter Tuning Results")
    st.dataframe(tuning_df)

    # Compare top model with baseline
    st.write("Baseline best model:", best_model_name, "Test acc:", results_df.iloc[0]['Test Accuracy'])
    st.write("Tuned Random Forest test acc:", rf_test_acc)
    st.write("Tuned Gradient Boosting test acc:", gb_test_acc)
    st.write("Tuned SVM test acc:", svm_test_acc)

    # Ensemble model w/ Voting
    st.subheader("Voting Classifier (Ensemble)")
    voting_clf = VotingClassifier(
        estimators=[('lr', classifiers['Logistic Regression']),
                    ('rf', classifiers['Random Forest']),
                    ('gb', classifiers['Gradient Boosting'])],
        voting='soft')
    voting_clf.fit(X_train, y_train)
    y_pred_vote = voting_clf.predict(X_test)
    st.write("Voting classifier test accuracy:", accuracy_score(y_test, y_pred_vote))

    # Handling Class Imbalance
    st.header("Handling Class Imbalance")
    class_counts = y.value_counts()
    st.write("Class distribution before balancing:")
    st.write(class_counts)

    if not imblearn_available:
        st.warning("imbalanced-learn is not available; install with 'pip install imbalanced-learn' to run SMOTE/oversampling/undersampling.")
    else:
        imbalance_results = []

        # SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_smote.fit(X_smote, y_smote)
        y_pred_smote = rf_smote.predict(X_test)
        report_smote = classification_report(y_test, y_pred_smote, output_dict=True, zero_division=0)
        imbalance_results.append({'Method': 'SMOTE', 'Test Accuracy': accuracy_score(y_test, y_pred_smote), 'Minority Avg F1': np.mean([report_smote[c]['f1-score'] for c in report_smote if c not in ['accuracy','macro avg','weighted avg']])})

        # Random Oversample
        ros = RandomOverSampler(random_state=42)
        X_ros, y_ros = ros.fit_resample(X_train, y_train)
        rf_ros = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_ros.fit(X_ros, y_ros)
        y_pred_ros = rf_ros.predict(X_test)
        report_ros = classification_report(y_test, y_pred_ros, output_dict=True, zero_division=0)
        imbalance_results.append({'Method': 'Random Oversampling', 'Test Accuracy': accuracy_score(y_test, y_pred_ros), 'Minority Avg F1': np.mean([report_ros[c]['f1-score'] for c in report_ros if c not in ['accuracy','macro avg','weighted avg']])})

        # Random Undersample
        rus = RandomUnderSampler(random_state=42)
        X_rus, y_rus = rus.fit_resample(X_train, y_train)
        rf_rus = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_rus.fit(X_rus, y_rus)
        y_pred_rus = rf_rus.predict(X_test)
        report_rus = classification_report(y_test, y_pred_rus, output_dict=True, zero_division=0)
        imbalance_results.append({'Method': 'Random Undersampling', 'Test Accuracy': accuracy_score(y_test, y_pred_rus), 'Minority Avg F1': np.mean([report_rus[c]['f1-score'] for c in report_rus if c not in ['accuracy','macro avg','weighted avg']])})

        # Class Weighting in RF
        rf_weighted = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf_weighted.fit(X_train, y_train)
        y_pred_weighted = rf_weighted.predict(X_test)
        report_weighted = classification_report(y_test, y_pred_weighted, output_dict=True, zero_division=0)
        imbalance_results.append({'Method': 'Class Weighting', 'Test Accuracy': accuracy_score(y_test, y_pred_weighted), 'Minority Avg F1': np.mean([report_weighted[c]['f1-score'] for c in report_weighted if c not in ['accuracy','macro avg','weighted avg']])})

        imbalance_df = pd.DataFrame(imbalance_results)
        st.subheader("Imbalance Treatment Results")
        st.dataframe(imbalance_df)

        st.subheader("Detailed Classification Report for Best Method")
        best_method = imbalance_df.sort_values('Minority Avg F1', ascending=False).iloc[0]['Method']
        st.write(f"Best method by minority avg F1: {best_method}")
        best_report = {'SMOTE': report_smote, 'Random Oversampling': report_ros, 'Random Undersampling': report_rus, 'Class Weighting': report_weighted}[best_method]
        st.text(classification_report(y_test, {'SMOTE': y_pred_smote, 'Random Oversampling': y_pred_ros, 'Random Undersampling': y_pred_rus, 'Class Weighting': y_pred_weighted}[best_method], zero_division=0))

else:
    st.warning("Target column 'Class' not found; cannot perform stratified split.")

# Build a Simple Classifier App
st.header("Bean Type Prediction Application")
st.write("Enter bean physical measurements below and click 'Predict' to see the predicted bean type.")

if target_col and 'best_model' in locals():
    with st.form(key='bean_predict_form'):
        st.subheader("Input Bean Measurements")
        input_values = {}
        for col in numerical_cols:
            input_values[col] = st.number_input(f"{col}", min_value=float(data[col].min()), max_value=float(data[col].max()), value=float(data[col].median()))

        if categorical_cols:
            input_cats = {}
            for col in categorical_cols:
                options = sorted(data[col].dropna().unique().tolist())
                input_cats[col] = st.selectbox(f"{col}", options)
        else:
            input_cats = {}

        predict_button = st.form_submit_button("Predict")

    if predict_button:
        # Construct input DataFrame
        X_new = pd.DataFrame([input_values])

        # Apply categorical encoding if needed
        if categorical_cols:
            cat_df = pd.DataFrame([input_cats])
            cat_ohe = pd.DataFrame(ohe.transform(cat_df), columns=ohe.get_feature_names_out(categorical_cols))
            X_new = pd.concat([X_new, cat_ohe], axis=1)

        # Ensure missing one-hot columns are added
        for c in data_scaled.drop(columns=[target_col]).columns:
            if c not in X_new.columns:
                X_new[c] = 0

        X_new = X_new[data_scaled.drop(columns=[target_col]).columns]

        # Apply skew transformation for skewed numerical values
        if skewed_cols:
            for col in skewed_cols:
                val = X_new.loc[0, col]
                if val <= 0:
                    val = val - data_encoded[col].min() + 1
                X_new.loc[0, col] = np.log1p(val)

        # Scale numeric features
        X_new[numerical_cols] = scaler.transform(X_new[numerical_cols])

        pred_class = best_model.predict(X_new)[0]
        if target_col and data[target_col].dtype == 'object':
            pred_label = le.inverse_transform([int(pred_class)])[0]
        else:
            pred_label = pred_class

        st.success(f"Predicted Bean Type: {pred_label}")

else:
    st.warning("Best model or target column not available for prediction.")



