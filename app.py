import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the saved models
rf_model = joblib.load("models/random_forest_model.joblib")
gb_model = joblib.load("models/gradient_boosting_model.joblib")
svm_model = joblib.load("models/svm_model.joblib")
mlp_model = joblib.load("models/neural_network_model.joblib")
lr_model = joblib.load("models/logistic_regression_model.joblib")
nb_model = joblib.load("models/naive_bayes_model.joblib")



def preprocess_choices(data,choices):

    # Convert categorical features to numerical using label encoding
    categorical_features =  ['age', 'workclass', 'education', 'occupation', 'race', 'sex', 'hours-per-week', 'native-country']
    cat = ['workclass', 'education', 'occupation', 'race', 'sex', 'native-country']
    data = data[categorical_features]
    choices = choices[categorical_features]
    label_encoder = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.strip()
            label_encoder.fit(data[col])
            choices[col] = label_encoder.transform(choices[col])
    
    # choices.loc[:cat] = label_encoder.transform(choices[cat].values.ravel())



    # Scale numerical features using StandardScaler
    scaler = StandardScaler()
    numerical_features = ['age', 'hours-per-week']
    scaler.fit(data.loc[:, numerical_features].values)
    choices.loc[:, numerical_features] = scaler.transform(choices.loc[:, numerical_features].values)

    return choices

def main():
    # Read the dataset from the .data file
    data = pd.read_csv(r'Dataset\adult.data', delimiter=',', header=None)

    # Remove the last column
    data = data.iloc[:, :-1]
    # Assign the column names
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country']
    
    data.columns = column_names

    # Set the title of the app
    st.title("Income Prediction App")

    # Load the column names from the .names file
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country']

    # Add input fields for other columns as per your dataset
    st.header("📋 Input Parameters")

    age = st.number_input("Age", value=30, min_value=1, max_value=100)
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
    fnlwgt = st.number_input("Fnlwgt", value=100, min_value=1)
    education = st.selectbox("Education", ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
    education_num = st.number_input("Education Num", value=10, min_value=1)
    marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
    occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
    relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
    race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
    sex = st.selectbox("Sex", ["Female", "Male"])
    capital_gain = st.number_input("Capital Gain", value=0, min_value=0)
    capital_loss = st.number_input("Capital Loss", value=0, min_value=0)
    hours_per_week = st.number_input("Hours per Week", value=40, min_value=1, max_value=100)
    native_country = st.selectbox("Native Country", ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])

    # Create a dictionary with the input values
    input_data = pd.DataFrame({
        "age": [age],
        "workclass": [workclass],
        "fnlwgt": [fnlwgt],
        "education": [education],
        "education-num": [education_num],
        "marital-status": [marital_status],
        "occupation": [occupation],
        "relationship": [relationship],
        "race": [race],
        "sex": [sex],
        "capital-gain": [capital_gain],
        "capital-loss": [capital_loss],
        "hours-per-week": [hours_per_week],
        "native-country": [native_country]
    })

    preprocessed_data = preprocess_choices(data, input_data)

    # Add model selection dropdown
    st.header("🧠 Model Selection")
    model_name = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "SVM", "Neural Network", "Logistic Regression", "Naive Bayes"])

    # Make predictions based on the selected model
    if model_name == "Random Forest":
        prediction = rf_model.predict(preprocessed_data)
    elif model_name == "Gradient Boosting":
        prediction = gb_model.predict(preprocessed_data)
    elif model_name == "SVM":
        prediction = svm_model.predict(preprocessed_data)
    elif model_name == "Neural Network":
        prediction = mlp_model.predict(preprocessed_data)
    elif model_name == "Logistic Regression":
        prediction = lr_model.predict(preprocessed_data)
    elif model_name == "Naive Bayes":
        prediction = nb_model.predict(preprocessed_data)

    # Display the prediction result
    st.header("🔮 Prediction Result")
    st.write("Income Category:", prediction)

# Run the app
if __name__ == "__main__":
    main()
