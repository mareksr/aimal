import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

AIMAL_PATH = os.getenv('AIMAL_PATH', '/opt/aimal')
RF_MODEL_FILE = os.path.join(AIMAL_PATH, 'rf_classifier_model.pkl')
VECTORIZER_FILE = os.path.join(AIMAL_PATH, 'vectorizer.pkl')
LABELBINARIZER_FILE = os.path.join(AIMAL_PATH, 'labelbinarizer.pkl')

def load_files_from_list(file_list):
    data = []
    with open(file_list, 'r') as f:
        files = f.readlines()
        for file_path in files:
            file_path = file_path.strip()
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    data.append(file.read())
    return data

def load_files_from_directory(directory):
    data = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    data.append(f.read())
    return data

def train_and_evaluate_classifiers(good_file, malware_directory):
    good_data = load_files_from_list(good_file)
    bad_data = load_files_from_directory(malware_directory)

    good_count = len(good_data)
    bad_count = len(bad_data)

    print(f"Number of good files: {good_count}")
    print(f"Number of bad files: {bad_count}")

    labels = [0] * good_count + [1] * bad_count  # 0 = good, 1 = bad
    data = good_data + bad_data

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data).toarray()

    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train.ravel())
    rf_predictions = rf_classifier.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    if not os.path.exists(AIMAL_PATH):
        os.makedirs(AIMAL_PATH)
    joblib.dump(rf_classifier, RF_MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(label_binarizer, LABELBINARIZER_FILE)

    print("Models trained and saved successfully.")
    print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

    print(f"Good/Bad file ratio: {good_count}/{bad_count} ({good_count / (good_count + bad_count):.2f}/{bad_count / (good_count + bad_count):.2f})")

def classify_file(file_path):
    rf_classifier = joblib.load(RF_MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    X = vectorizer.transform([content]).toarray()
    prediction = rf_classifier.predict(X)

    return prediction[0]

def load_model():
    rf_classifier = joblib.load(RF_MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    return rf_classifier, vectorizer
