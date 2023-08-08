import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Load the dataset
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
data = pd.read_csv('iris.data', names=feature_names + ['species'])

# Display the first few rows of the dataset
print(data.head())

# Check for any missing values
print(data.isnull().sum())

# Separate features (X) and target (y)
X = data.drop('species', axis=1)
y = data['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Calculate precision, recall, and F1-score for each class
precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=model.classes_)

for i, class_name in enumerate(model.classes_):
    print(f"Class: {class_name}")
    print(f"Precision: {precision_per_class[i]:.2f}")
    print(f"Recall: {recall_per_class[i]:.2f}")
    print(f"F1-score: {f1_per_class[i]:.2f}")
    print()

# ... (Plotting ROC curve)
