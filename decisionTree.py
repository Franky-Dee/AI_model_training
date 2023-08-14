import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('wine.data', header=None)
feature_names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
data.columns = ['class'] + feature_names

# Display the first few rows of the dataset
print(data.head())

# Check for any missing values
print(data.isnull().sum())

# Separate features (X) and target (y)
X = data.drop('class', axis=1)
y = data['class']

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
    
# Calculate ROC-AUC score
y_prob = model.predict_proba(X_test)
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')  # 'ovr' for multi-class AUC
print(f"ROC-AUC: {auc:.2f}")

# Generate and plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=2)  # Calculate ROC curve for class 2
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


