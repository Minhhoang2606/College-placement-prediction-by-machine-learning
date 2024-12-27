'''
Engineering placement prediction by machine learning
Author: Henry Ha
'''
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

#TODO EDA
data = pd.read_csv("collegePlace.csv")
print(data.info())  # Display data types and non-null counts
print(data.head())  # Display the first few rows
print(data.describe())  # Display summary statistics

# Check for missing values
print(data.isnull().sum())

# Visualizing Feature Distributions
data.hist(bins=15, figsize=(15, 10), color='teal', edgecolor='black')
plt.suptitle("Feature Distributions", y=1.02)
plt.show()

# Detecting outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[["Age", "Internships", "CGPA"]])
plt.title("Outlier Detection in Numerical Features")
plt.show()

# Categorical Feature Analysis

# Gender vs Placement Status
plt.figure(figsize=(8, 6))
ax = sns.countplot(x="Gender", hue="PlacedOrNot", data=data)
plt.title("Placement Status by Gender")

# Adding values on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.show()

# Stream vs Placement Status
plt.figure(figsize=(15, 6))
ax = sns.countplot(x="Stream", hue="PlacedOrNot", data=data)
plt.title("Placement Status by Stream")
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

# Adding values on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()

# Heatmap for Correlations
plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(numeric_only=True), annot=True, linewidths=3, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Effect of Internships on Placement
internships_effect = pd.crosstab(index=data['PlacedOrNot'],
                                 columns=data['Internships'],
                                 normalize='columns') * 100
internships_effect.plot(kind='bar', figsize=(10, 6))
plt.title("Effect of Internships on Placement")
plt.xlabel("Placement Status")
plt.ylabel("Percentage")
plt.show()

# Effect of CGPA on Placement
sns.boxplot(x="PlacedOrNot", y="CGPA", data=data)
plt.title("Effect of CGPA on Placement")
plt.xlabel("Placement Status")
plt.ylabel("CGPA")
plt.show()

# Feature pair analysis
sns.pairplot(data, hue="PlacedOrNot", vars=["CGPA", "Internships", "Age"])
plt.suptitle("Pairwise Relationships Between Features and Placement", y=1.02)
plt.show()

#TODO Data Preprocessing

# One-hot encoding for 'Stream'
stream_encoded = pd.get_dummies(data['Stream'], drop_first=True)

# Label encoding for 'Gender'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])  # Male = 1, Female = 0

# Combine all features
X = pd.concat([data.drop(columns=['PlacedOrNot', 'Stream']), stream_encoded], axis=1)
y = data['PlacedOrNot']

# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#TODO Model Training and Evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
}

# Create a figure for subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axes = axes.flatten()

# Train and evaluate models
for idx, (name, model) in enumerate(models.items()):
    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

    # Display confusion matrix in a subplot
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[idx])
    axes[idx].set_title(f"Confusion Matrix for {name}")

# Adjust layout
plt.tight_layout()
plt.show()

#TODO Model optimization

# Gradient Boosting Optimization

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Initialize the model
gb_model = GradientBoostingClassifier(random_state=42)

# Perform GridSearchCV
grid_gb = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, scoring='accuracy', cv=5, n_jobs=-1)
grid_gb.fit(X_train, y_train)

# Evaluate the best model on the test set
gb_best = grid_gb.best_estimator_  # Retrieve the best model
y_pred_gb = gb_best.predict(X_test)  # Predict on the test set
accuracy_gb_test = accuracy_score(y_test, y_pred_gb)

# Print the results
print(f"Best Parameters for Gradient Boosting: {grid_gb.best_params_}")
print(f"Test Accuracy for Gradient Boosting: {accuracy_gb_test * 100:.2f}%")

# Random Forest Optimization

from sklearn.ensemble import RandomForestClassifier

# Define the parameter grid
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize the model
rf_model = RandomForestClassifier(random_state=42)

# Perform GridSearchCV
grid_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, scoring='accuracy', cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)

# Evaluate the best model on the test set
rf_best = grid_rf.best_estimator_  # Retrieve the best model
y_pred_rf = rf_best.predict(X_test)  # Predict on the test set
accuracy_rf_test = accuracy_score(y_test, y_pred_rf)

# Print the results
print(f"Best Parameters for Random Forest: {grid_rf.best_params_}")
print(f"Test Accuracy for Random Forest: {accuracy_rf_test * 100:.2f}%")

import pickle

# Save the Gradient Boosting model (or use Random Forest if preferred)
with open('placement_model.pkl', 'wb') as file:
    pickle.dump(gb_best, file)  # Replace `gb_best` with `rf_best` if using Random Forest
print("Model saved as 'placement_model.pkl'")
