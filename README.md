# Heart Attack Prediction Using MLP

**Name:** Niralya J
**Register No.:** 212224230188
**Experiment No.:** 6
**Date:** 28/08/2026

## Aim

To construct a Multi-Layer Perceptron (MLP) to predict heart attack using Python.

## Algorithm

**Step 1:** Import the required libraries: NumPy, Pandas, `MLPClassifier`, `train_test_split`, `StandardScaler`, `accuracy_score`, and Matplotlib.

**Step 2:** Load the heart disease dataset from a file using `pd.read_csv()`.

**Step 3:** Separate the features and labels from the dataset using `data.iloc` values for features (`X`) and `data.iloc[:, -1].values` for labels (`y`).

**Step 4:** Split the dataset into training and testing sets using `train_test_split()`.

**Step 5:** Normalize the feature data using `StandardScaler()` to scale the features to have zero mean and unit variance.

**Step 6:** Create an `MLPClassifier` model with the required architecture and hyperparameters such as `hidden_layer_sizes`, `max_iter`, and `random_state`.

**Step 7:** Train the MLP model on the training data using `mlp.fit(X_train, y_train)`. The model adjusts its weights and biases iteratively to minimize the training loss.

**Step 8:** Make predictions on the testing set using `mlp.predict(X_test)`.

**Step 9:** Evaluate the model's accuracy by comparing the predicted labels (`y_pred`) with the actual labels (`y_test`) using `accuracy_score()`.

**Step 10:** Print the accuracy of the model.

**Step 11:** Plot the error convergence during training using `plt.plot()` and `plt.show()`.

## Program

```python
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv('heart.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    max_iter=1000,
    random_state=42
)

training_loss = mlp.fit(X_train, y_train).loss_curve_

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_rep)
```

## Output

### Training Loss Convergence

![MLP Training Loss Convergence](https://github.com/user-attachments/assets/f39c5938-b665-4e6b-ae2d-5915d6eb7822)

### Model Evaluation

![Model Evaluation](https://github.com/user-attachments/assets/63ec7574-acc0-408a-8fbb-fb668cd44bc6)

## Result

Thus, an Artificial Neural Network using a Multi-Layer Perceptron (MLP) was successfully constructed and trained to predict heart attack using Python.
