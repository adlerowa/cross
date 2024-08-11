import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Perform cross-validation
accuracies = []

for train_index, test_index in gkf.split(flattened_data, flattened_labels, groups=groups):
    X_train, X_test = flattened_data[train_index], flattened_data[test_index]
    y_train, y_test = flattened_labels[train_index], flattened_labels[test_index]
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)
    
    print(f'Fold accuracy: {accuracy:.4f}')

# Average accuracy across folds
average_accuracy = np.mean(accuracies)
print(f'Average accuracy: {average_accuracy:.4f}')
