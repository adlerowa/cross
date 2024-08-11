import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Defining cross-validation strategy
gkf = GroupKFold(n_splits=5)


# Define the model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Include if additional scaling/normalization is needed
    ('classifier', LinearRegression())
])

