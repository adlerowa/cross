from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Example: Data structure
# data[participant][run] with shape (num_participants, num_runs, num_features)
# labels[participant][run] with shape (num_participants, num_runs)

# Sample data (replace with your actual preprocessed data)
num_participants = 10
num_runs = 6
num_features = 100

# Generate synthetic data for illustration purposes
data = np.random.rand(num_participants, num_runs, num_features)
labels = np.random.randint(0, 2, (num_participants, num_runs))

# Flatten runs for each participant
flattened_data = np.array([run for participant in data for run in participant])
flattened_labels = np.array([label for participant in labels for label in participant])

# Create a group label for each participant
groups = np.array([i for i in range(num_participants) for _ in range(num_runs)])

                                                                                                                                                                                                                                                                            


