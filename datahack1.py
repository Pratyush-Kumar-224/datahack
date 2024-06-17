import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load the data
train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')
test_features = pd.read_csv('test_set_features.csv')
submission_format = pd.read_csv('submission_format.csv')

# Separate numeric and non-numeric columns
numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_features.select_dtypes(exclude=[np.number]).columns.tolist()

# Fill missing values in numeric columns with the median
train_features[numeric_cols] = train_features[numeric_cols].fillna(train_features[numeric_cols].median())
test_features[numeric_cols] = test_features[numeric_cols].fillna(test_features[numeric_cols].median())

# Fill missing values in categorical columns with the mode
train_features[categorical_cols] = train_features[categorical_cols].fillna(train_features[categorical_cols].mode().iloc[0])
test_features[categorical_cols] = test_features[categorical_cols].fillna(test_features[categorical_cols].mode().iloc[0])

# One-hot encode categorical variables
train_features = pd.get_dummies(train_features)
test_features = pd.get_dummies(test_features)

# Ensure the test set has the same columns as the training set
missing_cols = set(train_features.columns) - set(test_features.columns)
for col in missing_cols:
    test_features[col] = 0
test_features = test_features[train_features.columns]

# Extract features and labels
X = train_features.drop(['respondent_id'], axis=1)
y = train_labels[['xyz_vaccine', 'seasonal_vaccine']]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the base classifier
base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Wrap the classifier with MultiOutputClassifier
model = MultiOutputClassifier(base_classifier, n_jobs=-1)

# Fit the model
model.fit(X_train, y_train)

# Predict probabilities on the validation set
val_probs = model.predict_proba(X_val)

# Extract the probabilities for each target
val_probs_xyz = val_probs[0][:, 1]
val_probs_seasonal = val_probs[1][:, 1]

# Calculate ROC AUC for each label
roc_auc_xyz = roc_auc_score(y_val['xyz_vaccine'], val_probs_xyz)
roc_auc_seasonal = roc_auc_score(y_val['seasonal_vaccine'], val_probs_seasonal)

mean_roc_auc = (roc_auc_xyz + roc_auc_seasonal) / 2
print(f"ROC AUC for XYZ Vaccine: {roc_auc_xyz}")
print(f"ROC AUC for Seasonal Vaccine: {roc_auc_seasonal}")
print(f"Mean ROC AUC: {mean_roc_auc}")

# Prepare the test data
X_test = test_features.drop(['respondent_id'], axis=1)

# Predict probabilities on the test set
test_probs = model.predict_proba(X_test)

# Extract the probabilities for each target
test_probs_xyz = test_probs[0][:, 1]
test_probs_seasonal = test_probs[1][:, 1]

# Create the submission DataFrame
submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': test_probs_xyz,
    'seasonal_vaccine': test_probs_seasonal
})

# Save to CSV
submission.to_csv('submission_format.csv', index=False)

print(submission.head())
