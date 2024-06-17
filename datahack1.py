import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Step 1: Load the data
train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')
test_features = pd.read_csv('test_set_features.csv')
submission_format = pd.read_csv('submission_format.csv')

# Step 2: Preprocess and Prepare Data (similar to previous steps)
# Assuming preprocessing steps are similar to what was previously discussed

# Step 3: Train your model
# Separate features and labels
X = train_features.drop(['respondent_id'], axis=1)  # Assuming respondent_id is not a feature
y = train_labels[['xyz_vaccine', 'seasonal_vaccine']]

# Initialize the base classifier
base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Wrap the classifier with MultiOutputClassifier
model = MultiOutputClassifier(base_classifier, n_jobs=-1)

# Fit the model
model.fit(X, y)

# Step 4: Predict on test data
X_test = test_features.drop(['respondent_id'], axis=1)  # Assuming respondent_id is not a feature
test_probs = model.predict_proba(X_test)

# Extract the probabilities for each target
test_probs_xyz = test_probs[0][:, 1]
test_probs_seasonal = test_probs[1][:, 1]

# Step 5: Create submission DataFrame
submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': test_probs_xyz,
    'seasonal_vaccine': test_probs_seasonal
})

# Step 6: Save submission DataFrame to CSV
submission.to_csv('submission_format.csv', index=False)

print("Submission file updated successfully.")
