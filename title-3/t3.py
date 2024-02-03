
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your dataset from a CSV file
csv_path = "your_dataset.csv"
data = pd.read_csv(csv_path)

# One-hot encode the 'Gender' column
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

# Extract features and labels
X = data[['Age', 'Gender_Male', 'LiverFunctionTest1', 'LiverFunctionTest2', 'TumorSize', 'AFPLevel']]
y = data['Diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize your segmentation model (replace with your actual model)
model = RandomForestClassifier()

# Train the model for 20 iterations
for iteration in range(20):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate accuracy for each iteration
    accuracy = accuracy_score(y_test, predictions)
    print(f"Iteration {iteration + 1}: Accuracy = {accuracy}")

# Calculate the mean accuracy over 20 iterations
mean_accuracy = np.mean([accuracy_score(y_test, model.predict(X_test)) for _ in range(20)])
print(f"Mean Accuracy over 20 iterations: {mean_accuracy}")
