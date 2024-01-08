import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from datetime import datetime

# Load the sample dataset
data = load_iris()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Define a dictionary of classification models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Specify Public URL of EC2 instance where the MLflow tracking server is running
TRACKING_SERVER_HOST = "ec2-3-26-13-3.ap-southeast-2.compute.amazonaws.com"

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000") 
print(f"Tracking Server URI: '{mlflow.get_tracking_uri()}'")

#specify name of experiment (will be created if it does not exist)
mlflow.set_experiment("my-test-exp0")
# Train and log each classification model
for model_name, model in models.items():
    name = f'{model_name}:{datetime.now()}'

    with mlflow.start_run(run_name=name):

        # Set the tag for name of user who ran the experiment
        mlflow.set_tag("User", "Ugas")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model on the test set
        accuracy = model.score(X_test, y_test)
        
        # Log model metrics
        mlflow.log_metric("Accuracy", accuracy)

        print(f"Artifacts URI: '{mlflow.get_artifact_uri()}'")

        # Save the MLflow run ID
        run_id = mlflow.active_run().info.run_id
        print("MLflow Run ID:", run_id)