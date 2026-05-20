"""
Iris Classification Project
Author: Belkin Nikita
"""

# ---------------- TRAINING  ----------------
# I decided to import only the basic scikit-learn tools that are necessary for building and evaluating the model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

# I wanted to inspect how the dataset looks, so I printed a small part of it
print("Features (first 5 samples):\n", X[:5])
print("Labels (first 5 samples):\n", y[:5])

# I decided to split the data into training (80%) and testing (20%) sets
# because I need to evaluate how well the model generalizes to unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# I chose RandomForestClassifier because I wanted a stable and reliable model
# that works well on tabular data without requiring complex preprocessing
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
print("Model accuracy:", score)

# ---------------- USER INPUT ----------------
# I decided to allow user input so that they can test the model manually
# and verify that it works in a real interaction scenario
# For test you can use: Setosa(5.1, 3.5, 1.4, 0.2), Versicolor(6.0, 2.7, 4.1, 1.3), Virginica(6.5, 3.0, 5.2, 2.0)
while True:
    print("\nEnter 4 values (sepal_length, sepal_width, petal_length, petal_width):")

    features = [
        float(input("Sepal length: ")),
        float(input("Sepal width: ")),
        float(input("Petal length: ")),
        float(input("Petal width: "))
    ]

    user_prediction = model.predict([features])[0]
    print("Prediction:", iris.target_names[user_prediction])

    again = input("\nDo you want to try again? (y/n): ").lower()

    if again != "y":
        print("Exiting program.")
        break