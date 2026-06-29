import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv("Iris.csv") # Make sure this matches your filename exactly
print("--- First 5 Rows ---")
print(df.head())

# 2. Visualize the data (This is what employers want to see!)
sns.pairplot(df.drop('Id', axis=1), hue='Species')
plt.show()

# 3. Separate Features (X) and Target (y)
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# 4. Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the AI (K-Nearest Neighbors Classifier)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 6. Predict and Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n🎯 --- IRIS CLASSIFICATION RESULTS --- 🎯")
print(f"Model Accuracy: {round(accuracy * 100, 2)}%")
print("\n--- Detailed Performance Report ---")
print(classification_report(y_test, predictions))