import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data: text length, exclamation marks vs spam
data = {
    "text_length": [120, 20, 200, 15, 180, 25, 150, 18, 210, 30],
    "exclamation_count": [0, 5, 1, 6, 0, 4, 1, 7, 0, 5],
    "is_spam": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
print(df)

X = df[["text_length", "exclamation_count"]]
y = df["is_spam"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# No scaling needed for Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nPredictions:", predictions)
print("Actual:", y_test.values)
print("Probabilities:\n", probabilities)
print("Accuracy:", accuracy)

# Save results
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions,
    "Probability_of_Spam": probabilities[:, 1]
})
results_df.to_csv("results.csv", index=False)
print("\nSaved results.csv")