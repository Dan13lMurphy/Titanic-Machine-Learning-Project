# titanic_model.py

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ----------------------
# 1. Load Data
# ----------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ----------------------
# 2. Data Cleaning
# ----------------------

# Fill missing Age
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

# Fill Embarked
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

# Fill Fare
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# Convert Sex to numeric
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked
train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Embarked'])

# Ensure both datasets have same columns
for col in ['Embarked_C', 'Embarked_Q', 'Embarked_S']:
    if col not in test:
        test[col] = 0

# ----------------------
# 3. Feature Selection
# ----------------------
features = [
    'Pclass', 'Sex', 'Age',
    'SibSp', 'Parch', 'Fare',
    'Embarked_C', 'Embarked_Q', 'Embarked_S'
]

X = train[features]
y = train['Survived']
X_test = test[features]

# ----------------------
# 4. Train Model
# ----------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ----------------------
# 5. Evaluate Model
# ----------------------
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation accuracy:", scores.mean())

# ----------------------
# 6. Make Predictions
# ----------------------
predictions = model.predict(X_test)

# ----------------------
# 7. Save Submission
# ----------------------
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)

print("✅ submission.csv file created!")
