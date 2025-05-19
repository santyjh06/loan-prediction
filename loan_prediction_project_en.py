import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = {
    'income': [3000, 4500, 1500, 7000, 1000, 2000, 5000, 4000],
    'debt': [0, 500, 2000, 0, 3000, 1000, 400, 800],
    'loan_approved': ['yes', 'yes', 'no', 'yes', 'no', 'maybe', 'yes', 'maybe']
}

# Create DataFrame
df = pd.DataFrame(data)

# Features (X) and target (y)
X = df[['income', 'debt']]
y = df['loan_approved']

# Create and train model
model = DecisionTreeClassifier()
model.fit(X, y)


# Ask user for input
print("Enter the new customer's data:")
income = float(input("Monthly income: "))
debt = float(input("Current debt: "))

new_customer = [[income, debt]]
prediction = model.predict(new_customer)

print("Loan decision for the new customer:", prediction[0])