import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv(r"C:\Users\Dell\Downloads\insurance_claims.csv", na_values=['?'])

print("Missing values in each column:")
print(df.isnull().sum())

print("\nSummary statistics for numeric columns:")
print(df.describe())

print("\nData types of columns:")
print(df.dtypes)

sns.histplot(df['claim_amount'], kde=True)
plt.title('Distribution of Claim Amounts')
plt.show() 
plt.clf() 
df.columns = df.columns.str.strip()

if 'fraud_reported' in df.columns:
    print("\nColumn 'fraud_reported' exists!")

    df = df.dropna(subset=['fraud_reported'])

    categorical_columns = ['collision_type', 'authorities_contacted', 'property_damage', 'police_report_available']
    for col in categorical_columns:
        df[col].fillna('Unknown', inplace=True)

    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('fraud_reported', axis=1)
    y = df['fraud_reported']
else:
    print("\nColumn 'fraud_reported' not found!")
    exit()  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy (Initial Model):", accuracy_score(y_test, y_pred))
print("\nClassification Report (Initial Model):\n", classification_report(y_test, y_pred))

feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Features:")
print(feature_importances)

plt.figure()  
feature_importances.head(10).plot(kind='bar', title='Top 10 Important Features')
plt.show() 
plt.clf()  
param_grid = {
    'n_estimators': [100, 150], 
    'max_depth': [None, 10],  
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("\nBest Parameters from Grid Search:", best_params)

model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train, y_train)

y_pred_optimized = model.predict(X_test)

print("\nAccuracy (Optimized Model):", accuracy_score(y_test, y_pred_optimized))
print("\nClassification Report (Optimized Model):\n", classification_report(y_test, y_pred_optimized))

joblib.dump(model, 'fraud_detection_model_optimized.joblib')
print("\nOptimized model saved as 'fraud_detection_model_optimized.joblib'")

