import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import io
import zipfile

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    with z.open("student-mat.csv") as f:
        df = pd.read_csv(f, sep=";") 

# Feature Engineering
df['average_grade'] = (df['G1'] + df['G2'] + df['G3']) / 3
df['performance'] = df['average_grade'].apply(lambda x: 1 if x >= 10 else 0)  # 1 = Pass, 0 = Fail
df = df.drop(['G1', 'G2', 'G3', 'average_grade'], axis=1)

# Convert categorical columns to numeric
df = pd.get_dummies(df)

# Split data
X = df.drop('performance', axis=1)
y = df['performance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction & Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance Plot
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()
