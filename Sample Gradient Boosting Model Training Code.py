from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
# Load dataset
df = pd.read_csv(&#39;herb_dataset.csv&#39;)
X = df.drop(columns=[&#39;herb_name&#39;])
y = df[&#39;herb_name&#39;]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)
# Train model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
max_depth=3)
model.fit(X_train, y_train)
# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f&#39;Model Accuracy: {accuracy * 100:.2f}%&#39;)