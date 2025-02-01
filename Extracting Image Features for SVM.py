import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
data = []
labels = []
dataset_path = 'herb_images/'  # Directory of images
classes = os.listdir(dataset_path)  # Herb categories

for label, herb in enumerate(classes):
    herb_path = os.path.join(dataset_path, herb)
    for img_name in os.listdir(herb_path):
        img_path = os.path.join(herb_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        img = cv2.resize(img, (128, 128))  
        features = img.flatten()  # Convert to 1D vector
        data.append(features)
        labels.append(label)

# Convert to numpy array
X = np.array(data)
y = np.array(labels)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM Model
svm = SVC(kernel='rbf', C=10, gamma='scale')  # Using RBF kernel
svm.fit(X_train, y_train)

# Evaluate Model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")

# Save Model
joblib.dump(svm, 'svm_herb_model.pkl')
