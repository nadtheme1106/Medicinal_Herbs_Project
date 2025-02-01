import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define 3D CNN Model
model = Sequential([
    Conv3D(32, kernel_size=(3,3,3), activation='relu', input_shape=(128, 128, 3, 1)),  
    MaxPooling3D(pool_size=(2,2,2)),  
    Conv3D(64, kernel_size=(3,3,3), activation='relu'),
    MaxPooling3D(pool_size=(2,2,2)),  
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Assuming 10 herb categories
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and Train Model (example using ImageDataGenerator)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory('herb_images/', target_size=(128,128), batch_size=32, class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory('herb_images/', target_size=(128,128), batch_size=32, class_mode='categorical', subset='validation')

# Train Model
model.fit(train_generator, validation_data=val_generator, epochs=20)

# Save Model
model.save('3d_cnn_herb_model.h5')
