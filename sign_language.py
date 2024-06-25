import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
 
# Define data directories
train_data_dir = 'training_set'
test_data_dir = 'test_set'
 
# Define image dimensions and other parameters
img_width, img_height = 64, 64
batch_size = 32
epochs = 30
num_classes = 17 # Adjusted to match the actual number of classes
 
# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
 
test_datagen = ImageDataGenerator(rescale=1./255)
 
# Generate training and test sets
train_set = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'  
)
 
test_set = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'  
)
 
# Build CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # Adjusted to match the actual number of classes
 
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(
    train_set,
    steps_per_epoch=train_set.samples // batch_size,
    epochs=epochs,
    validation_data=test_set,
    validation_steps=test_set.samples // batch_size
)

from keras.preprocessing import image
test_image=image.load_img('prediction/H_test.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
# Perform the prediction
result = model.predict(test_image)
 
# Get the class indices mapping
class_indices = train_set.class_indices
 
# Invert the dictionary to map indices to class labels
inverse_class_indices = {v: k for k, v in class_indices.items()}
 
# Get the predicted class index (assuming it's a single-class classification)
predicted_class_index = np.argmax(result)
 
# Get the corresponding class label
predicted_class_label = inverse_class_indices[predicted_class_index]
 
# Display the predicted class label (alphabet)
print("Predicted alphabet:", predicted_class_label)

from keras.preprocessing import image
test_image=image.load_img('prediction/A.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
# Perform the prediction
result = model.predict(test_image)
 
# Get the class indices mapping
class_indices = train_set.class_indices
 
# Invert the dictionary to map indices to class labels
inverse_class_indices = {v: k for k, v in class_indices.items()}
 
# Get the predicted class index (assuming it's a single-class classification)
predicted_class_index = np.argmax(result)
 
# Get the corresponding class label
predicted_class_label = inverse_class_indices[predicted_class_index]
 
# Display the predicted class label (alphabet)
print("Predicted alphabet:", predicted_class_label)

from keras.preprocessing import image
test_image=image.load_img('prediction/P.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
# Perform the prediction
result = model.predict(test_image)
 
# Get the class indices mapping
class_indices = train_set.class_indices
 
# Invert the dictionary to map indices to class labels
inverse_class_indices = {v: k for k, v in class_indices.items()}
 
# Get the predicted class index (assuming it's a single-class classification)
predicted_class_index = np.argmax(result)
 
# Get the corresponding class label
predicted_class_label = inverse_class_indices[predicted_class_index]
 
# Display the predicted class label (alphabet)
print("Predicted alphabet:", predicted_class_label)
