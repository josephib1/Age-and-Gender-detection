import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.models import load_model
# Load the dataset
path = r'PATH TO YOUR DATASET'
files = os.listdir(path)
images = []
ages = []
genders = []

for file in files:
    image = cv2.imread(os.path.join(path, file))
    image = cv2.resize(image, (224, 224))
    images.append(image)

    age, gender = file.split("_")[:2]
    ages.append(age)
    genders.append(gender)

images = np.array(images)
ages = np.array(ages).astype(int)
genders = np.array(genders).astype(int)

# Prepare the model
base_model = MobileNet(weights='imagenet', include_top=False,input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Define two output layers
age_output = Dense(1, name='age_output')(x)  # Assuming age is a regression problem
gender_output = Dense(2, activation='softmax', name='gender_output')(x)  # Assuming gender is a classification problem

# Define the model with the two outputs
model = Model(inputs=base_model.input, outputs=[age_output, gender_output])

# Compile the model with two loss functions
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss={'age_output': 'mse', 'gender_output': 'sparse_categorical_crossentropy'}, 
              metrics={'age_output': 'mae', 'gender_output': 'accuracy'})

# Train the model with a dictionary of outputs
history = model.fit(images, {'age_output': ages, 'gender_output': genders}, epochs=5, validation_split=0.2)

# If you want to save the entire model (architecture, optimizer state, etc.), you can use:
model.save('MobileNetV2-1.h5')

# If you saved the entire model, you can load it with:
model = keras.models.load_model('MobileNetV2-1')

# Then continue training
#history = model.fit(images, {'age_output': ages, 'gender_output': genders}, epochs=5, validation_split=0.2)


# Access training history
training_loss_age = history.history['age_output_loss']
training_loss_gender = history.history['gender_output_loss']
training_accuracy_gender = history.history['gender_output_accuracy']

validation_loss_age = history.history['val_age_output_loss']
validation_loss_gender = history.history['val_gender_output_loss']
validation_accuracy_gender = history.history['val_gender_output_accuracy']

# Plot training and validation accuracy for gender output
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(training_accuracy_gender, label='Training Accuracy')
plt.plot(validation_accuracy_gender, label='Validation Accuracy')
plt.title('Gender Output Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss for gender output
plt.subplot(1, 2, 2)
plt.plot(training_loss_gender, label='Training Loss')
plt.plot(validation_loss_gender, label='Validation Loss')
plt.title('Gender Output Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Plot training and validation loss for age output
plt.figure(figsize=(6, 4))
plt.plot(training_loss_age, label='Training Loss')
plt.plot(validation_loss_age, label='Validation Loss')
plt.title('Age Output Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.summary()
# Load the new image for testing
image_path = r'PUT YOUR TEST IMAGES FOLDER PATH HERE'
image_file = os.listdir(image_path)
for file in image_file:
    new_image = cv2.imread(os.path.join(image_path, file))
    new_image = cv2.resize(new_image, (224, 224))  # Resize the image to match the input size of the model
    new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension
    #new_image = preprocess_input(new_image)  # Apply preprocessing specific to MobileNet

    # Make predictions
    predictions = model.predict(new_image)

    # Extract age and gender predictions
    age_prediction = round(predictions[0][0].item())  # Assuming age is the first output
    gender_prediction = np.argmax(predictions[1][0])  # Assuming gender is the second output

    # Display the image
    plt.imshow(cv2.cvtColor(new_image[0], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    print(f"Predicted Age: {round(age_prediction)}")
    print(f"Predicted Gender: {'Female' if gender_prediction == 1 else 'Male'}")