# Age and Gender Detection using MobileNetV2

## Introduction
This repository discusses the implementation of an age and gender detection system using the MobileNetV2 architecture. The code utilizes the TensorFlow and Keras libraries to build, train, and evaluate the model. The dataset used for training is the [UTKFace dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new), which contains facial images annotated with age and gender information.

## Model Architecture
The model architecture is based on MobileNetV2, a lightweight convolutional neural network (CNN) designed for mobile and edge devices. The pre-trained MobileNetV2 is used as a base model, with two additional dense layers added for age and gender predictions. Global Average Pooling (GAP) is applied to reduce spatial dimensions before the final output layers.

- **Age Output Layer:** Treated as a regression problem, using a single-node output layer with mean squared error (MSE) loss.
- **Gender Output Layer:** Treated as a classification problem with two classes (male and female). The output layer employs softmax activation and sparse categorical crossentropy loss. The model is compiled with the Adam optimizer, and the learning rate is set to 0.0001.

During training, the model is fed with the UTKFace dataset, and both age and gender outputs are optimized simultaneously using their respective loss functions.

## Training
The model is trained for 2 sessions, each consisting of 5 epochs with a validation split of 20%. Training metrics, including mean absolute error (MAE) for age and accuracy for gender, are monitored.

## Training Analysis
The training history is visualized with plots for gender output accuracy and loss, as well as age output loss. These plots provide insights into the model's performance and convergence during training.

**First 5 epochs results:**
![Picture1](https://github.com/josephib1/Age-and-Gender-detection/assets/105210115/4be266f8-284d-4ea4-8bee-c9cb3eb9292c)
![Picture2](https://github.com/josephib1/Age-and-Gender-detection/assets/105210115/76b0d8b8-132d-453d-88da-0e4b342350c9)
![Picture3](https://github.com/josephib1/Age-and-Gender-detection/assets/105210115/f9ea1f8d-7ca3-4040-ba3c-13d149227895)

**Second 5 epochs Results:**
![Picture4](https://github.com/josephib1/Age-and-Gender-detection/assets/105210115/63e1429f-e619-4941-8c00-b75f22000b79)
![Picture5](https://github.com/josephib1/Age-and-Gender-detection/assets/105210115/a59aaf5f-06fb-4265-a110-c5dbd9765035)

## Model Evaluation
The last section of the code demonstrates how to use the trained model for making predictions on new images. A sample image from the 'my image' directory is loaded, preprocessed, and passed through the model. The predicted age and gender are then displayed alongside the input image.

## Conclusion
In conclusion, this code showcases the implementation of age and gender detection using the MobileNetV2 architecture. The model is trained on the UTKFace dataset, and its performance is analyzed through training history plots. Additionally, the code provides a practical example of using the trained model for real-world predictions on new images.
