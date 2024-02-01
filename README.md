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

![First 5 epochs results](C:\Users\joseph.ibrahim02\Desktop\Deep Learning\Plots\Picture1.png)(C:\Users\joseph.ibrahim02\Desktop\Deep Learning\Plots\Picture2.png)(C:\Users\joseph.ibrahim02\Desktop\Deep Learning\Plots\Picture3.png)

**Second 5 epochs Results:**
*Visualization goes here*

## Model Evaluation
The last section of the code demonstrates how to use the trained model for making predictions on new images. A sample image from the 'my image' directory is loaded, preprocessed, and passed through the model. The predicted age and gender are then displayed alongside the input image.

## Conclusion
In conclusion, this code showcases the implementation of age and gender detection using the MobileNetV2 architecture. The model is trained on the UTKFace dataset, and its performance is analyzed through training history plots. Additionally, the code provides a practical example of using the trained model for real-world predictions on new images.
