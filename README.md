# Neural Network Classification in Python

Neural networks are a powerful class of machine learning models that have gained popularity due to their ability to effectively learn complex patterns in data. In this article, we'll explore how to implement a neural network classification model in Python using the TensorFlow library. We'll walk through the process of building, training, and evaluating a neural network for classification tasks, using a classic example dataset.

## Introduction to Neural Network Classification

Classification is a fundamental task in machine learning where the goal is to categorize input data points into predefined classes or categories. Neural network classification involves using a neural network model to learn a mapping from input features to output class labels. This process typically involves training the model on a labeled dataset, where each data point is associated with a specific class label.

## Dataset: Iris Flowers

For our classification example, we'll use the Iris dataset, a classic dataset in machine learning. The Iris dataset consists of 150 samples of iris flowers, each belonging to one of three species: Setosa, Versicolor, or Virginica. Each sample contains four features: sepal length, sepal width, petal length, and petal width.

## Implementation with TensorFlow

We'll implement the neural network classification model in Python using the TensorFlow library. TensorFlow is an open-source machine learning framework developed by Google that provides a comprehensive ecosystem of tools and libraries for building and deploying machine learning models.

### Building the Neural Network Model

We'll use TensorFlow's high-level Keras API to build the neural network model. Keras provides a user-friendly interface for building and training neural networks, making it easy to define the model architecture.

        import tensorflow as tf
        from tensorflow.keras import models, layers
        
        # Define the model architecture
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(4,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        
        # Compile the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

In this example, we define a simple feedforward neural network with two hidden layers, each containing 64 neurons and ReLU activation functions. The output layer consists of three neurons corresponding to the three classes, with a softmax activation function to output class probabilities.

### Training the Model

Next, we'll train the model on the Iris dataset. We'll split the dataset into training and testing sets, with 80% of the data used for training and 20% for testing.

        # Train the model
        history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

During training, the model learns to minimize the categorical cross-entropy loss function using the Adam optimizer.

### Evaluating the Model

Once the model is trained, we can evaluate its performance on the testing set to assess its accuracy.

        # Evaluate the model on the testing set
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f'Test Accuracy: {accuracy}')
        
Output:

        1/1 [==============================] - 0s 33ms/step - loss: 0.2643 - accuracy: 0.9667
        Test Accuracy: 0.9666666388511658

We can also visualize the training process using Matplotlib to plot the training and validation accuracy over epochs.

        # Plot training and validation accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        
Output:

![image](https://github.com/bintangrizqikhairullah/Neural-Network-Regression/assets/101108509/70c32bc3-e9b4-41d2-902d-c07ae1ac6f56)

### Conclusion
In this article, we've explored how to implement a neural network classification model in Python using the TensorFlow library. We used the Iris dataset as a simple example to demonstrate the process of building, training, and evaluating a neural network for classification tasks. TensorFlow provides a powerful and flexible framework for building various types of machine learning models, making it a valuable tool for practitioners and researchers alike.
