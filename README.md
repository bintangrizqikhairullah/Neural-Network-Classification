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

Output :

        Epoch 1/50
        WARNING:tensorflow:From c:\Users\ASUS\anaconda3\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.
        
        WARNING:tensorflow:From c:\Users\ASUS\anaconda3\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.
        
        3/3 [==============================] - 1s 128ms/step - loss: 1.7795 - accuracy: 0.3263 - val_loss: 1.3638 - val_accuracy: 0.3750
        Epoch 2/50
        3/3 [==============================] - 0s 20ms/step - loss: 1.3021 - accuracy: 0.3263 - val_loss: 1.0877 - val_accuracy: 0.3750
        Epoch 3/50
        3/3 [==============================] - 0s 21ms/step - loss: 1.0550 - accuracy: 0.4526 - val_loss: 0.9771 - val_accuracy: 0.6250
        Epoch 4/50
        3/3 [==============================] - 0s 22ms/step - loss: 0.9776 - accuracy: 0.5368 - val_loss: 0.9451 - val_accuracy: 0.4167
        Epoch 5/50
        3/3 [==============================] - 0s 22ms/step - loss: 0.9373 - accuracy: 0.4105 - val_loss: 0.9155 - val_accuracy: 0.3750
        Epoch 6/50
        3/3 [==============================] - 0s 24ms/step - loss: 0.9029 - accuracy: 0.4632 - val_loss: 0.8741 - val_accuracy: 0.5417
        Epoch 7/50
        3/3 [==============================] - 0s 21ms/step - loss: 0.8588 - accuracy: 0.5579 - val_loss: 0.8243 - val_accuracy: 0.7083
        Epoch 8/50
        3/3 [==============================] - 0s 21ms/step - loss: 0.8181 - accuracy: 0.6737 - val_loss: 0.7744 - val_accuracy: 0.6250
        Epoch 9/50
        3/3 [==============================] - 0s 23ms/step - loss: 0.7769 - accuracy: 0.6737 - val_loss: 0.7355 - val_accuracy: 0.6250
        Epoch 10/50
        3/3 [==============================] - 0s 20ms/step - loss: 0.7341 - accuracy: 0.6737 - val_loss: 0.6929 - val_accuracy: 0.6250
        Epoch 11/50
        3/3 [==============================] - 0s 21ms/step - loss: 0.7045 - accuracy: 0.6737 - val_loss: 0.6590 - val_accuracy: 0.7500
        Epoch 12/50
        3/3 [==============================] - 0s 20ms/step - loss: 0.6773 - accuracy: 0.7053 - val_loss: 0.6258 - val_accuracy: 0.9167
        Epoch 13/50
        3/3 [==============================] - 0s 21ms/step - loss: 0.6502 - accuracy: 0.7579 - val_loss: 0.5965 - val_accuracy: 0.8750
        Epoch 14/50
        3/3 [==============================] - 0s 21ms/step - loss: 0.6202 - accuracy: 0.7579 - val_loss: 0.5656 - val_accuracy: 0.8750
        Epoch 15/50
        3/3 [==============================] - 0s 20ms/step - loss: 0.5904 - accuracy: 0.7789 - val_loss: 0.5358 - val_accuracy: 0.9167
        Epoch 16/50
        3/3 [==============================] - 0s 21ms/step - loss: 0.5654 - accuracy: 0.7158 - val_loss: 0.5194 - val_accuracy: 0.7083
        Epoch 17/50
        3/3 [==============================] - 0s 22ms/step - loss: 0.5487 - accuracy: 0.6842 - val_loss: 0.5019 - val_accuracy: 0.7083
        Epoch 18/50
        3/3 [==============================] - 0s 21ms/step - loss: 0.5263 - accuracy: 0.7263 - val_loss: 0.4740 - val_accuracy: 0.9167
        Epoch 19/50
        3/3 [==============================] - 0s 20ms/step - loss: 0.5076 - accuracy: 0.8526 - val_loss: 0.4516 - val_accuracy: 1.0000
        Epoch 20/50
        3/3 [==============================] - 0s 20ms/step - loss: 0.4948 - accuracy: 0.9053 - val_loss: 0.4327 - val_accuracy: 1.0000
        Epoch 21/50
        3/3 [==============================] - 0s 26ms/step - loss: 0.4820 - accuracy: 0.9368 - val_loss: 0.4207 - val_accuracy: 1.0000
        Epoch 22/50
        3/3 [==============================] - 0s 29ms/step - loss: 0.4673 - accuracy: 0.9158 - val_loss: 0.4082 - val_accuracy: 1.0000
        Epoch 23/50
        3/3 [==============================] - 0s 24ms/step - loss: 0.4556 - accuracy: 0.8421 - val_loss: 0.4013 - val_accuracy: 1.0000
        Epoch 24/50
        3/3 [==============================] - 0s 25ms/step - loss: 0.4480 - accuracy: 0.7895 - val_loss: 0.3896 - val_accuracy: 1.0000
        Epoch 25/50
        3/3 [==============================] - 0s 31ms/step - loss: 0.4331 - accuracy: 0.8526 - val_loss: 0.3693 - val_accuracy: 1.0000
        Epoch 26/50
        3/3 [==============================] - 0s 33ms/step - loss: 0.4251 - accuracy: 0.9368 - val_loss: 0.3528 - val_accuracy: 1.0000
        Epoch 27/50
        3/3 [==============================] - 0s 35ms/step - loss: 0.4166 - accuracy: 0.9474 - val_loss: 0.3460 - val_accuracy: 1.0000
        Epoch 28/50
        3/3 [==============================] - 0s 25ms/step - loss: 0.4054 - accuracy: 0.9053 - val_loss: 0.3380 - val_accuracy: 1.0000
        Epoch 29/50
        3/3 [==============================] - 0s 23ms/step - loss: 0.3954 - accuracy: 0.9053 - val_loss: 0.3234 - val_accuracy: 1.0000
        Epoch 30/50
        3/3 [==============================] - 0s 21ms/step - loss: 0.3859 - accuracy: 0.9474 - val_loss: 0.3128 - val_accuracy: 1.0000
        Epoch 31/50
        3/3 [==============================] - 0s 21ms/step - loss: 0.3789 - accuracy: 0.9474 - val_loss: 0.3047 - val_accuracy: 1.0000
        Epoch 32/50
        3/3 [==============================] - 0s 23ms/step - loss: 0.3691 - accuracy: 0.9368 - val_loss: 0.2939 - val_accuracy: 1.0000
        Epoch 33/50
        3/3 [==============================] - 0s 24ms/step - loss: 0.3622 - accuracy: 0.9474 - val_loss: 0.2826 - val_accuracy: 1.0000
        Epoch 34/50
        3/3 [==============================] - 0s 28ms/step - loss: 0.3553 - accuracy: 0.9474 - val_loss: 0.2739 - val_accuracy: 1.0000
        Epoch 35/50
        3/3 [==============================] - 0s 32ms/step - loss: 0.3470 - accuracy: 0.9579 - val_loss: 0.2615 - val_accuracy: 1.0000
        Epoch 36/50
        3/3 [==============================] - 0s 23ms/step - loss: 0.3396 - accuracy: 0.9579 - val_loss: 0.2531 - val_accuracy: 1.0000
        Epoch 37/50
        3/3 [==============================] - 0s 24ms/step - loss: 0.3311 - accuracy: 0.9684 - val_loss: 0.2459 - val_accuracy: 1.0000
        Epoch 38/50
        3/3 [==============================] - 0s 25ms/step - loss: 0.3241 - accuracy: 0.9579 - val_loss: 0.2398 - val_accuracy: 1.0000
        Epoch 39/50
        3/3 [==============================] - 0s 26ms/step - loss: 0.3189 - accuracy: 0.9474 - val_loss: 0.2288 - val_accuracy: 1.0000
        Epoch 40/50
        3/3 [==============================] - 0s 23ms/step - loss: 0.3110 - accuracy: 0.9579 - val_loss: 0.2210 - val_accuracy: 1.0000
        Epoch 41/50
        3/3 [==============================] - 0s 23ms/step - loss: 0.3027 - accuracy: 0.9684 - val_loss: 0.2125 - val_accuracy: 1.0000
        Epoch 42/50
        3/3 [==============================] - 0s 23ms/step - loss: 0.2971 - accuracy: 0.9684 - val_loss: 0.2083 - val_accuracy: 1.0000
        Epoch 43/50
        3/3 [==============================] - 0s 22ms/step - loss: 0.2885 - accuracy: 0.9579 - val_loss: 0.1980 - val_accuracy: 1.0000
        Epoch 44/50
        3/3 [==============================] - 0s 27ms/step - loss: 0.2815 - accuracy: 0.9684 - val_loss: 0.1880 - val_accuracy: 1.0000
        Epoch 45/50
        3/3 [==============================] - 0s 23ms/step - loss: 0.2753 - accuracy: 0.9684 - val_loss: 0.1810 - val_accuracy: 1.0000
        Epoch 46/50
        3/3 [==============================] - 0s 25ms/step - loss: 0.2687 - accuracy: 0.9684 - val_loss: 0.1743 - val_accuracy: 1.0000
        Epoch 47/50
        3/3 [==============================] - 0s 31ms/step - loss: 0.2658 - accuracy: 0.9684 - val_loss: 0.1657 - val_accuracy: 1.0000
        Epoch 48/50
        3/3 [==============================] - 0s 28ms/step - loss: 0.2569 - accuracy: 0.9684 - val_loss: 0.1628 - val_accuracy: 1.0000
        Epoch 49/50
        3/3 [==============================] - 0s 31ms/step - loss: 0.2533 - accuracy: 0.9579 - val_loss: 0.1567 - val_accuracy: 1.0000
        Epoch 50/50
        3/3 [==============================] - 0s 23ms/step - loss: 0.2428 - accuracy: 0.9684 - val_loss: 0.1460 - val_accuracy: 1.0000


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
