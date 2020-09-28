import tensorflow as tf

# load the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# convert the integers to floating point numbers
x_train, x_test = x_train / 255.0, x_test / 255.0


# build the model by stacking layers, choose an optimizer and loss function
print("-----------------------------CREATING MODEL---------------------------")
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
print("-----------------------------MODEL CREATED----------------------------")

# for each example the model returns a vector of "logits" scores, one for each
# class
predictions = model(x_train[:1]).numpy()

# The tf.nn.softmax function converts these logits to probs for each class
tf.nn.softmax(predictions).numpy()

# Note: It is possible to bake this tf.nn.softmax in as the activation
# function for the last layer of the network. While this can make the model
# output more directly interpretable, this approach is discouraged as it's
# impossible to provide an exact and numerically stable loss calculation for
# all models when using a softmax output.

# The losses.SparseCategoricalCrossentropy loss takes a vector of logits and
# a true index and returns a scalar loss for each example

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# This loss = the negative log prob of the true class, it is zero if the model
# is sure of the correct class

# This untrained model gives probabilities close to random (1/10 for each
# class), initial loss should be ~ -tf.log(1/10) -= 2.3

# print(loss_fn(y_train[:1], predictions).numpy())

print("compiling model")
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
print("compiled")
# The model.fit method adjusts the model parameters to minimize the loss
print("----------------------------TRAINING MODEL----------------------------")
model.fit(x_train, y_train, epochs=5)
print("----------------------------MODEL TRAINED-----------------------------")

# the Model.evaluate method checks the models performance, usually on a
# "validation set"
model.evaluate(x_test, y_test, verbose=2)
print("----------------------------MODEL EVALUATED---------------------------")

# The image classifier is now trained to ~98% accuracy on this dataset

# If we want the model to return a probability, you can wrap the trained model
# and attach the softmax to it
print("\n\n")
print("--------------------------PROBABILITY MODEL---------------------------")
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

print("model: " + str(probability_model(x_test[:5])))

print("hello world")