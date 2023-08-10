import tensorflow as tf
import json


def load_data(client_id):
    path = 'femnist_13users_train.json'  # make sure to update server path (see server.__init__()
    out = list()
    with open(path) as train:
        train_data = json.load(train)
        user = train_data['users'][client_id]
        out.append(train_data['user_data'][user]['x'])
        out.append(train_data['user_data'][user]['y'])
    path = 'femnist_13users_test.json'
    with open(path) as test:
        test_data = json.load(test)
        user = test_data['users'][client_id]
        out.append(test_data['user_data'][user]['x'])
        out.append(test_data['user_data'][user]['y'])
    return out


def create_model():
    """for each (non-input/dropout) layer: there are two elements of get_weights(): list of weight vectors, list of bias for each unit
        - list of weight vectors: for each unit of previous layer there is a weight vector of weights associated with the connection between that unit and each unit of the current layer
        (each weight vector: lists the weights associated with the connection of the given unit to each unit of previous layer connecting to it)
    """
    num_classes = 62
    image_size = 28
    # model credit: cnn.py in LEAF/models/femnist repo and https://www.kaggle.com/code/jeckowturtle/letter-emnist-cnn-92-training-acc-90-test-acc
    model = tf.keras.models.Sequential()
    # specifying input_shape implicitly creates InputLayer
    model.add(tf.keras.layers.Reshape((image_size, image_size, 1), input_shape=(image_size * image_size,)))
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           strides=2))  # reduce spatial size - replace each 2x2 square with its max
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same",
                                     activation='relu'))  # set padding="same" for output to have same size as input
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same", activation='relu'))  #
    model.add(
        tf.keras.layers.Flatten())  # reshape for neural network; shape inference with '-1'
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=.0003),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    '''# toy model -- focus isnt on accuracy
    # Fully Connected Neural Network -- poor performance bc # classes And available data
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(784,)))  # doesn't contribute any weights
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))  # size of get_weights independent of # units
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    # model.add(tf.keras.layers.Dropout(rate=0.2)) #doesn't contribute any weights
    model.add(tf.keras.layers.Dense(units=62,
                                    activation='softmax'))  # EMNIST classes include digits and upper/lowercase letters
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=.003),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    '''
    return model
