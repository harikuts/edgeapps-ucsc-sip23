import tensorflow as tf
import json


def get_num_samples(num_clients, dataset='non-iid'):
    """returns list of number of (training) samples for first num_client clients"""
    if dataset == 'non-iid':
        path = 'data/femnist_niid_train.json'
        with open(path) as train:
            return json.load(train)['num_samples'][0:num_clients]
    if dataset == 'iid':
        samples = []
        for i in range(7):  # 7 clients
            path = 'data/femnist_iid_train_user_' + str(i + 1) + '.json'
            with open(path) as train:
                samples.append(json.load(train)['num_samples'][0])
        return samples


def load_data(client_id, dataset='non-iid'):
    """Load train and test data from data folder"""
    if dataset == 'non-iid':
        path = 'data/femnist_niid_train.json' 
        out = list()
        with open(path) as train:
            train_data = json.load(train)
            user = train_data['users'][client_id]
            out.append(train_data['user_data'][user]['x'])
            out.append(train_data['user_data'][user]['y'])
        path = 'data/femnist_niid_test.json'
        with open(path) as test:
            test_data = json.load(test)
            user = test_data['users'][client_id]
            out.append(test_data['user_data'][user]['x'])
            out.append(test_data['user_data'][user]['y'])
        return out
    if dataset == 'iid':
        path = 'data/femnist_iid_train_user_' + str(client_id + 1) + '.json'
        out = list()
        with open(path) as train:
            train_data = json.load(train)
            user = train_data['users'][0]
            out.append(train_data['user_data'][user]['x'])
            out.append(train_data['user_data'][user]['y'])
        path = 'data/femnist_iid_test_user_' + str(client_id + 1) + '.json'
        with open(path) as test:
            test_data = json.load(test)
            user = test_data['users'][0]
            out.append(test_data['user_data'][user]['x'])
            out.append(test_data['user_data'][user]['y'])
        return out


def create_model(dataset='non-iid'):
    num_classes = 62
    image_size = 28
    # model credit: cnn.py in LEAF/models/femnist repo and https://www.kaggle.com/code/jeckowturtle/letter-emnist-cnn-92-training-acc-90-test-acc
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((image_size, image_size, 1), input_shape=(image_size * image_size,)))
    if dataset == 'non-iid':
        model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                               strides=2))  
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same",
                                         activation='relu'))  
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same", activation='relu'))  #
    if dataset == 'iid':
        model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                               strides=2))  
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same",
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same", activation='relu'))


    model.add(tf.keras.layers.Flatten())  
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=.0003),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model