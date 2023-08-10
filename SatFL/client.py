import pandas as pd
import utils


class Client:
    epochs = 100
    batch_size = 15

    def __init__(self, client_id, data):
        self.model = utils.create_model()
        self.id = client_id
        self.x_train_normalized = data[0]
        self.y_train = data[1]
        self.x_test_normalized = data[2]
        self.y_test = data[3]

    # source: Google Machine Learning Crash Course
    def train_model(self):
        history = self.model.fit(x=self.x_train_normalized, y=self.y_train, batch_size=Client.batch_size,
                                 epochs=Client.epochs,
                                 shuffle=True)

        epochs = history.epoch
        hist = pd.DataFrame(history.history)

        return epochs, hist

    def evaluate(self):
        self.model.evaluate(x=self.x_test_normalized, y=self.y_test, batch_size=Client.batch_size)


# # relatively poor performance (~50 accuracy) expected given limited data
# '''training on individual client: non-federated, centralized ML training'''
# client_0 = Client(0, utils.load_data(0))
# client_0.train_model()
# client_0.evaluate()
