"""
Sequential implementation of centralized federated learning (server aggregation):
Each training iteration:
Server sends model to each device
Local training on device
Send local updates to server
server aggregates into new model

see https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399 and the implementation of server.py in LEAF repo
"""

import json
import client
import utils
import copy


class Server:
    def __init__(self):
        self.model = utils.create_model()  # identical model to clients
        self.num_samples = []  # specifically number of samples per client in training
        path = 'femnist_13users_train.json'
        with open(path) as train:
            self.num_samples = json.load(train)['num_samples']

    def get_scaling(self):
        """Calculates scaling of the weights (for aggregation) of each client based on client dataset contribution
        to total dataset"""
        total_samples = sum(self.num_samples)
        return [device_samples / total_samples for device_samples in self.num_samples]

    def aggregate(self, client_models):
        """aggregation of all clients updates using federated averaging and updates global model with aggregate"""

        scaling = self.get_scaling()
        weights = copy.deepcopy(self.model.get_weights())
        for client_id, client_model in enumerate(client_models):
            '''scale local updates based on device's relative contribution to the whole dataset'''
            scaled_local_update = copy.deepcopy(client_model.get_weights())
            for ind, local_weight in enumerate(client_model.get_weights()):
                scaled_local_update[ind] = local_weight * scaling[client_id]
            '''global model is aggregate/sum of scaled local updates'''
            for ind, local_weight in enumerate(scaled_local_update):
                weights[ind] += local_weight
        '''update global model with the aggregate and return'''
        self.model.set_weights(weights)
        return self.model


'''IMPORTANT:'''

# BASELINE: 74.72% accuracy on FEMNIST (https://arxiv.org/pdf/1812.01097.pdf)
#     #-results approach this for femnist_large_train (100 users) -- likely to match for full dataset (3,550 users)
#


# '''Centralized Federated Learning Procedure:'''
#
# serv = Server()
# print(serv.num_samples)
#
# # create and initialize clients
# clients = []
# for i in range(len(serv.num_samples)):
#     clients.append(client.Client(i, utils.load_data(i)))
# '''
# for c in client_models:
#     print(c.id, len(c.x_train_normalized))
# '''
#
# '''for each iteration of fed avg (could be several epochs each bc unreliable connectivity): '''
# # make sure to regularly re-average and update all local models to that average (each iteration) to make sure local
# # models don't diverge
# num_iters = 3
# for it in range(num_iters):
#     for c in clients:
#         c.model = serv.model  # download global model
#         c.train_model()  # local training
#     # send local updates to server and aggregate
#     serv.aggregate([c.model for c in clients])
#     # print(serv.model.get_weights())
#
# # test final server model for each client's test set
# for c in clients:
#     serv.model.evaluate(x=c.x_test_normalized, y=c.y_test, batch_size=15)
# # FL training should nearly match non-FL training (see client.py) for successful implementation
