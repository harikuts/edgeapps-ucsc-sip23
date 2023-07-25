"""Implementation of the FedISL algorithm
from On-Board Federated Learning in Satellite Constellations by Razmi et al."""

import server
import utils
import cluster_client
import cluster

serv = server.Server()
map_id_to_cluster = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2}
map_cluster_to_ids = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11]}
'''create and initialize clients, clusters: cluster 1: {0,1}, cluster 2: {2,3,4}'''

clients = []
clusters = []

# create clients
for i in range(len(map_id_to_cluster)):
    clust = map_id_to_cluster[i]
    clients.append(
        cluster_client.ClusterClient(client_id=i, data=utils.load_data(i), cluster=clust,
                                     rank=map_cluster_to_ids[clust].index(i),
                                     scaling=serv.get_scaling())) # initially ClusterClient.cluster initialized with
    # cluster_id not cluster object

# create clusters
for clust, ids in map_cluster_to_ids.items():
    members = [clients[c] for c in ids]
    # for simulation: initialized with arbitrary source, sink
    clusters.append(cluster.Cluster(members, members[0], members[1], clust))

# reassign actual cluster objects (not integer cluster_id's) to ClusterClient.cluster fields
for clust in clusters:
    for client in clust.members:
        client.cluster = clust

'''for each iteration of fed avg (could be several epochs each bc unreliable connectivity):'''
num_iters = 3
for it in range(num_iters):
    print("ITERATION", it)
    '''Model distribution'''
    # server distributes global model to source of each cluster
    for clust in clusters:
        clust.source.model = serv.model
        # each source distributes the received model to all nodes in cluster
        clust.source.distribute_model()
    print("DONE", "MODEL DISTRIBUTION")
    '''Local Training'''
    # server signal for local training phase received on cluster-by-cluster basis
    for clust in clusters:
        for client in clust.members:
            client.train_model()
    print("DONE", "LOCAL TRAINING")
    '''Partial Aggregation and Communicating Updates'''
    # partial aggregate along ring network
    # assume ring topology

    for clust in clusters:
        curr_rank = clust.sink.rank
        start = True
        while start or curr_rank != clust.sink.rank:
            # print(curr_rank)
            clust.members[curr_rank].partial_aggregate(clust.members[(curr_rank + 1) % len(clust.members)])
            curr_rank = (curr_rank + 1) % len(clust.members)
            start = False

    print("DONE", "Partial Aggregation and Communicating Updates")
    '''Final Aggregation and End of FedAvg Iteration'''
    models = [clust.sink.model for clust in clusters]
    serv.aggregate(models)
    print("DONE ITERATION")

# test final server model for each client's test set
for c in clients:
    serv.model.evaluate(x=c.x_test_normalized, y=c.y_test, batch_size=15)
