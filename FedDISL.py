"""Decentralized FedISL algorithm with multiple source/sinks per cluster and no reliance on server during the
training"""

import server
import cluster
import utils
import cluster_client
import copy

# although algorithm is 'decentralized', server still needs to initialize all clients (after that server is completely
# ignored)
serv = server.Server()
# define clusters
map_id_to_cluster = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2}
map_cluster_to_ids = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11]}

# in simulation define arbitrary initial cluster sources - initial source_i,j is the first satellite from cluster i
# to come into contact with a satellite from cluster j

# maps each cluster index to a dictionary which maps each client rank within cluster to an initial list of cluster
# indices that the client is a source with respect to

# IMPORTANT:
# each client has a list of 'source roles' (given that multiple sources could be at same client node)
# ie client in cluster 0 could be source with respect to cluster 1 and cluster 2: so it would have source roles of source_0,1 and source_0,2
# in the same manner, each client has a list of 'sink roles'
# because a client can have multiple roles (as a source inclusive-or sink), looping through roles NOT same as looping through clients

initial_cluster_sources = {
    0: {0: [1], 1: [2], 2: [], 3: []},
    1: {0: [0, 2], 1: [], 2: [], 3: []},
    2: {0: [1], 1: [0], 2: [], 3: []}
}

'''create and initialize clients, clusters'''

clients = []  # temporary list of all clients from which clusters' members can be initialized
clusters = []

# create clients
for i in range(len(map_id_to_cluster)):
    clust = map_id_to_cluster[i]
    clients.append(
        cluster_client.ClusterClient(client_id=i, data=utils.load_data(i), cluster=clust,
                                     rank=map_cluster_to_ids[clust].index(i),
                                     scaling=serv.get_scaling()))  # initially ClusterClient.cluster initialized with
    # cluster_id not cluster object

# create clusters

for clust, ids in map_cluster_to_ids.items():
    members = [clients[c] for c in ids]
    sources = initial_cluster_sources[clust]
    sinks = {rank: [] for rank in range(len(ids))}  # initialize sinks as empty list
    clusters.append(cluster.Cluster(members, sources, sinks, clust))

# reassign actual cluster objects (not integer cluster_id's) to ClusterClient.cluster fields
for clust in clusters:
    for client in clust.members:
        client.cluster = clust

# establish same initial weights across all clients, cluster base models to enable model synchronization
for clust in clusters:
    for client in clust.members:
        client.model.set_weights(copy.deepcopy(clusters[0].base_model.get_weights()))
    clust.base_model.set_weights(copy.deepcopy(clusters[0].base_model.get_weights()))

# IMPORTANT: !!!make sure to *deep copy* weights (so that one model's updates don't affect another model it got
# weights from - otherwise get_weights() only returns references of weights, not the weight values)!!!

'''Training Procedure:'''
num_iters = 3

for it in range(num_iters):
    print("ITERATION", it)
    '''Sink calculation'''
    # within each cluster: loop through all source roles, calculate new sink roles with respect to each external cluster
    # ClusterClient.sink feature was reset at the end of the previous epoch
    for clust in clusters:
        for client_rank, source_roles in clust.source.items():
            # for each source role's external cluster that it corresponds to (ex source_self.cluster,j corresponding
            # to cluster j) update the new sink role that will correspond to that external cluster (ie for source_i,
            # j calculate the new sink_i,j)
            for ext_clust_id in source_roles:
                clust.sink[clust.members[client_rank].get_sink(ext_clust_id)].append(ext_clust_id)
        print(clust.sink)
    print("DONE", "SINK CALCULATION") #barrier synchronization (at end of each training phase)
    '''Model Distribution'''
    # by end of phase every client will have global aggregate of all client models across all clusters

    # for each node x that takes on source roles: x.model = Cluster.base_model (see cluster.py) + (sum of the
    # external cluster aggregates that each source role of x contributes)
    #
    # call (sum of the external cluster
    # aggregates that each source role of x contributes) client x's "contribution"
    #
    # sum up the contribution of every
    # client in cluster then update every client model in cluster to this sum + Cluster.base_model
    for clust in clusters:
        new_model_weights = copy.deepcopy(clust.base_model.get_weights()) #will be sum of client contributions + cluster's base_model
        for client in clust.members:
            for ind, local_weight in enumerate(client.model.get_weights()):
                new_model_weights[ind] = new_model_weights[ind] + (local_weight - clust.base_model.get_weights()[ind]) #take advantage of numpy addition/subtraction
        # set every client model in cluster to new_model
        for client in clust.members:
            client.model.set_weights(new_model_weights)
    print("DONE", "MODEL DISTRIBUTION")  # barrier synchronization (at end of each training phase)
    '''Local Training'''
    for clust in clusters:
        for client in clust.members:
            client.train_model()
    print("DONE", "LOCAL TRAINING")  # barrier synchronization (at end of each training phase)

    '''Partial Aggregation'''
    # at the end of this phase every client in a given cluster has its model updated to be its cluster aggregate

    # partial aggregating along ring topology is done in two passes: first pass sums up local updates into an
    # aggregate, second pass updates every client model to that aggregate
    for clust in clusters:
        curr_rank = 0 # treat node of rank 0 as arbitrary start of each pass/traversal
        start = True
        while start or curr_rank != 0:
            # print(curr_rank)
            # partial aggregate already scales based on proportion of device contribution to global dataset
            clust.members[curr_rank].partial_aggregate(clust.members[(curr_rank + 1) % len(clust.members)])
            curr_rank = (curr_rank + 1) % len(clust.members)
            start = False
        # ends with the cluster aggregate stored at client of rank 0

        curr_rank = 0  # treat node of rank 0 as arbitrary start of each pass/traversal
        start = True
        while start or curr_rank != 0:
            # print(curr_rank)
            curr_weights = copy.deepcopy(clust.members[curr_rank].model.get_weights())
            clust.members[(curr_rank + 1) % len(clust.members)].model.set_weights(curr_weights)
            curr_rank = (curr_rank + 1) % len(clust.members)
            start = False
        # ends with every client storing cluster aggregate

        # Update Cluster Base-Model based on *copy* of one of the client models (since that model may change in future)
        # CANNOT deepcopy model, can deepcopy model weights (just list of numpy arrays)
        clust.base_model.set_weights(copy.deepcopy(clust.members[0].model.get_weights()))

    print("Weights after Partial Aggregation")
    for clust in clusters:
        for client in clust.members:
            print('client', client.id)
            print(client.model.get_weights()[3])

    print("DONE", "PARTIAL AGGREGATION")  # barrier synchronization (at end of each training phase)
    '''Sink Cluster Communication'''
    # for every pair of sink roles sink_i,j and sink_j,i:
    # (treat Cluster.base_model as the 'cluster aggregate')
    # sink_i,j will share cluster i's aggregate with sink_j,i and sink_j,i will share cluster j's aggregate with sink_i,j
    # then sink_i,j aggregates its base_model with the received cluster j aggregate and vice versa for sink_j,i
    # finally sink_i,j and sink_j,i become the new source roles for their respective cluster pairings
    # (bc just as sinks are within communication range, sources also need to be within communication range for the new epoch)
    for clust in clusters:
        for client in clust.members:
            client_weights = copy.deepcopy(clust.base_model.get_weights())
            for ext_clust_id in clust.sink[client.rank]:
                # update client with ext_clust (external cluster) aggregate
                for ind, local_weight in enumerate(clusters[ext_clust_id].base_model.get_weights()):
                    client_weights[ind] = client_weights[ind] + local_weight
            client.model.set_weights(client_weights)
    # make sinks new sources (ie Cluster.source = Cluster.sink) and empty Cluster.sink for recalculation
    for clust in clusters:
        clust.source = clust.sink
        clust.sink = {rank: [] for rank in range(len(clust.members))}

    print("Weights after Sink Cluster Communication")
    for clust in clusters:
        for client in clust.members:
            print('client', client.id)
            print(client.model.get_weights()[3])

    print("DONE", "SINK CLUSTER COMMUNICATION")  # barrier synchronization (at end of each training phase)
    print("DONE ITERATION")

# One final post-training model distribution phase to ensure every client has the same global model
for clust in clusters:
    new_model_weights = copy.deepcopy(
        clust.base_model.get_weights())  # will be sum of client contributions + cluster's base_model
    for client in clust.members:
        for ind, local_weight in enumerate(client.model.get_weights()):
            new_model_weights[ind] = new_model_weights[ind] + (local_weight - clust.base_model.get_weights()[
                ind])  # take advantage of numpy addition/subtraction
    # set every client model in cluster to new_model
    for client in clust.members:
        client.model.set_weights(new_model_weights)

print("Final Weights")
for clust in clusters:
    for client in clust.members:
        print('client', client.id)
        print(client.model.get_weights()[3])

# test the global model on each client's data
global_model = clusters[0].members[0].model
for c in clients:
    global_model.evaluate(x=c.x_test_normalized, y=c.y_test, batch_size=15)
