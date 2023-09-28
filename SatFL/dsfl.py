"""A (naive, non-physically simulated) implementation of Decentralized Satellite Federated Learning
in DSFL: Decentralized Satellite Federated Learning for Energy-Aware LEO Constellation Computing by Wu et al."""

'''
We use the terms client/satellite/sat/node and cluster/plane/orbit interchangeably

Message Tags (Completely disambiguate messages: with msg tag to id training phase AND source location specified by recv call):
Model Initialization: 1 (tag value)
Model Distribution, 1st pass (along each cluster's ring structure): 2
Model Distribution, 2nd pass: 3
Partial Aggregation, 1st pass: 4
Partial Aggregation, 2nd pass: 5
Cluster Communication: 6+c


(Simulated on 8-core M1 Mac)
'''

import random
from mpi4py import MPI
import numpy as np
import utils
import copy

''' Initialize Client data'''
comm = MPI.COMM_WORLD

rank = comm.Get_rank()
num_sats = 7  # 'sat' is shorthand for 'satellite'
server_rank = num_sats
self_model = utils.create_model('iid')  # every client/server has internal model
# maps each satellite rank to a list of neighbor ranks (ring topology in-orbit -- 2 neighbors)
neighbors = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4], 4: [3], 5: [6], 6: [5]}
samples = utils.get_num_samples(num_sats,'iid')  # list of # of samples for every client (across clusters)
print(samples)

map_rank_to_cluster_id = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: -1}
cluster_ranks = {0: [0, 1, 2], 1: [3, 4], 2: [5, 6]}
if rank != server_rank:
    clust = map_rank_to_cluster_id[rank]
    self_clust_ranks = cluster_ranks[clust]
    local_rank = cluster_ranks[clust].index(rank)
num_clusters = 3  # number of orbital planes/clusters (of satellites)

if rank != server_rank:
    data = utils.load_data(
        rank,'iid')  # data: [train_x, train_y, test_x, test_y] | The local data collected during satellite in-orbit movement

# Assume all satellites are initialized with information about constellation structure
# and initial position (3 coordinates) of every satellite across all clusters
# - given initial positions, (real elapsed) time is the only parameter necessary to determine sat position
# (system is entirely constrained, state can be entirely calculated via orbital mechanics at any instance)

epochs = 100
batch_size = 15
num_iters = 5  # total number of FedAvg training iterations (several epochs/iteration)

'''Parameter Server Operation'''
if rank == server_rank:
    # initiate training (initialize all clients to same initial model)
    for ranks in cluster_ranks.values():  # send initial model to 'first' client of each cluster, that client propagates model within their own cluster from there
        comm.send(copy.deepcopy(self_model.get_weights()), dest=ranks[0])
    for it in range(num_iters + 1):
        '''Inter-Orbit Path Generation: (only for non-physical simulation)'''
        # for every cluster c, server generates randomized path from c across all clusters
        # these generated paths broadcasted to all satellites
        '''Caveats'''
        # Given graph of cluster nodes and edges representing cluster connectivity (weighted by min communication cost btw two sats in the pair of clusters),
        # each generated path represents shortest-path tree as calculated by Dijkstra's algorithm

        # path: sequence of sat's (that spans all clusters)

        # given this simulation is non-physical, so communication costs cannot be calculated, paths are randomized instead of actually calculated by Dijkstra's
        # (only in simulation is server involved each iteration, in reality path would be defined by the physical positions of satellites but can't model this without satellite tool kit)
        paths = {c: [c] for c in range(
            num_clusters)}  # "shortest-path tree" from each starting cluster (approximated as a randomly generated path)
        for c in range(num_clusters):
            other_clusts = []
            for k in range(num_clusters):
                if not k == c:
                    other_clusts.append(k)
            random.shuffle(other_clusts)
            paths[c] = paths[c] + other_clusts
            # update path to include individual sat.'s as members
            for ind, k in enumerate(paths[c]):
                paths[c][ind] = cluster_ranks[k][random.randint(0, len(cluster_ranks[k]) - 1)]

        comm.bcast(obj=paths, root=server_rank)
    # receive final global model (every satellite will contain the same one at the end of training)
    global_model_weights = comm.recv(source=0)
    self_model.set_weights(global_model_weights)

'''Satellite Operation'''
if rank != server_rank:
    '''Model Initialization'''
    left_cluster_neighbor = cluster_ranks[clust][(local_rank - 1) % len(cluster_ranks[clust])]
    right_cluster_neighbor = cluster_ranks[clust][(local_rank + 1) % len(cluster_ranks[clust])]
    if self_clust_ranks[0] == rank:  # 'first' client of the cluster (cluster rank of 0)
        self_model.set_weights(comm.recv(source=server_rank))
        # start propagation of model via right (arbitrary) neighbor of first client (in ring)
        comm.send(obj=(rank, copy.deepcopy(self_model.get_weights()), 1), dest=right_cluster_neighbor, tag=1)
    else:
        model_info = comm.recv(source=left_cluster_neighbor,
                               tag=1)  # model_info[0]: sender_rank, model_info[1]: initial model, model_info[2]: num of clients in cluster model has been propagated to
        self_model.set_weights(model_info[1])
        if model_info[2] + 1 < len(
                cluster_ranks[map_rank_to_cluster_id[rank]]):  # make sure not sending model to sat alr having model
            comm.send(obj=(rank, copy.deepcopy(model_info[1]), model_info[2] + 1), dest=right_cluster_neighbor, tag=1)

    external_clusters_aggregate = []  # aggregate of external cluster aggregates received by client in the previous epoch
    # Single client *can* receive multiple cluster aggregates
    # (filled with zeros if no external cluster aggregate was received)

    for ind, weight in enumerate(self_model.get_weights()):
        external_clusters_aggregate.append(np.zeros(weight.shape))


    # once initial model has been received, training is initiated:
    for it in range(num_iters + 1):
        paths = {c: [c] for c in range(num_clusters)}
        paths = comm.bcast(obj=paths, root=server_rank)
        '''Distribution Phase'''
        # every client in cluster ends up with global model
        # find global_ext_update based on two passes along ring: see https://mpitutorial.com/tutorials/mpi-send-and-receive/
        '''first pass: loop through every client in cluster ring, find global_ext_update'''
        global_external_update = copy.deepcopy(
            external_clusters_aggregate)  # global external update: aggregate of all external cluster aggregates received by cluster
        if local_rank != 0:
            global_external_update = comm.recv(source=left_cluster_neighbor, tag=2)
            local_weights = copy.deepcopy(external_clusters_aggregate)
            for ind, weight in enumerate(global_external_update):
                global_external_update[ind] = local_weights[ind] + weight

        comm.send(copy.deepcopy(global_external_update),
                  dest=right_cluster_neighbor, tag=2)

        if local_rank == 0:
            global_external_update = comm.recv(source=left_cluster_neighbor, tag=2)
            external_clusters_aggregate = global_external_update
        # end of first pass: ext_clust of 'first client' set to the global external update

        '''Second pass'''
        #   set every ext_clust to global_ext_update
        if local_rank != 0:
            external_clusters_aggregate = comm.recv(source=left_cluster_neighbor,
                                                    tag=3)  # receives global external update
        comm.send(copy.deepcopy(external_clusters_aggregate),
                  dest=right_cluster_neighbor, tag=3)
        if local_rank == 0:
            external_clusters_aggregate = comm.recv(source=left_cluster_neighbor, tag=3)
        # set self_model to sum of global_ext_update and original self_model (update self_model to the global model)
        global_weights = copy.deepcopy(self_model.get_weights())
        for ind, weight in enumerate(global_weights):
            global_weights[ind] = external_clusters_aggregate[ind] + weight
        self_model.set_weights(global_weights)

        if it == num_iters:  # stop training at point when all clients have same global model
            if rank == 0:
                comm.send(dest=server_rank, obj=copy.deepcopy(self_model.get_weights()))
            break

        '''Local Training Phase'''
        self_model.fit(x=data[0], y=data[1], batch_size=batch_size,
                       epochs=epochs,
                       shuffle=True)
        '''Partial Aggregation Phase'''
        partial_aggregate = copy.deepcopy(
            self_model.get_weights())  # outgoing partial aggregate (computed in aggregation phase)
        # partial aggregate via two passes along ring - see https://mpitutorial.com/tutorials/mpi-send-and-receive/
        '''first pass: calculate cluster aggregate'''
        if local_rank != 0:
            partial_aggregate = comm.recv(source=left_cluster_neighbor, tag=4)  # weights of the partial aggregate
            local_weights = copy.deepcopy(self_model.get_weights())
            # aggregate with local model & 'amplify' based on size of local dataset
            for ind, weight in enumerate(partial_aggregate):
                partial_aggregate[ind] = local_weights[ind] * samples[rank] + weight

        comm.send(copy.deepcopy(partial_aggregate),
                  dest=right_cluster_neighbor, tag=4)
        if local_rank == 0:
            partial_aggregate = comm.recv(source=left_cluster_neighbor, tag=4)  # weights of the partial aggregate
            local_weights = copy.deepcopy(self_model.get_weights())
            # aggregate with local model & 'amplify' based on size of local dataset
            for ind, weight in enumerate(partial_aggregate):
                partial_aggregate[ind] = local_weights[ind] * samples[rank] + weight
            self_model.set_weights(partial_aggregate)

        '''second pass: propagate cluster aggregate'''
        if local_rank != 0:
            partial_aggregate = comm.recv(source=left_cluster_neighbor, tag=5)
            self_model.set_weights(partial_aggregate)

        comm.send(copy.deepcopy(partial_aggregate), dest=right_cluster_neighbor, tag=5)

        if local_rank == 0:
            partial_aggregate = comm.recv(source=left_cluster_neighbor, tag=5)
            self_model.set_weights(partial_aggregate)

        # scale each self_model by the sum of all samples
        scaled_weights = copy.deepcopy(self_model.get_weights())
        for ind, weight in enumerate(scaled_weights):
            scaled_weights[ind] = weight / sum(samples)
        self_model.set_weights(scaled_weights)

        # end of second pass: every client in cluster stores cluster aggregate under self_model
        '''Cluster Communication Phase'''
        # clusters exchange cluster aggregates st every cluster ends up with every other cluster's aggregate in
        # addition to its own (which then combined to form global model)
        ext_clust_aggr = []
        for c, path in paths.items():
            if rank in path:
                ind = path.index(rank)
                if ind == 0:  # start of path
                    comm.send(dest=path[ind+1], obj=copy.deepcopy(self_model.get_weights()), tag=6+c)
                if ind == len(path) - 1:  # end of path
                    ext_clust_aggr = comm.recv(source=path[ind-1], tag=6+c)
                if 0 < ind < len(path) - 1:  # to avoid deadlock order send/recv btw pair of sats in-connection based on rank
                    ext_clust_aggr = comm.recv(source=path[ind - 1], tag=6+c)
                    comm.send(dest=path[ind + 1], obj=ext_clust_aggr, tag=6+c)

                for ind, weight in enumerate(ext_clust_aggr):
                    external_clusters_aggregate[ind] = external_clusters_aggregate[ind] + weight

# # Final Evaluation of global model - first initialize all to final global model, then eval on their local dataset
if rank == server_rank:
    comm.bcast(self_model.get_weights(), root=server_rank)
else:
    global_weights = None
    global_weights = comm.bcast(global_weights, root=server_rank)
    self_model.set_weights(global_weights)
    self_model.evaluate(x=data[2], y=data[3], batch_size=batch_size)
