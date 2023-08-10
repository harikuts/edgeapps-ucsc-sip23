"""Parallelized implementation of the FedISL algorithm from On-Board Federated Learning in Satellite Constellations
by Razmi et al.: very straightforward/direct implementation -- ignores fallback routing, assumes perfect connectivity
(experiments only focus on measuring final loss, epochs to convergence))

(Simulated on 8-core Mac)
"""
import time
import random

from mpi4py import MPI
import utils
import copy
import isl_helpers

''' Initialize Client/server data'''
comm = MPI.COMM_WORLD
'''for fed_isl_parallel, fed_disl_parallel !!!Ranks are non-local, they are not specific to cluster!!!'''
rank = comm.Get_rank()
num_sats = 7  # 'sat' is shorthand for 'satellite'
server_rank = num_sats
self_model = utils.create_model()  # every client/server has internal model
# maps each satellite rank to a list of neighbor ranks (ring topology in-orbit -- 2 neighbors)
neighbors = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4], 4: [3], 5: [6], 6: [5]}
samples = utils.get_num_samples(num_sats)  # list of # of samples for every client (across clusters)
print(samples)
# clusters 0, 1, 2 | Server has cluster_id = -1
map_rank_to_cluster_id = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: -1}
cluster_ranks = {0: [0, 1, 2], 1: [3, 4], 2: [5, 6]}
num_clusters = 3  # number of orbital planes/clusters (of satellites)

data = utils.load_data(rank)  # data: [train_x, train_y, test_x, test_y]

# for calculating sink nodes would also assume all clients init. with location of every satellite in their cluster
# and time-tracker


epochs = 100
batch_size = 15
num_iters = 3  # total number of FedAvg training iterations (several epochs/iteration)
for it in range(num_iters):

    '''Parameter Server Operation'''
    if map_rank_to_cluster_id[rank] == -1:
        # artificially designate each cluster's source for sake of simulation
        sources = [cluster_ranks[clust][random.randint(0, len(cluster_ranks[clust]) - 1)] for clust in
                   range(num_clusters)]
        comm.bcast(obj=sources, root=server_rank)
        '''Distribution Phase'''
        print(sources)
        ind = 0
        distributed = []  # list of cluster_ids to whom server has distributed model
        while len(distributed) < num_clusters:
            sat_id = sources[ind]  # 'upon incoming connection by (source) satellite k'
            orbit = map_rank_to_cluster_id[sat_id]
            if orbit not in distributed:
                # distribute *deepcopy* of server model weights to cluster source
                comm.send(obj=copy.deepcopy(self_model.get_weights()), dest=sat_id)
                distributed.append(orbit)
            print('distr', distributed, 'iter', it)
            # else: terminate connection
            ind = ind + 1
        # print('server', self_model.get_weights()[0])
        # comm.barrier()
        '''Aggregation Phase'''
        received = []
        global_weights = -1 # global model replaced with aggregate of cluster aggregates each iteration
        while len(received) < num_clusters:
            local_info = comm.recv() # local_info[0]: sink_rank, local_info[1]: cluster aggregate weights
            orbit = map_rank_to_cluster_id[local_info[0]]
            amp_local_weights = copy.deepcopy(local_info[1]) # cluster aggregate weights; received weights were 'amplified' by the size of their local dataset (ie D_k * w_k)
            if orbit not in received:
                # update global weight
                if global_weights == -1: # init. global_weights
                    global_weights = amp_local_weights
                else:
                    for ind, local_weight in enumerate(amp_local_weights):
                        global_weights[ind] += local_weight
                received.append(orbit)
        # scale amplified global_weights to (w_global/D_global) to account for contribution of each client *relative* whole dataset
        for ind, weight in enumerate(global_weights):
            global_weights[ind] = weight / sum(samples)
        self_model.set_weights(global_weights)

    '''Satellite Operation'''
    if map_rank_to_cluster_id[rank] != -1:
        sources = None
        sources = comm.bcast(obj=sources, root=server_rank)
        print(sources)
        '''Distribution Phase'''
        if rank in sources:  # for sim assume PS visible only to ranks of 'sources' list
            self_model.set_weights(comm.recv())  # update local model to distributed global model
            clust = map_rank_to_cluster_id[rank]
            # if source, calculate sink
            sink = isl_helpers.get_sink(cluster_ranks[clust])
            print('sink', sink)
            # start propagation of model via left (arbitrary) neighbor of source
            comm.send(obj=(rank, sink, copy.deepcopy(self_model.get_weights()), 1), dest=neighbors[rank][0])
        else:
            # if not a source:
            # model_info[0]: sender_rank, model_info[1]: sink_rank, model_info[2]: copied model weights, model_info[3]: number of sats model has been distributed to in cluster
            model_info = comm.recv()
            sink = model_info[1]
            self_model.set_weights(model_info[2])
            for sat_rank in neighbors[rank]:
                if sat_rank != model_info[0] and model_info[3] + 1 < len(cluster_ranks[map_rank_to_cluster_id[rank]]): # make sure not sending model to sat alr having model
                    comm.send(obj=(rank, model_info[1], copy.deepcopy(model_info[2]), model_info[3]+1), dest=sat_rank)
        # print('rank', rank, self_model.get_weights()[0])

        '''Computation Phase'''
        self_model.fit(x=data[0], y=data[1], batch_size=batch_size,
                       epochs=epochs,
                       shuffle=True)
        print('LTrank', rank, self_model.get_weights()[0])
        # comm.barrier()
        '''Partial Aggregation Phase'''
        partial_aggregate = copy.deepcopy(self_model.get_weights() ) # outgoing partial aggregate (computed in aggregation phase)
        # partial aggregate via two passes along ring - see https://mpitutorial.com/tutorials/mpi-send-and-receive/
        clust = map_rank_to_cluster_id[rank]
        local_rank = cluster_ranks[clust].index(rank)
        if local_rank != 0:  # avoid deadlock
            partial_aggregate = comm.recv()  # weights of the partial aggregate
            local_weights = copy.deepcopy(self_model.get_weights())
            # aggregate with local model & 'amplify' based on size of local dataset
            for ind, weight in enumerate(partial_aggregate):
                partial_aggregate[ind] = local_weights[ind] * samples[rank] + partial_aggregate[ind]

        comm.send(copy.deepcopy(partial_aggregate),
                  dest=cluster_ranks[clust][(local_rank + 1) % len(cluster_ranks[clust])])

        if local_rank == 0:
            partial_aggregate = comm.recv()  # weights of the partial aggregate
            local_weights = copy.deepcopy(self_model.get_weights())
            # aggregate with local model & 'amplify' based on size of local dataset
            for ind, weight in enumerate(partial_aggregate):
                partial_aggregate[ind] = local_weights[ind] * samples[rank] + partial_aggregate[ind]
            self_model.set_weights(partial_aggregate)
        # end of first pass: local_rank 0 has cluster aggregate
        # second pass: propagate cluster aggregate (rather than routing tree - simpler implementation)
        if local_rank != 0:
            partial_aggregate = comm.recv()
            self_model.set_weights(partial_aggregate)
        comm.send(copy.deepcopy(partial_aggregate),
                  dest=cluster_ranks[clust][(local_rank + 1) % len(cluster_ranks[clust])])
        if local_rank == 0:
            partial_aggregate = comm.recv()
            self_model.set_weights(partial_aggregate)
        print('PArank', rank, self_model.get_weights()[0])
        '''Cluster Aggregate Communication'''
        if rank == sink: # no fall-back routing, assume predictive routing accurate
            comm.send(obj=(rank, partial_aggregate), dest=server_rank)

# Final Evaluation of global model - first initialize all to final global model, then eval on their local dataset
if rank == server_rank:
    comm.bcast(self_model.get_weights(), root=server_rank)
else:
    global_weights = None
    global_weights = comm.bcast(global_weights, root=server_rank)
    self_model.set_weights(global_weights)
    print('GLrank', rank, self_model.get_weights()[0])
    self_model.evaluate(x=data[2], y=data[3], batch_size=batch_size)
