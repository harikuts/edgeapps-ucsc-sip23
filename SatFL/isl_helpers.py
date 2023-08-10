import random


def get_sink(cluster_ranks, locations=[], times=[], train_iter_time=0):
    """Redefine for realistic simulations - for now, randomized sinking
    returns absolute (across all processes) rank of calculated sink node

    (ranks are *non-local* (absolute across all processes) in the context of parallelized fed_isl, fed_disl) Use
    orbital mechanics, initial locations of each satellite in cluster & current time, and fedavg iteration time of cluster (change in
    time), to predict satellite with the greatest period of connectivity to the parameter server
    """
    cluster_size = len(cluster_ranks)
    return cluster_ranks[random.randint(0, cluster_size - 1)]
