import utils


class Cluster:  # cluster of clients (ex satellites within given orbit)
    """Cluster as used in SatDISL:
       members: list of client objects in cluster

       source: dictionary: maps external cluster
       id to client *rank* that is source node with respect to it in the given cluster (ie clusters[i].source[j] returns
       source_i,j - the cluster rank of the node responsible for distributing the cluster aggregate from cluster j to its own cluster i)

       sink: dictionary: maps external cluster id to client *rank* that is sink node with respect to it in the given
       cluster (ie clusters[i].sink[j] returns sink_i,j - the cluster rank of the node responsible for sharing its cluster i aggregate with the
       corresponding sink node in cluster j)

       id: 0-indexed
       """
    def __init__(self, members, source, sink, id):
        self.members = members  # members: list of client objects in cluster
        # for simulation define arbitrary source and sink cluster objects
        self.source = source
        self.sink = sink
        self.id = id
        # represents the common 'base' model shared by all elements of the cluster at the end of the partial aggregation phase of the previous epoch
        self.base_model = utils.create_model() # in real-life will be stored on a single satellite per cluster with greatest memory
