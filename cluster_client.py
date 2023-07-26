import client
import copy
import random


class ClusterClient(client.Client):

    # client_id: 'global' id across all clusters, rank: 'local' id within cluster
    def __init__(self, client_id, data, cluster, rank, scaling):
        super().__init__(client_id, data)
        self.cluster = cluster  # cluster object containing the ClusterClient: sometimes initialized with cluster_id
        # then reassigned actual cluster object

        # for simulation we assume (efficient) comm possible btw all pairs of clients within cluster (treat comm cost
        # in in-orbit links as negligible for now -- treat orbit/cluster as 'fully connected' and ignore ring
        # topology) (directly or indirectly based on cluster topology) assume source will communicate to every node
        # in cluster the source and sink ids at beginning of epoch
        self.rank = rank  # id within the cluster
        self.scaling = scaling

    def distribute_model(self):  # if source, distribute global model to all elements of cluster
        for neighbor_client in self.cluster.members:
            # setting neighbor.model = self.model BAD - makes all point in cluster to same model
            neighbor_client.model.set_weights(copy.deepcopy(self.model.get_weights()))

    def partial_aggregate(self, next):
        """aggregates the (current) local model of self with the local model of ClusterClient next - then sets next's model to that
        partial aggregate - scaling accounted for (weighting for individual client's contribution to whole (across clusters) dataset)
        """
        curr_aggr = copy.deepcopy(self.model.get_weights())
        for ind, local_weight in enumerate(next.model.get_weights()):
            curr_aggr[ind] += local_weight * self.scaling[next.id]
        next.model.set_weights(curr_aggr)

    def get_sink(self, source_cluster_id):
        """For SatDISL - if a source with respect to source_cluster: given info about location of each satellite in
        self's cluster and external source cluster and predicted training time for the epoch, calculate *rank* of sink
        node (in the given cluster) to the external source cluster (based on node predicted to be closest to the
        source_cluster at end of epoch)

        Sink_i,j calculation done with source_i,j communicating with source_j,i
        """
        '''hard-coded arbitrary values for the sake of initial sim
                if self.cluster.id == 0:
                    if source_cluster_id == 1:
                        return 3
                    if source_cluster_id == 2:
                        return 2
                if self.cluster.id == 1:
                    if source_cluster_id == 0:
                        return 0
                    if source_cluster_id == 2:
                        return 3
                if self.cluster.id == 2:
                    if source_cluster_id == 0:
                        return 3
                    if source_cluster_id == 1:
                        return 3
        '''
        '''random location for sink role (in reality based on satellite orbit intersection at end of epoch)'''
        cluster_size = len(self.cluster.members)
        return random.randint(0, cluster_size - 1)
