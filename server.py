import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import expm
from scipy.special import logsumexp
import torch
import torch.optim as optim

from pgmpy.estimators import HillClimbSearch, ExpertKnowledge, BayesianEstimator, BIC, MaximumLikelihoodEstimator, PC, GES
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import os
import json
import math
from collections import Counter
from tqdm import tqdm
from typing import List, Dict, Tuple

import logging
import warnings
from tqdm import tqdm
from functools import partialmethod

logging.getLogger("pgmpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

from client import Client

class Server(Client):
    def aggregate_pams(self,
        client_edge_strengths: List[Dict[Tuple, float]],
        client_weights: List):
        """
        Performs weighted averaging of client PAMs (as returned by create_pam /
        refine_local_pam), where weight is the normalized dataset size of each
        client. Returns a sparse {(u, v): strength} dict.
        """
        if not client_edge_strengths:
            return {}

        all_edges = set()
        for edge_strengths in client_edge_strengths:
            all_edges.update(edge_strengths.keys())

        global_pam = {}
        for edge in all_edges:
            value = sum(
                edge_strengths.get(edge, 0.0) * weight
                for edge_strengths, weight in zip(client_edge_strengths, client_weights)
            )
            value = round(value, 2)
            if value > 0.0:
                global_pam[edge] = value

        return global_pam

    def aggregate_betas(self,
        client_flat_vectors: List,
        client_weights: List):
        """
        Weighted average of clients' flattened beta parameter vectors (as
        returned by Client.flatten_betas), where weight is the normalized
        dataset size of each client — same pattern as aggregate_pams.
        All vectors must be the same length and follow the same node_order
        (guaranteed since every client flattens against the one final
        global DAG).

        Returns the aggregated flat vector; pass it into
        Client.unflatten_betas(agg_vector, node_order, positions) to get
        back the global network's parameter dict for evaluation.
        """
        if not client_flat_vectors:
            return np.array([])

        lengths = {len(v) for v in client_flat_vectors}
        if len(lengths) > 1:
            raise ValueError(
                f"client_flat_vectors have mismatched lengths {lengths} — "
                "every client must flatten against the same node_order/positions."
            )

        weights = np.asarray(client_weights, dtype=float)
        if np.sum(weights) == 0:
            raise ValueError("Sum of client_weights is 0")
        weights = weights / np.sum(weights)

        stacked = np.stack([np.asarray(v, dtype=float) for v in client_flat_vectors], axis=0)
        aggregated = np.tensordot(weights, stacked, axes=(0, 0))

        return aggregated

    def merge_category_levels(self, client_levels_list: List):
        """
        Union, per column, of every client's locally observed category
        values (as returned by Client.get_local_category_levels). This
        becomes the GLOBAL schema every client then fits/flattens against,
        so aggregate_betas always receives same-length vectors.
        """
        merged = {}
        for local_levels in client_levels_list:
            for col, values in local_levels.items():
                merged.setdefault(col, set()).update(values)
        return {col: sorted(values) for col, values in merged.items()}

    def average_payload_size(self, payload_sizes: List) -> float:
        """
        Mean of the tracked per-message sizes (bytes) accumulated over the
        course of the workflow — one entry per send_data_to_coordinator /
        broadcast_data call, as recorded via Client.payload_size_bytes.
        Returns 0.0 if nothing was tracked (e.g. a run that never sent
        anything, or has_target=False skipping the parameter rounds).
        """
        if not payload_sizes:
            return 0.0
        return sum(payload_sizes) / len(payload_sizes)

    def unpack_smpc_aggregate(self, aggregated_vector: np.ndarray) -> np.ndarray:
        """
        Reverses Client.pack_flat_vector_for_smpc after FeatureCloud's
        controller has SMPC-summed every client's packed
        [weight * flat_vector, weight] vector (via
        self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=True)).

        The summed vector is [sum(weight_i * flat_vector_i), sum(weight_i)];
        dividing the first part by the last element recovers the same
        size-weighted average beta vector that aggregate_betas computes
        from plaintext data — but here the coordinator only ever saw the
        aggregated sums, never an individual client's vector or weight.
        """
        aggregated_vector = np.asarray(aggregated_vector, dtype=float)
        if aggregated_vector.size < 2:
            raise ValueError(
                "SMPC-aggregated vector is too short to contain a weight "
                "component — expected [weighted_betas..., total_weight]."
            )

        total_weight = aggregated_vector[-1]
        if total_weight == 0:
            raise ValueError("SMPC-aggregated total weight is 0 — cannot compute weighted average.")

        return aggregated_vector[:-1] / total_weight

    def is_converged(self, global_pam, tol=1e-6):
        """
        Returns True if every entry in the global PAM is within tol of 1.0.
        Entries that rounded/aggregated down to 0.0 are already absent from
        the sparse global_pam dict, so this only needs to check the upper bound.
        """
        return all(math.isclose(value, 1.0, abs_tol=tol) for value in global_pam.values())

    def pams_equal(self, pam_a, pam_b, tol=1e-9):
        if set(pam_a.keys()) != set(pam_b.keys()):
            return False
        return all(math.isclose(pam_a[edge], pam_b[edge], abs_tol=tol) for edge in pam_a)

    def remove_cycles_by_weakest_edge(self, global_pam, nodes):
        """
        Builds a directed graph from global_pam and repeatedly breaks any cycle
        by removing its lowest-strength edge, until the graph is acyclic.
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        for (u, v), strength in global_pam.items():
            graph.add_edge(u, v, weight=strength)

        while True:
            try:
                cycle = nx.find_cycle(graph, orientation='original')
            except nx.NetworkXNoCycle:
                break

            cycle_edges = [(u, v) for u, v, _ in cycle]
            weakest_edge = min(cycle_edges, key=lambda e: graph.edges[e]['weight'])
            self.log_removed_edge = weakest_edge
            graph.remove_edge(*weakest_edge)

        return graph

    def binarize_dag(self, graph, threshold):
        """
        Keeps only edges with weight >= threshold. threshold=2/3 means an edge
        must be supported by at least two thirds of total client weight
        (e.g. 2 of 3 equally-weighted clients).
        """
        binary_dag = nx.DiGraph()
        binary_dag.add_nodes_from(graph.nodes())
        for u, v, data in graph.edges(data=True):
            if data['weight'] >= threshold:
                binary_dag.add_edge(u, v)
        return binary_dag

    def finalize_dag(self, global_pam, nodes, threshold=2/3):
        """
        Full finalization pipeline: break cycles using continuous edge strengths,
        then binarize at threshold, then verify the result is a valid DAG.
        Returns (final_dag, is_valid).
        """
        acyclic_graph = self.remove_cycles_by_weakest_edge(global_pam, nodes)
        final_dag = self.binarize_dag(acyclic_graph, threshold)
        is_valid = nx.is_directed_acyclic_graph(final_dag)
        return final_dag, is_valid