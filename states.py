from FeatureCloud.app.engine.app import AppState, Role, State, app_state, SMPCOperation
import time
import os
import re
import warnings
import traceback
import json
import sys
import itertools
import yaml
import numpy as np
import pandas as pd
import statistics
import networkx as nx
from pgmpy.utils import get_example_model

import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='pgmpy')
warnings.filterwarnings('ignore', message='.*Replacing existing CPD.*')
warnings.filterwarnings('ignore', message='.*pgmpy.*')

import functools

import client
import server
from store import store


def _record_payload_size(state, payload):
    """
    Measures `payload` (via Client.payload_size_bytes) and appends the
    result to this node's running 'payload_sizes' list, so the average
    shared-payload size across the whole workflow can be reported once it
    finishes (see EvaluationState). Call this once for every payload
    handed to send_data_to_coordinator / broadcast_data.
    """
    size_bytes = client.Client().payload_size_bytes(payload)
    payload_sizes = state.load('payload_sizes') or []
    payload_sizes.append(size_bytes)
    state.store('payload_sizes', payload_sizes)
    return size_bytes

INITIAL = 'initial'
FETCH_DATA = 'read config and dataset'
LOCAL_LEARNING = 'local learning'
AGGREGATION = 'aggregation'
AWAIT_AGGREGATION = 'await aggregation'
LOCAL_REFINEMENT = 'local refinement'
FINAL = 'final'
PARAMETER_LEARNING = 'parameter learning'
CATEGORY_LEVELS = 'gather category levels'
AWAIT_CATEGORY_LEVELS = 'await category levels'
PARAMETER_AGGREGATION = 'aggregate parameters'
AWAIT_PARAMETERS_AGGREGATION = 'await parameter aggregation'
CPT_LEARNING = 'cpt learning'
CPT_AGGREGATION = 'aggregate cpts'
AWAIT_CPT_AGGREGATION = 'await cpt aggregation'
EVALUATION = 'evaluation'
VISUALIZE = 'visualize'
TERMINAL = 'terminal'

# Dedicated memos so the finish handshake and the metric exchange never collide
# with the GATHERROUND* traffic of an ordinary round.
FINISH_SIGNAL = 'finish'
FINISH_MEMO = 'FEDPAM_FINISH'
EVAL_MEMO = 'FEDPAM_EVAL'


def park_on_error(run_method):
    """Never let a failing state tear the container down.

    FeatureCloud's engine catches anything escaping a state, marks the run ERROR
    and finishes -- killing the container and the dashboard with it. Wrapped
    states publish the traceback to the UI and divert to VISUALIZE instead.
    """
    @functools.wraps(run_method)
    def wrapper(self):
        try:
            return run_method(self)
        except Exception:
            tb = traceback.format_exc()
            self.log(f"[ERROR] state failed, diverting to {VISUALIZE}:\n{tb}")
            store.update(error=tb)
            return VISUALIZE
    return wrapper

@app_state(name = INITIAL, role = Role.BOTH)
class InitialState(AppState):
    """
    InitialState is the class is used for initializing the FedPAM workflow. 
    In this state, each client takes up the role of a participant.
    In the next state, dataset and configuration files are read.
    """
    def register(self):
        self.register_transition(target=FETCH_DATA, role=Role.BOTH)
    
    def run(self):
        self.log("Initializing FedPAM Application...")
        store.update(is_coordinator=self.is_coordinator, client_id=self.id,
                     current_state=INITIAL)
        self.log(f"Role: {'coordinator' if self.is_coordinator else 'participant'}")
        self.log("Initial State to Fetch Data State")
        return FETCH_DATA

@app_state(name = FETCH_DATA, role = Role.BOTH)
class FetchDataState(AppState):
    """
    FetchDataState is the class used for reading private dataset, configuration and expert knowledge files local storage. 
    In this state, each participant sends its dataset size and expert knowledge file to the coordinator.
    In the next state, the coordinator prepares aggregated metadata while the participants wait for the coordinator to finish. 
    """
    def register(self):
        self.register_transition(target=LOCAL_LEARNING, role=Role.BOTH)
    
    def read_config_file(self, input_dir):
        self.log("Reading config file...")
        config_file_path = os.path.join(input_dir, 'config.yml')
        
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found at {config_file_path}.")
        
        store.update(config_path=config_file_path)
        with open(config_file_path) as cfp:
            config_file = yaml.safe_load(cfp)

        configs = config_file['fc-feddypam']
        self.store('dataset_location', configs['input']['dataset_location'])
        self.store('test_dataset_location', configs['input'].get('test_dataset_location', 'test.csv'))
        self.store('has_target', configs['input']['has_target'])
        store.update(has_target=bool(configs['input']['has_target']),
                     target=configs['input'].get('target')
                     if configs['input']['has_target'] else None)
        self.store('target', configs['input']['target'])
        self.store('has_id_variable', configs['input'].get('has_id_variable', False))
        self.store('id_variable', configs['input'].get('id_variable', None))
        self.store('has_time_variable', configs['input'].get('has_time_variable', False))
        self.store('time_variable', configs['input'].get('time_variable', None))
        # Publish the id/time column names to the UI (None when not configured).
        store.update(
            id_variable=configs['input'].get('id_variable')
            if configs['input'].get('has_id_variable', False) else None,
            time_variable=configs['input'].get('time_variable')
            if configs['input'].get('has_time_variable', False) else None)
        self.store('split_mode', configs['split']['mode'])
        self.store('split_dir', configs['split']['dir'])
        
        # Load hyperparameters
        self.store('max_iterations', configs['max_iterations'])
        self.store('num_bootstrap_iterations', configs['num_bootstrap_iterations'])
        self.store('alpha', configs['alpha'])
        self.store('gamma', configs['gamma']) 
        self.store('homogeneous', configs['homogeneous']) 
        self.store('testing', configs['testing'])
        store.update(testing_enabled=bool(configs['testing']),
                     benchmark=configs.get('benchmark'))
        self.store('benchmark', configs['benchmark'])
        self.store('threshold', configs['threshold'])
        self.store('num_samples', configs['num_samples'])
        self.store('num_jobs', configs['num_jobs'])

        parameter_test = configs.get('parameter_test', False)
        if parameter_test and not configs['input']['has_target']:
            raise ValueError(
                "config.yml has 'parameter_test: true' but no target variable "
                "('input.has_target: false') — the parameter/CPT comparison "
                "needs the parameter-learning round, which only runs when a "
                "target is configured. Disable parameter_test or set has_target."
            )
        self.store('parameter_test', parameter_test)

        temporal_configs = configs.get('temporal', {})
        dbn_order = temporal_configs.get('dbn_order', 1)
        use_delay = temporal_configs.get('use_delay', False)
        target_is_current_slice = temporal_configs.get('target_is_current_slice', False)

        if not isinstance(dbn_order, int) or dbn_order < 1:
            raise ValueError(
                f"config.yml 'temporal.dbn_order' must be an integer >= 1 (got {dbn_order!r})."
            )

        self.store('dbn_order', dbn_order)
        store.update(dbn_order=dbn_order, use_delay=bool(use_delay))
        self.store('use_delay', use_delay)
        self.store('target_is_current_slice', target_is_current_slice)

        testing = configs['testing']
        if testing and use_delay:
            raise ValueError(
                "config.yml has both 'testing: true' and 'temporal.use_delay: true'. "
                "Benchmark-based testing is not supported together with delay "
                "variables — disable one of the two."
            )

        mode_label = ("higher_order_dbn" if dbn_order > 1 else "2tbn") + ("_with_delay" if use_delay else "")
        self.log(f"Temporal mode: {mode_label} (dbn_order={dbn_order}, use_delay={use_delay}).")

        splits = {}
        if self.load('split_mode') == 'directory':
            split_base_dir = os.path.join(input_dir, self.load('split_dir'))
            if os.path.exists(split_base_dir):
                splits = {f.path: None for f in os.scandir(split_base_dir) if f.is_dir()}
            else:
                splits = {input_dir: None}
        else:
            splits = {input_dir: None}

        roles = {}
        for split_path in splits.keys():
            output_path = split_path.replace('/input/', '/output/')
            os.makedirs(output_path, exist_ok=True)

        self.log("Configuration loaded successfully!")
        
        return splits, roles
    
    def apply_temporal_config(self, dataset, dataset_path):
        """
        Validates the dataset's temporal columns against config.yml's
        temporal.dbn_order / temporal.use_delay, and drops delay_* columns
        when use_delay is off (so they never leak into structure learning
        or parameter fitting for the 3 non-delay modes).
        """
        dbn_order = self.load('dbn_order')
        use_delay = self.load('use_delay')

        timepoint_re = re.compile(r"\((t|tm(\d+))\)$")
        temporal_lags = set()
        for col in dataset.columns:
            if col.startswith('delay_'):
                continue
            match = timepoint_re.search(col)
            if match:
                temporal_lags.add(int(match.group(2)) if match.group(2) else 0)

        max_detected_lag = max(temporal_lags) if temporal_lags else 0
        if max_detected_lag > dbn_order:
            raise ValueError(
                f"Dataset at {dataset_path} contains a '(tm{max_detected_lag})' column, "
                f"which needs config.yml 'temporal.dbn_order' >= {max_detected_lag} "
                f"(currently {dbn_order})."
            )

        delay_cols = [c for c in dataset.columns if c.startswith('delay_')]
        if use_delay and not delay_cols:
            raise ValueError(
                f"config.yml has 'temporal.use_delay: true' but no 'delay_*' "
                f"columns were found in the dataset at {dataset_path}."
            )
        if not use_delay and delay_cols:
            self.log(
                f"'temporal.use_delay' is false — dropping {len(delay_cols)} "
                f"delay_* column(s) found in the dataset: {delay_cols}."
            )
            dataset = dataset.drop(columns=delay_cols).reset_index(drop=True)

        return dataset

    def read_test_dataset(self, split_path, dataset_columns):
        """
        Reads this split's held-out test file (test_dataset_location, same
        folder as the training dataset_location) and applies the SAME
        id/time-column drop + temporal-config handling as the training set,
        so its columns line up exactly with what the network was fit on.
        Unlike the training dataset, it is never shuffled or subsampled —
        it's a fixed evaluation set, not something structure/PAM learning
        ever touches.
        """
        has_id_variable = self.load('has_id_variable')
        id_variable = self.load('id_variable')
        has_time_variable = self.load('has_time_variable')
        time_variable = self.load('time_variable')
        test_dataset_location = self.load('test_dataset_location')

        test_path = os.path.join(split_path, test_dataset_location)
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Test dataset file not found at location: {test_path}. "
                f"Expected a '{test_dataset_location}' file alongside the training "
                f"dataset in the same split folder."
            )

        self.log(f"Reading test dataset from {test_path}...")
        test_dataset = pd.read_csv(test_path).reset_index(drop=True)

        if has_id_variable:
            if id_variable not in test_dataset.columns:
                raise ValueError(
                    f"id_variable '{id_variable}' not found in test dataset columns at "
                    f"{test_path}: {list(test_dataset.columns)}."
                )
            test_dataset = test_dataset.drop(columns=[id_variable]).reset_index(drop=True)

        if has_time_variable:
            if time_variable not in test_dataset.columns:
                raise ValueError(
                    f"time_variable '{time_variable}' not found in test dataset columns at "
                    f"{test_path}: {list(test_dataset.columns)}."
                )
            test_dataset = test_dataset.drop(columns=[time_variable]).reset_index(drop=True)

        test_dataset = self.apply_temporal_config(test_dataset, test_path)

        missing = set(dataset_columns) - set(test_dataset.columns)
        if missing:
            raise ValueError(
                f"Test dataset at {test_path} is missing column(s) present in the "
                f"training dataset: {missing}."
            )

        self.log(
            f"Test dataset from {split_path}: {test_dataset.shape[0]} observations "
            f"and {test_dataset.shape[1]} variables."
        )
        return test_dataset

    def read_dataset(self, splits, roles):
        has_id_variable = self.load('has_id_variable')
        id_variable = self.load('id_variable')
        has_time_variable = self.load('has_time_variable')
        time_variable = self.load('time_variable')
        id_series_by_split = {}
        test_splits = {}

        for split_path in splits.keys():
            roles[split_path] = 'coordinator' if self.is_coordinator else 'client'
            dataset_location = self.load('dataset_location')
            has_target = self.load('has_target')

            dataset_path = os.path.join(split_path, dataset_location)
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found at location: {dataset_path}.")
            
            self.log("Reading dataset...")
            testing = self.load('testing')
            if testing:
                num_samples = self.load('num_samples')

            dataset = pd.read_csv(dataset_path)
            # randomly shuffle the dataset to prevent sorted target values
            if testing and num_samples:
                dataset = dataset.sample(n = num_samples, random_state=23).reset_index(drop=True)
            else:
                dataset = dataset.sample(frac=1, random_state=23).reset_index(drop=True)

            dataset = dataset.reset_index(drop=True)
            if has_id_variable:
                if not id_variable:
                    raise ValueError(
                        "has_id_variable is true in config.yml but id_variable is not set."
                    )
                if id_variable not in dataset.columns:
                    raise ValueError(
                        f"id_variable '{id_variable}' not found in dataset columns at {dataset_path}: "
                        f"{list(dataset.columns)}."
                    )
                # Unlike the time column, ID is also kept (separately, aligned by
                # row order) as a grouping key for EvaluationState's group-aware
                # CV split — it's excluded from the network but not thrown away.
                id_series_by_split[split_path] = dataset[id_variable].reset_index(drop=True)
                dataset = dataset.drop(columns=[id_variable]).reset_index(drop=True)

            if has_time_variable:
                if not time_variable:
                    raise ValueError(
                        "has_time_variable is true in config.yml but time_variable is not set."
                    )
                if time_variable not in dataset.columns:
                    raise ValueError(
                        f"time_variable '{time_variable}' not found in dataset columns at {dataset_path}: "
                        f"{list(dataset.columns)}."
                    )
                # Time is excluded from the network entirely — it's not used for
                # CV grouping (only id_variable is), just dropped as a feature.
                dataset = dataset.drop(columns=[time_variable]).reset_index(drop=True)

            dataset = self.apply_temporal_config(dataset, dataset_path)

            splits[split_path] = dataset
            test_splits[split_path] = self.read_test_dataset(split_path, dataset.columns)
            self.log(
                f"Local dataset from {split_path}: {dataset.shape[0]} observations "
                f"and {dataset.shape[1]} variables."
            )

        client_split_path = None
        client_id = str(self.id).lower()
        for split_path in splits.keys():
            split_dirname = os.path.basename(split_path).lower()
            self.log(f"Comparing client ID '{client_id}' with split directory '{split_dirname}'...")
            if client_id in split_dirname:
                client_split_path = split_path
                break

        if client_split_path is None:
            if len(splits) == 1:
                client_split_path = next(iter(splits.keys()))
                self.log(f"Using split directory: {client_split_path}.")
            else:
                raise RuntimeError(f"No matching split directory for client ID {client_id}.")

        self.store('dataset', splits[client_split_path])
        store.update(dataset=splits[client_split_path],
                     dataset_path=client_split_path,
                     dataset_csv_path=os.path.join(
                         client_split_path, self.load('dataset_location')))
        self.log(f"[viz] dataset published to UI: "
                 f"{splits[client_split_path].shape}")
        self.store('test_dataset', test_splits[client_split_path])
        self.store('client_split_path', client_split_path)
        if has_id_variable:
            self.store('id_series', id_series_by_split[client_split_path])
            self.log(
                f"ID-based evaluation enabled on '{id_variable}': "
                f"{id_series_by_split[client_split_path].nunique()} unique IDs in this client's data."
            )
        else:
            self.store('id_series', None)

        return splits, roles
    
    def run(self):
        iteration = 1
        self.store('iteration', iteration)

        input_dir = "/mnt/input"
        output_dir = "/mnt/output"
        self.store('input_dir', input_dir)
        self.store('output_dir', output_dir)
        
        splits_init, roles_init = self.read_config_file(input_dir)
        splits, roles = self.read_dataset(splits_init, roles_init)
        self.store('splits', splits)
        self.store('roles', roles)

        local_number = np.random.randint(1, 10, 1)
        self.log(f"LOCAL NUMBER: {local_number}")

        self.log("Fetch data state to local learning")
        return LOCAL_LEARNING


@app_state(name = LOCAL_LEARNING, role = Role.BOTH)
class LocalLearningState(AppState):
    def register(self):
        self.register_transition(target = AWAIT_AGGREGATION, role = Role.PARTICIPANT)
        self.register_transition(target = AGGREGATION, role = Role.COORDINATOR)

    def run(self):
        dataset = self.load('dataset')
        dataset_size = len(dataset)
        testing_flag = self.load('testing')
        num_bootstrap_iterations = self.load('num_bootstrap_iterations')
        benchmark = self.load('benchmark')
        num_jobs = self.load('num_jobs')
        has_target = self.load('has_target')

        participant = client.Client()

        if has_target:
            target = self.load('target')
            local_edge_strengths, allowed_edges = participant.create_pam(dataset = dataset, has_target = has_target, target = target, num_iterations = num_bootstrap_iterations, seed = 23, n_jobs=num_jobs)
            local_dag = participant.learn_constrained_local_dag(dataset = dataset, allowed_edges = allowed_edges, has_target = has_target, target = target)
        else:
            local_edge_strengths, allowed_edges = participant.create_pam(dataset = dataset, has_target = False, target = None, num_iterations = num_bootstrap_iterations, seed = 23, n_jobs=num_jobs)
            local_dag = participant.learn_constrained_local_dag(dataset = dataset, allowed_edges = allowed_edges)

        self.store('local_pam', local_edge_strengths)
        self.store('local_dag', local_dag)
        self.store('first_local_dag', local_dag)
        store.update(local_structure=sorted(local_dag.edges()),
                     first_local_structure=sorted(local_dag.edges()))  # never overwritten by LOCAL_REFINEMENT, unlike 'local_dag'

        if testing_flag:
            benchmark_network = get_example_model(benchmark)
            true_edges = set(benchmark_network.edges())

            # ---------------- SINGLE ITERATION ----------------
            single_iter_network = participant.learn_local_structure(
                dataset, False, None, True
            )

            single_iter_shd = participant.compute_shd(
                benchmark_network, single_iter_network
            )

            single_pred_edges = set(single_iter_network.edges())

            single_tp = len(single_pred_edges & true_edges)
            single_fp = len(single_pred_edges - true_edges)
            single_fn = len(true_edges - single_pred_edges)

            single_precision = (
                single_tp / len(single_pred_edges)
                if single_pred_edges else 0
            )
            single_recall = (
                single_tp / len(true_edges)
                if true_edges else 0
            )
            single_f1 = (
                2 * single_precision * single_recall /
                (single_precision + single_recall)
                if (single_precision + single_recall) > 0 else 0
            )

            self.log("[TESTING] --- SINGLE ITERATION ---")
            self.log(f"Number of edges: {single_iter_network.number_of_edges()}")
            self.log(f"SHD: {single_iter_shd}")
            self.log(f"TP: {single_tp}, FP: {single_fp}, FN: {single_fn}")
            self.log(f"Precision: {single_precision:.3f}")
            self.log(f"Recall (TPR): {single_recall:.3f}")
            self.log(f"F1-score: {single_f1:.3f}")

            # ---------------- BOOTSTRAP NETWORK ----------------
            bootstrap_network_shd = participant.compute_shd(
                benchmark_network, local_dag
            )

            bootstrap_pred_edges = set(local_dag.edges())

            bootstrap_tp = len(bootstrap_pred_edges & true_edges)
            bootstrap_fp = len(bootstrap_pred_edges - true_edges)
            bootstrap_fn = len(true_edges - bootstrap_pred_edges)

            bootstrap_precision = (
                bootstrap_tp / len(bootstrap_pred_edges)
                if bootstrap_pred_edges else 0
            )
            bootstrap_recall = (
                bootstrap_tp / len(true_edges)
                if true_edges else 0
            )
            bootstrap_f1 = (
                2 * bootstrap_precision * bootstrap_recall /
                (bootstrap_precision + bootstrap_recall)
                if (bootstrap_precision + bootstrap_recall) > 0 else 0
            )

            self.log("[TESTING] --- BOOTSTRAP NETWORK ---")
            self.log(f"Number of edges: {local_dag.number_of_edges()}")
            self.log(f"SHD: {bootstrap_network_shd}")
            self.log(f"TP: {bootstrap_tp}, FP: {bootstrap_fp}, FN: {bootstrap_fn}")
            self.log(f"Precision: {bootstrap_precision:.3f}")
            self.log(f"Recall (TPR): {bootstrap_recall:.3f}")
            self.log(f"F1-score: {bootstrap_f1:.3f}")

            # ---------------- COMPARISON ----------------
            self.log("[TESTING] --- COMPARISON ---")
            self.log(f"ΔSHD: {bootstrap_network_shd - single_iter_shd:+d}")
            self.log(f"ΔPrecision: {bootstrap_precision - single_precision:+.3f}")
            self.log(f"ΔRecall: {bootstrap_recall - single_recall:+.3f}")
            self.log(f"ΔF1-score: {bootstrap_f1 - single_f1:+.3f}")

        if testing_flag:
            local_payload = {
                "client_data_size": dataset_size,
                "client_pam": local_edge_strengths,
                "num_local_edges": local_dag.number_of_edges(),
                "local_shd": bootstrap_network_shd,
                "local_tpr": bootstrap_recall
            }
        else:
            local_payload = {
                "client_data_size": dataset_size,
                "client_pam": local_edge_strengths,
                "num_local_edges": local_dag.number_of_edges()
            }

        _record_payload_size(self, local_payload)
        self.send_data_to_coordinator(local_payload)

        if self.is_coordinator:
            return AGGREGATION
        else:
            return AWAIT_AGGREGATION


@app_state(name = AGGREGATION, role = Role.COORDINATOR)
class AggregationState(AppState):
    def register(self):
        self.register_transition(target = LOCAL_REFINEMENT, role = Role.COORDINATOR)
        self.register_transition(target = FINAL, role = Role.COORDINATOR)

    def run(self):
        iteration = self.load('iteration')
        self.log(f"ITERATION: {iteration}")
        testing = self.load('testing')
        max_iterations = self.load('max_iterations')
        patience = 5

        client_payloads = self.gather_data()
        client_data_sizes = [cp["client_data_size"] for cp in client_payloads]
        client_weights = [cds / sum(client_data_sizes) for cds in client_data_sizes]
        client_pams = [cp["client_pam"] for cp in client_payloads]

        if testing and iteration == 1:
            client_num_edges = [cp["num_local_edges"] for cp in client_payloads]
            client_shds = [cp["local_shd"] for cp in client_payloads]
            client_tprs = [cp["local_tpr"] for cp in client_payloads]

            mean_client_num_edges = statistics.mean(client_num_edges)
            mean_client_shd = statistics.mean(client_shds)
            mean_client_tpr = statistics.mean(client_tprs)

            std_client_num_edges = statistics.stdev(client_num_edges)
            std_client_shd = statistics.stdev(client_shds)
            std_client_tpr = statistics.stdev(client_tprs)

            self.log(f"[TESTING] Client local metrics")
            self.log(f"Local SHD: {mean_client_shd} ± {std_client_shd}")
            self.log(f"Local TPR: {mean_client_tpr} ± {std_client_tpr}")
            self.log(f"Number of local edges: {mean_client_num_edges} ± {std_client_num_edges}")

        self.log(f"[COORDINATOR]: Client weights -> {client_weights}")

        coordinator = server.Server()
        global_pam = coordinator.aggregate_pams(client_pams, client_weights)
        self.log(f"[COORDINATOR]: GLOBAL PAM: {global_pam}")

        self.store('global_pam', global_pam)

        prev_global_pam = self.load('prev_global_pam')
        patience_counter = self.load('patience_counter') or 0

        if prev_global_pam is not None and coordinator.pams_equal(global_pam, prev_global_pam):
            patience_counter += 1
        else:
            patience_counter = 0

        self.store('prev_global_pam', global_pam)
        self.store('patience_counter', patience_counter)

        self.log(f"[COORDINATOR]: Patience counter -> {patience_counter}/{patience}")

        stagnated = patience_counter >= patience
        capped = iteration >= max_iterations

        if not stagnated and not capped:
            iteration += 1
            self.store('iteration', iteration)
            message = "continue"
            coordinator_payload = {
                "message": message,
                "global_pam": global_pam
            }

            _record_payload_size(self, coordinator_payload)
            self.broadcast_data(coordinator_payload)
            return LOCAL_REFINEMENT
        else:
            if stagnated:
                self.log(f"[COORDINATOR]: Stopped — global PAM unchanged for {patience} iterations (round {iteration}).")
            else:
                self.log(f"[COORDINATOR]: Stopped at max_iterations ({iteration}) without stagnation.")
            message = "stop"
            coordinator_payload = {
                "message": message,
                "global_pam": global_pam
            }

            _record_payload_size(self, coordinator_payload)
            self.broadcast_data(coordinator_payload)
            return FINAL


@app_state(name = AWAIT_AGGREGATION, role = Role.PARTICIPANT)
class AwaitAggregationState(AppState):
    def register(self):
        self.register_transition(target = LOCAL_REFINEMENT, role = Role.PARTICIPANT)
        self.register_transition(target = FINAL, role = Role.PARTICIPANT)

    def run(self):
        coordinator_payload = self.await_data() 
        message = coordinator_payload["message"]
        global_pam = coordinator_payload["global_pam"]
        self.store('global_pam', global_pam)

        if message == "continue":
            return LOCAL_REFINEMENT
        else:
            return FINAL


@app_state(name = LOCAL_REFINEMENT, role = Role.BOTH)
class LocalRefinementState(AppState):
    def register(self):
        self.register_transition(target = AWAIT_AGGREGATION, role = Role.PARTICIPANT)
        self.register_transition(target = AGGREGATION, role = Role.COORDINATOR)
    
    def run(self):
        self.log("DO LOCAL REFINEMENT")
        dataset = self.load('dataset')
        dataset_size = len(dataset)
        local_pam_prev = self.load('local_pam')
        local_dag = self.load('local_dag')
        global_pam = self.load('global_pam')
        alpha = self.load('alpha')
        gamma = self.load('gamma')
        has_target = self.load('has_target')
        target = self.load('target') if has_target else None

        participant = client.Client()
        forbidden_edges = None
        if has_target and target:
            forbidden_edges = [(target, var) for var in dataset.columns if var != target]
        local_pam, _ = participant.refine_local_pam(
            local_pam_prev, global_pam, local_dag, alpha, gamma, forbidden_edges=forbidden_edges
        )
        allowed_edges = global_pam.keys() #flag
        local_dag = participant.learn_constrained_local_dag(
            dataset, allowed_edges, has_target=has_target, target=target
        )
        self.store('local_dag', local_dag)
        self.store('local_pam', local_pam)
        store.update(local_structure=sorted(local_dag.edges()),
                     iteration=self.load('iteration'))
        local_payload = {
            "client_data_size": dataset_size,
            "client_pam": local_pam
        }

        _record_payload_size(self, local_payload)
        self.send_data_to_coordinator(local_payload)
        if self.is_coordinator:
            return AGGREGATION
        else: 
            return AWAIT_AGGREGATION


@app_state(name = FINAL, role = Role.BOTH)
class FinalState(AppState):
    def register(self):
        self.register_transition(target = VISUALIZE, role = Role.BOTH)
        self.register_transition(target = CATEGORY_LEVELS, role = Role.COORDINATOR)
        self.register_transition(target = AWAIT_CATEGORY_LEVELS, role = Role.PARTICIPANT)

    def run(self):
        dataset = self.load('dataset')
        nodes = sorted(dataset.columns.tolist())
        global_pam = self.load('global_pam')
        testing_flag = self.load('testing')
        benchmark = self.load('benchmark')
        threshold = self.load('threshold')
        has_target = self.load('has_target')
        first_local_dag = self.load('first_local_dag')
        local_dag = self.load('local_dag')

        participant = client.Client()
        coordinator = server.Server()
        final_dag, is_valid = coordinator.finalize_dag(global_pam, nodes, threshold=threshold)

        self.log(f"[FINAL]: Final DAG edges -> {list(final_dag.edges())}")
        self.log(f"[FINAL]: Valid DAG -> {is_valid}")
        self.log(f"No. of edges: {final_dag.number_of_edges()}")

        self.store('final_dag_valid', is_valid)

        if testing_flag: 
            benchmark_network = get_example_model(benchmark)
            true_edges = [edge for edge in final_dag.edges() if edge in benchmark_network.edges()]
            final_network_shd = coordinator.compute_shd(benchmark_network, final_dag)
            self.log(f"[TESTING] FINAL NETWORK")
            self.log(f"Final network SHD: {final_network_shd}")
            self.log(f"TPR: {len(true_edges) / len(benchmark_network.edges())}")

            true_edges_local_first = [edge for edge in first_local_dag.edges() if edge in benchmark_network.edges()]
            first_local_network_shd = participant.compute_shd(benchmark_network, first_local_dag)
            self.log(f"[TESTING] FIRST LOCAL NETWORK")
            self.log(f"First Local network SHD: {first_local_network_shd}")
            self.log(f"TPR: {len(true_edges_local_first) / len(benchmark_network.edges())}")
            
            true_edges_local = [edge for edge in local_dag.edges() if edge in benchmark_network.edges()]
            local_network_shd = participant.compute_shd(benchmark_network, local_dag)
            self.log(f"[TESTING] FINAL LOCAL NETWORK")
            self.log(f"Local network SHD: {local_network_shd}")
            self.log(f"TPR: {len(true_edges_local) / len(benchmark_network.edges())}")

        use_delay = self.load('use_delay')
        if use_delay:
            delay_columns = [c for c in dataset.columns if c.startswith('delay_')]
            target_variable = self.load('target') if has_target else None
            target_is_current_slice = self.load('target_is_current_slice')
            final_dag = participant.add_delay_edges(
                final_dag,
                delay_columns,
                target_variable=target_variable,
                target_is_current_slice=target_is_current_slice,
            )
            self.log(
                f"[FINAL]: Added delay-parent edges for irregular time series "
                f"-> DAG now has {final_dag.number_of_edges()} edges."
            )

        self.store('final_dag', final_dag)
        store.update(global_structure=sorted(final_dag.edges()))

        if has_target:
            participant = client.Client()
            local_levels = participant.get_local_category_levels(dataset)
            _record_payload_size(self, local_levels)
            self.send_data_to_coordinator(local_levels)

            if self.is_coordinator:
                return CATEGORY_LEVELS
            else:
                return AWAIT_CATEGORY_LEVELS
        else:
            self.log("No target column: skipping parameter learning, "
                     "going to the waiting state.")
            return VISUALIZE


@app_state(name = CATEGORY_LEVELS, role = Role.COORDINATOR)
class CategoryLevelsAggregationState(AppState):
    def register(self):
        self.register_transition(target = PARAMETER_LEARNING, role = Role.COORDINATOR)

    def run(self):
        client_levels_list = self.gather_data()
        coordinator = server.Server()
        category_levels = coordinator.merge_category_levels(client_levels_list)

        self.store('category_levels', category_levels)
        _record_payload_size(self, category_levels)
        self.broadcast_data(category_levels)
        self.log(f"Merged global category levels for {len(category_levels)} columns.")
        return PARAMETER_LEARNING


@app_state(name = AWAIT_CATEGORY_LEVELS, role = Role.PARTICIPANT)
class AwaitCategoryLevelsState(AppState):
    def register(self):
        self.register_transition(target = PARAMETER_LEARNING, role = Role.PARTICIPANT)

    def run(self):
        category_levels = self.await_data()
        self.store('category_levels', category_levels)
        self.log(f"Received global category levels for {len(category_levels)} columns.")
        return PARAMETER_LEARNING


@app_state(name = PARAMETER_LEARNING, role = Role.BOTH)
class ParameterLearningState(AppState):
    def register(self):
        self.register_transition(target = PARAMETER_AGGREGATION, role = Role.COORDINATOR)
        self.register_transition(target = AWAIT_PARAMETERS_AGGREGATION, role = Role.PARTICIPANT)

    def run(self):
        train_dataset = self.load('dataset')
        final_dag = self.load('final_dag')
        category_levels = self.load('category_levels')
        if not category_levels:
            raise RuntimeError(
                "category_levels is missing at PARAMETER_LEARNING — this client "
                "is likely running a stale build that predates the CATEGORY_LEVELS "
                "round. Rebuild/redeploy this client's image before retrying."
            )
        self.log(f"Using {len(category_levels)} globally-agreed column category sets.")
        train_dataset = train_dataset.astype('str').astype('category')

        participant = client.Client()
        final_dag_edges = participant.create_network_dict(final_dag.edges(), train_dataset.columns)
        local_params = participant.compute_beta_params_fixed(train_dataset, final_dag_edges, category_levels)

        node_order = participant.get_node_order(final_dag, train_dataset.columns)
        flat_vector, positions = participant.flatten_betas(local_params, node_order)

        self.log(f"Fitted local params for {len(local_params)} nodes.")
        self.store('local_params', local_params)
        self.store('node_order', node_order)
        self.store('local_positions', positions)

        client_data_size = len(train_dataset)
        smpc_payload = participant.pack_flat_vector_for_smpc(flat_vector, client_data_size)

        _record_payload_size(self, smpc_payload)
        self.configure_smpc(operation=SMPCOperation.ADD)
        self.send_data_to_coordinator(smpc_payload, use_smpc=True)

        if self.is_coordinator:
            return PARAMETER_AGGREGATION
        else:
            return AWAIT_PARAMETERS_AGGREGATION


@app_state(name = PARAMETER_AGGREGATION, role = Role.COORDINATOR)
class ParameterAggregationState(AppState):
    def register(self):
        self.register_transition(target = EVALUATION, role = Role.COORDINATOR)
        self.register_transition(target = CPT_LEARNING, role = Role.COORDINATOR)

    def run(self):
        self.log("Parameter Aggregation")
        coordinator = server.Server()

        # aggregate_data(use_smpc=True) waits for every client's SMPC-packed
        # payload (sent in ParameterLearningState) and returns the SUM the
        # FeatureCloud controller computed over them -- the coordinator
        # never receives an individual client's beta vector directly.
        smpc_aggregate = self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=True)
        global_flat_vector = coordinator.unpack_smpc_aggregate(smpc_aggregate)
        self.log("Aggregated beta parameters via SMPC secure aggregation.")

        node_order = self.load('node_order')
        positions = self.load('local_positions')

        broadcast_payload = {
            "global_flat_vector": global_flat_vector,
            "node_order": node_order,
            "positions": positions,
        }
        _record_payload_size(self, broadcast_payload)
        self.broadcast_data(broadcast_payload)

        global_params = coordinator.unflatten_betas(global_flat_vector, node_order, positions)
        self.store('global_params', global_params)
        store.update(global_params=global_params,
                     category_levels=self.load('category_levels'),
                     columns=list(self.load('dataset').columns))
        self.log(f"Aggregated global params for {len(global_params)} nodes.")

        if self.load('parameter_test'):
            return CPT_LEARNING
        else:
            return EVALUATION


@app_state(name = AWAIT_PARAMETERS_AGGREGATION, role = Role.PARTICIPANT)
class AwaitParametersAggregationState(AppState):
    def register(self):
        self.register_transition(target = EVALUATION, role = Role.PARTICIPANT)
        self.register_transition(target = CPT_LEARNING, role = Role.PARTICIPANT)

    def run(self):
        self.log("Await Parameter Aggregation")
        broadcast_payload = self.await_data()
        global_flat_vector = broadcast_payload["global_flat_vector"]
        node_order = broadcast_payload["node_order"]
        positions = broadcast_payload["positions"]

        participant = client.Client()
        global_params = participant.unflatten_betas(global_flat_vector, node_order, positions)

        self.store('global_params', global_params)
        store.update(global_params=global_params,
                     category_levels=self.load('category_levels'),
                     columns=list(self.load('dataset').columns))
        self.log(f"Received global params for {len(global_params)} nodes.")

        if self.load('parameter_test'):
            return CPT_LEARNING
        else:
            return EVALUATION


@app_state(name = CPT_LEARNING, role = Role.BOTH)
class CPTLearningState(AppState):
    """
    Only runs when config.yml sets 'parameter_test: true'. Each client
    fits classic tabular CPTs directly from its own local data (same final
    global DAG structure + same global category_levels as the multi-logit
    round), then contributes them to a secure-aggregated GLOBAL CPT model
    the same way ParameterLearningState contributes to the global beta
    model -- via SMPC ADD, so the coordinator never sees any individual
    client's CPT values. See EvaluationState for the actual size/performance
    comparison against the multi-logit representation.
    """
    def register(self):
        self.register_transition(target = CPT_AGGREGATION, role = Role.COORDINATOR)
        self.register_transition(target = AWAIT_CPT_AGGREGATION, role = Role.PARTICIPANT)

    def run(self):
        self.log("[PARAMETER_TEST] Fitting local CPTs (global DAG structure) for secure CPT aggregation")

        train_dataset = self.load('dataset').astype('str').astype('category')
        final_dag = self.load('final_dag')
        category_levels = self.load('category_levels')
        node_order = self.load('node_order')

        participant = client.Client()
        final_dag_edges = participant.create_network_dict(final_dag.edges(), train_dataset.columns)
        cpt_model = participant.build_bayesian_network_from_data(train_dataset, final_dag_edges, category_levels)
        cpds = {node: cpt_model.get_cpds(node) for node in node_order}
        flat_cpt_vector, cpt_positions = participant.flatten_cpds(cpds, node_order)

        self.store('cpt_positions', cpt_positions)

        client_data_size = len(train_dataset)
        smpc_payload = participant.pack_flat_vector_for_smpc(flat_cpt_vector, client_data_size)

        _record_payload_size(self, smpc_payload)
        self.configure_smpc(operation=SMPCOperation.ADD)
        self.send_data_to_coordinator(smpc_payload, use_smpc=True)

        if self.is_coordinator:
            return CPT_AGGREGATION
        else:
            return AWAIT_CPT_AGGREGATION


@app_state(name = CPT_AGGREGATION, role = Role.COORDINATOR)
class CPTAggregationState(AppState):
    def register(self):
        self.register_transition(target = EVALUATION, role = Role.COORDINATOR)

    def run(self):
        self.log("[PARAMETER_TEST] Aggregating CPTs via SMPC secure aggregation")
        coordinator = server.Server()

        smpc_aggregate = self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=True)
        global_cpt_flat_vector = coordinator.unpack_smpc_aggregate(smpc_aggregate)

        node_order = self.load('node_order')
        cpt_positions = self.load('cpt_positions')

        broadcast_payload = {
            "global_cpt_flat_vector": global_cpt_flat_vector,
            "cpt_positions": cpt_positions,
        }
        _record_payload_size(self, broadcast_payload)
        self.broadcast_data(broadcast_payload)

        global_cpds = coordinator.unflatten_cpds(global_cpt_flat_vector, node_order, cpt_positions)
        final_dag = self.load('final_dag')
        final_dag_edges = coordinator.create_network_dict(final_dag.edges(), node_order)
        global_cpt_model = coordinator.build_bayesian_network_from_cpds(final_dag_edges, global_cpds)

        self.store('global_cpt_model', global_cpt_model)
        self.log("[PARAMETER_TEST] Aggregated global CPT model via SMPC.")
        return EVALUATION


@app_state(name = AWAIT_CPT_AGGREGATION, role = Role.PARTICIPANT)
class AwaitCPTAggregationState(AppState):
    def register(self):
        self.register_transition(target = EVALUATION, role = Role.PARTICIPANT)

    def run(self):
        self.log("[PARAMETER_TEST] Await CPT Aggregation")
        broadcast_payload = self.await_data()
        global_cpt_flat_vector = broadcast_payload["global_cpt_flat_vector"]
        cpt_positions = broadcast_payload["cpt_positions"]

        participant = client.Client()
        node_order = self.load('node_order')
        global_cpds = participant.unflatten_cpds(global_cpt_flat_vector, node_order, cpt_positions)

        final_dag = self.load('final_dag')
        final_dag_edges = participant.create_network_dict(final_dag.edges(), node_order)
        global_cpt_model = participant.build_bayesian_network_from_cpds(final_dag_edges, global_cpds)

        self.store('global_cpt_model', global_cpt_model)
        self.log("[PARAMETER_TEST] Received aggregated global CPT model.")
        return EVALUATION


@app_state(name = EVALUATION, role = Role.BOTH)
class EvaluationState(AppState):
    def register(self):
        self.register_transition(target = VISUALIZE, role = Role.BOTH)

    @park_on_error
    def run(self):
        self.log("Evaluation")
        store.update(current_state=EVALUATION)

        target = self.load('target')
        category_levels = self.load('category_levels')
        train_dataset = self.load('dataset').astype('str').astype('category')
        test_dataset = self.load('test_dataset').astype('str').astype('category')

        participant = client.Client()
        target_classes = category_levels[target]
        y_test = pd.Categorical(test_dataset[target], categories=target_classes).codes

        self.log(
            f"[EVALUATION] Fitting on {len(train_dataset)} training rows, "
            f"scoring on {len(test_dataset)} held-out test rows (test.csv)."
        )

        def evaluate(dag, label):
            """
            Fits params on the FULL local training set (train_dataset) for
            the given DAG structure, then scores on the held-out
            test_dataset — a real train/test split, no CV, no refitting per
            fold, and (crucially) test_dataset was never touched by fitting.
            """
            edges = participant.create_network_dict(dag.edges(), train_dataset.columns)
            params = participant.compute_beta_params_fixed(train_dataset, edges, category_levels)
            y_prob = participant.predict_node_probability_from_beta(test_dataset, target, params)
            metrics = participant.evaluate_predictions(y_test, y_prob)

            self.log(f"[EVALUATION] {label} (held-out test.csv):")
            for metric_name, value in metrics.items():
                self.log(f"  {metric_name}: {value:.4f}")

            return metrics

        def evaluate_full_evidence(dag, label):
            """
            Same regression params as evaluate(), but prediction runs
            through pgmpy VariableElimination with evidence_scope="full" --
            conditioning on every observed non-target column (including the
            target's children), not just its direct parents.
            """
            edges = participant.create_network_dict(dag.edges(), train_dataset.columns)
            params = participant.compute_beta_params_fixed(train_dataset, edges, category_levels)
            model = participant.build_bayesian_network_from_beta(edges, params, category_levels)
            y_prob = participant.predict_target_probability_via_inference(
                model, test_dataset, target, target_classes, evidence_scope="full"
            )
            metrics = participant.evaluate_predictions(y_test, y_prob)

            self.log(f"[EVALUATION] {label} (held-out test.csv):")
            for metric_name, value in metrics.items():
                self.log(f"  {metric_name}: {value:.4f}")

            return metrics

        def evaluate_fixed_params(params, label):
            """
            For params that are already fixed going into EvaluationState
            (e.g. the federated-aggregated global_params, computed once by
            Server.aggregate_betas upstream on train_dataset -- NOT this
            test_dataset). No refitting happens; score directly on the
            held-out test set. Since global_params was fit only on
            train_dataset (this client's contribution) and other clients'
            train_dataset, and test_dataset is a separate file that was
            never sent anywhere, this is leakage-free.
            """
            y_prob = participant.predict_node_probability_from_beta(test_dataset, target, params)
            metrics = participant.evaluate_predictions(y_test, y_prob)

            self.log(f"[EVALUATION] {label} (held-out test.csv, fixed params, no refit):")
            for metric_name, value in metrics.items():
                self.log(f"  {metric_name}: {value:.4f}")

            return metrics

        first_local_dag = self.load('first_local_dag')
        self.log(f"FIRST LOCAL DAG: {first_local_dag.edges()}")
        published = {}

        def publish(ui_label, metrics):
            """Hand one block to the dashboard as soon as it is ready.

            The UI expects {mean, std}; a held-out test set yields one score per
            metric rather than a spread across folds, so std stays empty.
            """
            if metrics:
                published[ui_label] = {"mean": dict(metrics), "std": {}, "folds": []}
                store.update(evaluation=dict(published))
            return metrics

        first_local_metrics = publish(
            "(1) Initial local network + local params",
            evaluate(first_local_dag,
                     "(1) Initial local network (bootstrap DAG) + local params"))

        last_local_dag = self.load('local_dag')
        self.log(f"FINAL LOCAL DAG: {last_local_dag.edges()}")
        last_local_metrics = publish(
            "(2) Final local network + local params",
            evaluate(last_local_dag,
                     "(2) Final local network (refined local DAG) + local params"))

        final_dag = self.load('final_dag')
        self.log(f"FINAL GLOBAL DAG: {final_dag.edges()}")
        final_global_metrics = publish(
            "(3) Final global network + local params",
            evaluate(final_dag,
                     "(3) Final global network (final DAG) + local params"))

        global_params = self.load('global_params')
        final_global_agg_metrics = publish(
            "(4) Final global network + aggregated global params",
            evaluate_fixed_params(
                global_params,
                "(4) Final global network (final DAG) + aggregated global params"))

        # Every client sends its own metrics upstream; only the coordinator
        # receives the set, so only its dashboard shows the cross-client view.
        try:
            self.send_data_to_coordinator(
                {"client": self.id, "evaluation": published}, memo=EVAL_MEMO)
            if self.is_coordinator:
                collected = {}
                for payload in self.gather_data(memo=EVAL_MEMO):
                    if isinstance(payload, dict) and payload.get("client"):
                        collected[payload["client"]] = payload["evaluation"]
                store.update(all_evaluations=collected)
                self.log(f"[EVALUATION] coordinator collected results from "
                         f"{len(collected)} clients.")
        except Exception:
            self.log(f"[EVALUATION] could not share results across clients:\n"
                     f"{traceback.format_exc()}")

        self.store('first_local_metrics', first_local_metrics)
        self.store('last_local_metrics', last_local_metrics)
        self.store('final_global_metrics', final_global_metrics)
        self.store('final_global_agg_metrics', final_global_agg_metrics)

        if self.load('parameter_test'):
            self.log("[PARAMETER_TEST] Comparing multi-logit parameters against CPTs fit from data")

            def metric_diff(beta_metrics, cpt_metrics):
                common_keys = sorted(set(beta_metrics) & set(cpt_metrics))
                return {k: beta_metrics[k] - cpt_metrics[k] for k in common_keys}

            # ---- LOCAL: local_dag structure, this client's own data ----
            local_dag_edges = participant.create_network_dict(last_local_dag.edges(), train_dataset.columns)
            local_node_order = participant.get_node_order(last_local_dag, train_dataset.columns)

            local_beta_params = participant.compute_beta_params_fixed(train_dataset, local_dag_edges, category_levels)
            local_beta_flat, _ = participant.flatten_betas(local_beta_params, local_node_order)
            local_beta_param_count = len(local_beta_flat)
            local_cpt_param_count = participant.cpt_parameter_count(local_dag_edges, category_levels)

            local_cpt_model = participant.build_bayesian_network_from_data(train_dataset, local_dag_edges, category_levels)
            local_cpt_y_prob = participant.predict_target_probability_via_inference(
                local_cpt_model, test_dataset, target, target_classes, evidence_scope="parents_only"
            )
            local_cpt_metrics = participant.evaluate_predictions(y_test, local_cpt_y_prob)
            local_performance_diff = metric_diff(last_local_metrics, local_cpt_metrics)

            self.log("[PARAMETER_TEST] LOCAL (refined local DAG, this client's own data):")
            self.log(f"  Multi-logit parameter count: {local_beta_param_count}")
            self.log(f"  CPT free-parameter count:    {local_cpt_param_count}")
            for metric_name, value in local_cpt_metrics.items():
                self.log(f"  CPT {metric_name}: {value:.4f}")
            for metric_name, value in local_performance_diff.items():
                self.log(f"  Performance diff (multi-logit - CPT) {metric_name}: {value:+.4f}")

            # ---- GLOBAL: final DAG, SMPC-aggregated across all clients ----
            final_dag_edges = participant.create_network_dict(final_dag.edges(), train_dataset.columns)
            global_node_order = participant.get_node_order(final_dag, train_dataset.columns)

            global_beta_flat, _ = participant.flatten_betas(global_params, global_node_order)
            global_beta_param_count = len(global_beta_flat)
            global_cpt_param_count = participant.cpt_parameter_count(final_dag_edges, category_levels)

            global_cpt_model = self.load('global_cpt_model')
            global_cpt_y_prob = participant.predict_target_probability_via_inference(
                global_cpt_model, test_dataset, target, target_classes, evidence_scope="parents_only"
            )
            global_cpt_metrics = participant.evaluate_predictions(y_test, global_cpt_y_prob)
            global_performance_diff = metric_diff(final_global_agg_metrics, global_cpt_metrics)

            self.log("[PARAMETER_TEST] GLOBAL (final DAG, SMPC-aggregated across clients):")
            self.log(f"  Multi-logit parameter count: {global_beta_param_count}")
            self.log(f"  CPT free-parameter count:    {global_cpt_param_count}")
            for metric_name, value in global_cpt_metrics.items():
                self.log(f"  CPT {metric_name}: {value:.4f}")
            for metric_name, value in global_performance_diff.items():
                self.log(f"  Performance diff (multi-logit - CPT) {metric_name}: {value:+.4f}")

            parameter_test_results = {
                "local": {
                    "beta_param_count": local_beta_param_count,
                    "cpt_param_count": local_cpt_param_count,
                    "beta_metrics": last_local_metrics,
                    "cpt_metrics": local_cpt_metrics,
                    "performance_diff": local_performance_diff,
                },
                "global": {
                    "beta_param_count": global_beta_param_count,
                    "cpt_param_count": global_cpt_param_count,
                    "beta_metrics": final_global_agg_metrics,
                    "cpt_metrics": global_cpt_metrics,
                    "performance_diff": global_performance_diff,
                },
            }
            self.store('parameter_test_results', parameter_test_results)

        payload_sizes = self.load('payload_sizes') or []
        coordinator = server.Server()
        avg_payload_size = coordinator.average_payload_size(payload_sizes)

        self.store('avg_payload_size_bytes', avg_payload_size)
        if payload_sizes:
            self.log(
                f"[WORKFLOW SUMMARY] Sent {len(payload_sizes)} payload(s) over the "
                f"network; average shared payload size: {avg_payload_size:,.1f} bytes "
                f"({avg_payload_size / 1024:,.2f} KB)."
            )
        else:
            self.log("[WORKFLOW SUMMARY] No payloads were tracked during this run.")

        return VISUALIZE


@app_state(name = VISUALIZE, role = Role.BOTH)
class VisualizeState(AppState):
    """
    Holds the container open so results stay readable in the dashboard.

    FeatureCloud tears the container down as soon as the app reaches 'terminal',
    and only then collects /mnt/output, so results are written BEFORE this state
    blocks. The coordinator alone decides when the run ends: it waits for its
    Finish button, then broadcasts a sentinel every participant waits for.
    Nothing here may raise -- an exception would reach the engine, flip the run
    to ERROR and kill the dashboard.
    """
    def register(self):
        self.register_transition(target = TERMINAL, role = Role.BOTH)

    def write_results(self):
        output_dir = '/mnt/output'
        os.makedirs(output_dir, exist_ok=True)
        metrics = {
            'first_local': self.load('first_local_metrics'),
            'last_local': self.load('last_local_metrics'),
            'final_global': self.load('final_global_metrics'),
            'final_global_aggregated': self.load('final_global_agg_metrics'),
        }
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as fh:
            json.dump(metrics, fh, indent=2, default=str)

        final_dag = self.load('final_dag')
        if final_dag is not None:
            with open(os.path.join(output_dir, 'global_structure.json'), 'w') as fh:
                json.dump([list(e) for e in final_dag.edges()], fh, indent=2)
        local_dag = self.load('local_dag')
        if local_dag is not None:
            with open(os.path.join(output_dir, 'local_structure.json'), 'w') as fh:
                json.dump([list(e) for e in local_dag.edges()], fh, indent=2)
        self.log(f"[VISUALIZE] Results written to {output_dir}")

    def wait_as_coordinator(self):
        self.log("[VISUALIZE] Results ready. Holding the workflow open until "
                 "Finish is clicked...")
        waited = 0
        while not store.finish_clicked:
            time.sleep(1)
            waited += 1
            if waited % 60 == 0:
                self.log(f"[VISUALIZE] still waiting for Finish ({waited}s)")
        self.log("[VISUALIZE] Finish clicked. Telling participants to shut down.")
        # send_to_self=False: this client is finishing anyway, and the engine
        # drains data_outgoing before the container exits.
        self.broadcast_data(FINISH_SIGNAL, send_to_self=False, memo=FINISH_MEMO)

    def wait_as_participant(self):
        self.log("[VISUALIZE] Results ready. Holding the workflow open until "
                 "the coordinator ends it...")
        while True:
            try:
                signal = self.await_data(memo=FINISH_MEMO)
            except Exception:
                self.log(f"[VISUALIZE] await_data failed, retrying in 5s:\n"
                         f"{traceback.format_exc()}")
                time.sleep(5)
                continue
            if signal == FINISH_SIGNAL:
                self.log("[VISUALIZE] Coordinator ended the workflow.")
                return
            self.log(f"[VISUALIZE] ignoring unexpected payload: {signal!r}")

    def run(self):
        role = 'coordinator' if self.is_coordinator else 'participant'
        prefix = os.getenv("PATH_PREFIX")
        store.update(current_state=VISUALIZE)
        self.log(f"[VISUALIZE] entered as {role}; PATH_PREFIX={prefix!r}; "
                 f"evaluation blocks in UI="
                 f"{len(store.evaluation) if store.evaluation else 0}")

        try:
            self.write_results()
        except Exception:
            self.log(f"[VISUALIZE] Could not write results:\n{traceback.format_exc()}")

        try:
            if self.is_coordinator:
                self.wait_as_coordinator()
            else:
                self.wait_as_participant()
        except Exception:
            self.log(f"[VISUALIZE] wait failed:\n{traceback.format_exc()}")
            store.update(error="The finish handshake failed. Results are still "
                               "readable; stop the run from the FeatureCloud UI.")
            while True:
                time.sleep(5)

        store.update(finish_signalled=True)
        self.log("[VISUALIZE] -> terminal")
        return TERMINAL