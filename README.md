# FedDy-PAM 
FedDy-PAM is a federated framework for dynamic Bayesian network learning using Probabilistic Adjacency Matrices (PAMs).

### Datasets Used
1. **[DyNOPS Time-Series Patient Dataset](https://www.tu-braunschweig.de/psychologie/psychotherapieambulanz/forschung/dynops):** 
   * Contains 266152 samples for 2584 patients with ~40 discrete demographic (static) and questionnaire response variables (temporal) **binary classification** of full remission.
   * Split into $K={5, 10, 15}$ client datasets.

### Config File
Modify the hyperparameters in `config.yml` file based on your requirements.

```
fc-feddypam:
  input:
    dataset_location: client.csv
    test_dataset_location: test.csv
    has_target: true
    target: 'full_remission'
    has_id_variable: true
    id_variable: 'Chiffre'
    has_time_variable: true
    time_variable: 'Date'
  split:
    mode: 'file'
    dir: '.'
  max_iterations: 100
  num_bootstrap_iterations: 2
  alpha: 0.1
  gamma: 0.15
  homogeneous: true
  testing: false
  benchmark: child
  threshold: 0.5
  num_samples: null
  num_jobs: 1
  n_splits: 5
  parameter_test: true
  temporal:
    dbn_order: 1
    use_delay: true
    target_is_current_slice: false
```

#### Description of Hyperparameters:

1. `dataset_location`: Location of the csv file containing the discrete dataset. 
During <b>app testing </b>, use the following directory structure:
```
data
└───clients_datasets_directory
│   └──client1
│       │client.csv
│   └──client2
│       │client.csv
│   └──client3
│       │client.csv
```

Check the `data` directory in the fc-fedpam repository before running the app to avoid any errors related to file paths. To test the app, change `clients_datasets_directory` to your desired dataset directory.

During actual federated workflow, you will be required to upload a shared `config.yml` file and each client's `client.csv` and `test.csv` files containing the training and testing datasets, respectively.

2. `has_target`: A boolean variable to inform the model if a target variable is present in the dataset or not.

3. `target`: Set this to the target variable in the dataset if it exists. For CKD-400, use 'class' and for the students success prediction dataset, use 'Target'.

4. `has_id_variable`: A boolean variable to inform the model if an ID variable is present in the dataset or not. This variable is used to group the observation rows.

5. `id_variable`: Set this to the ID variable in the dataset if it exists. 

6. `has_time_variable`: A boolean variable to inform the model if a time variable is present in the dataset or not. This variable is used to order the observation rows.

7. `time_variable`: Set this to the time variable in the dataset if it exists. 

8. `mode`: Controls how the app finds data splits. If set to `mode: 'directory'`, the app looks for subdirectories inside a base folder to use as separate client data splits. Otherwise, it uses the main `/mnt/input` directory as the single split. During testing, you can change client data directories using the FeatureCloud test-bed/workflow interface.

9. `dir`: The base directory (relative to `/mnt/input`) that contains subdirectories for each client's data split. 

10. `alpha`: Hyperparameter $\alpha$ for tuning the dominance of local PAM $P_k$ over global PAM $P_{global}$, to mitigate local drifts due to statistical heterogeneity. Results show that for homogeneous settings, keeping $\alpha=0.5$ gives the best results as both local and global PAM are created from statistically similar data. However, using lower values like 0.1 or 0.2 is preferred for heterogeneous cases.

11. `gamma`: Hyperparameter $\gamma$ for controlling the speed of convergence. Essentially, this hyperparameter acts as a weight for local DAG in each iteration, ensuring that while the algorithm learns from global knowledge, the local evidence is also preserved. Based on experiments, values like 0.1 and 0.15 show faster convergence to low SHD values for a variety of structural complexities.

12. `num_bootstrap_iterations`: Total number of bootstrap iterations. By default, this number $B$ is set to 100 for higher variability and more structural exploration.

13. `max_iterations`: Total number of federated learning rounds.

14. `homogeneous`: Boolean hyperparameter to switch between homogeneous and heterogeneous learning modes. If the existing client data is "known" to be homogeneous, set `homogeneous: true`. Otherwise, set `homogeneous: False`. In fact, in real-world scenarios, keeping the latter is suggested as the client distributions are usually unknown.

15. `testing`: To evaluate the algorithm on BN benchmarks, set this parameter to `true`. Otherwise, `false`.

16. `benchmark`: Name of the BN benchmark used for structure-only evaluation.

17. `threshold`: Across FL rounds, the local refinement followed by server-side aggregation pushes pushes the PAM probabilities towards either 0 or 1. Therefore, a `threshold` value of 0.5 acts as a suitable measure to cluster the PAM elements into two groups - significant edges and insignificant edges. As a result, the PAM is binzarized and only the significant edges are included in the final global network.

18. `num_samples`: If all clients need to have the same number of samples, use this parameter to control the common sample size. Otherwise, keep it to the default value `null` to ensure variability in client sample size.

19. `num_jobs`: The algorithm supports parallelization of bootstrapping. Use this parameter to allocate $c$ CPU cores for running each Hill Climbing algorithm. During testing, keep in mind that for $K$ clients, $K \times c$ CPU cores will be used in total.

20. `parameter_test`: A boolean variable to decide if multi-logit regression parameters need to be tested against CPT-based parameters. If set to `true`, the global network will be evaluated using both methods and an analysis will be presented to compare the efficiency and computational complexities of both methods.

21. `dbn_order`: Decides the order of the DBN. If set to 1, a 2TBN representation is used for visualizing the network. For higher orders, a visualization similar to static BNs is used but each node has its time slice associated with it.

22. `use_delay`: For irregular time-series data, set this boolean variable to  `true` for including the delay variable in the network. This variable is only used during parameter estimation and to avoid visual clutter, it won't be added to the visualized network.

23. `target_is_current_slice`: If the target variable is a temporal variable, set this to true. Otherwise, if the target is a static variable that does not evolve over time, keep it to false.

### Steps to run FedDy-PAM application:
1. Install [Docker](https://docs.docker.com/desktop/setup/install/windows-install) and pip package `featurecloud`:

```
pip install featurecloud
```

2. Download the FedPAM image from FeatureCloud Docker repository using

```
featurecloud app download featurecloud.ai/feddypam
```

3. OR build the app locally using:

```
featurecloud app build featurecloud.ai/feddypam
```

## User Interface
The FedDy-PAM app provides an interactive user interface to monitor local and global structures during the workflow, followed by an analyses of both structural and evaluation results. Moreover, it allows the user to modify the learned global DAG interactively and store the results in an expert-knowledge JSON file.
To run the interface, use FeatureCloud's dedicated UI button.

IMPORTANT: The workflow will run and finish as intended but will only be terminated when the coordinator clicks on the `Finish` button (top-right corner on coordinator's app UI page) or the `Stop` button in the Featurecloud workflow UI.


## Testing FedDy-PAM Locally
To test FedPAM on locally stored datasets and simulate the federated learning workflow, you can use the [FeatureCloud test-bed](https://featurecloud.ai/development/test) or [FeatureCloud Workflow](https://featurecloud.ai/projects). You can also use CLI to run the app:

```
featurecloud test start --app-image featurecloud.ai/feddypam --client-dirs './dynops/clients_05/client_1,./dynops/
clients_03/client_2,./dynops/clients_03/client_3' --generic-dir './generic'
```

<b>Important</b>: Keep the shared `config.yml` file in the `generic` directory.

The results of tests will be stored in `feddy-pam/data/tests`.
