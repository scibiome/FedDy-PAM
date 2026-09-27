import threading

TRACKED = (
    "dataset",
    "dataset_path",
    "dataset_csv_path",
    "config_path",
    "local_structure",
    "first_local_structure",
    "global_structure",
    "evaluation",
    "all_evaluations",
    "predictions",
    "structure_metrics",
    "validation",
    "has_target",
    "target",
    "client_id",
    "global_params",
    "category_levels",
    "columns",
    "dbn_order",
    "use_delay",
    "testing_enabled",
    "benchmark",
    "current_state",
    "iteration",
    "is_coordinator",
    "finish_clicked",
    "finish_signalled",
    "error",
)


class ResultStore:
    def __init__(self):
        self.dataset = None
        self.dataset_path = None
        self.dataset_csv_path = None
        self.config_path = None
        self.local_structure = None
        self.first_local_structure = None  
        self.global_structure = None      
        self.evaluation = None            
        self.all_evaluations = None       
        self.predictions = None           
        self.predictions_model = None
        self.predictions_accuracy = None
        self.structure_metrics = None     
        self.validation = None
        self.has_target = None            
        self.target = None                
        self.client_id = None
        self.global_params = None
        self.category_levels = None
        self.columns = None
        self.dbn_order = None
        self.use_delay = None
        self.testing_enabled = None      
        self.benchmark = None             
        self.current_state = None
        self.iteration = None
        self.is_coordinator = None 
        self.finish_clicked = False
        self.finish_signalled = False
        self.error = None
        self._lock = threading.Lock()
        self._revisions = {name: 0 for name in TRACKED}

    def update(self, **kwargs):
        """Set fields under a lock, bumping the revision of each one touched.

        The state-machine thread writes while the Dash thread reads. CPython
        attribute assignment is atomic, so the lock is belt-and-braces, but it
        keeps multi-field updates from being observed half-applied.
        """
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)
                if key in self._revisions:
                    self._revisions[key] += 1

    def revision(self, *fields):
        """Combined revision of the given fields.

        The UI uses this as a cheap change token: same token means nothing the
        caller cares about has changed, so there is nothing to re-render.
        """
        with self._lock:
            return tuple(self._revisions.get(f, 0) for f in fields)


store = ResultStore()