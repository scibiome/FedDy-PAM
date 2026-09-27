import os
from threading import Thread

from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.engine.app import app
from bottle import Bottle
import pandas as pd
from visualization import app as dash_app

import states  # registers all @app_state decorators
from states import VISUALIZE
from store import store

server = Bottle()

path_prefix = os.getenv("PATH_PREFIX")
env = 'fc' if path_prefix else 'native'


def install_terminal_gate():
    """Make 'terminal' unreachable until the coordinator has ended the run.

    states.py is already written so the only route to terminal is through
    VisualizeState's finish handshake, but this closes the door at the engine
    level: whatever any state returns, and whatever gets added to the workflow
    later, the app cannot shut down while store.finish_signalled is False. Any
    attempt is logged and diverted back to the waiting state, which the engine's
    run loop then simply re-enters.

    Only VisualizeState sets finish_signalled, and only after the coordinator's
    Finish click (or, on a participant, after the coordinator's broadcast).
    """
    real_transition = app.transition

    def gated_transition(name):
        real_transition(name)
        if app.current_state.name == 'terminal' and not store.finish_signalled:
            app.log(f"[GATE] blocked transition '{name}' into terminal: the "
                    f"coordinator has not finished the workflow yet. "
                    f"Diverting to '{VISUALIZE}'.")
            app.current_state = app.states[VISUALIZE]

    app.transition = gated_transition


def install_setup_guard():
    """Ignore a repeated /setup while the workflow is already running.

    FeatureCloud's handle_setup is unconditional: it resets current_state to
    'initial' and starts another thread every time it is called. If the
    controller posts /setup twice for the same client -- which it sometimes
    does -- two state machines end up sharing one app object, stepping on each
    other's current_state. That shows up as duplicated log lines and then

        RuntimeError: transition await aggregation_await aggregation not found

    because one thread returns a transition for a state the other has already
    moved past. The first setup wins; later ones are logged and dropped.
    """
    real_setup = app.handle_setup

    def guarded_setup(client_id, coordinator, clients, coordinatorID=None):
        if app.thread is not None and app.thread.is_alive():
            app.log(f"[GATE] ignoring duplicate /setup for client {client_id}: "
                    f"the state machine is already running in state "
                    f"'{app.current_state.name if app.current_state else None}'.")
            return
        return real_setup(client_id, coordinator, clients, coordinatorID)

    app.handle_setup = guarded_setup


def start_dash():
    """Serve the Dash UI on 8050. nginx proxies container port 9001 to it."""
    from visualization import app as dash_app
    # debug must stay False here: Dash runs in a background thread and the
    # Werkzeug reloader only works from the main thread.
    dash_app.run(host="0.0.0.0", port=8050, debug=False)


def start_app():
    app.register()
    install_terminal_gate()  # after register(), so app.states is populated
    install_setup_guard()
    server.mount('/api', api_server)
    server.run(host='localhost', port=5000)


if __name__ == '__main__':
    if env == 'fc':
        print(f"Starting in fc mode (PATH_PREFIX={path_prefix})", flush=True)
        Thread(target=start_dash, daemon=True).start()
        start_app()  # blocks
    else:
        print("Starting in native mode", flush=True)
        # For local dev: run Dash directly without FeatureCloud. The state
        # machine does not run here (FetchDataState reads a hardcoded
        # /mnt/input), so seed the store from sample data to work on the UI

        sample = os.getenv("SAMPLE_DATA", "./sample/c1/data.csv")
        if os.path.exists(sample):
            store.dataset = pd.read_csv(sample)
            store.update(dataset_csv_path=os.path.abspath(sample))
            print(f"Seeded UI with {sample} {store.dataset.shape}", flush=True)
        else:
            print(f"No sample data at {sample}; dataset tab will be empty.", flush=True)

        dash_app.run(host="0.0.0.0", port=8050, debug=True)