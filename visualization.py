import json
import os
import re
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_cytoscape as cyto
from dash import (ALL, Dash, dcc, html, Input, Output, State, callback,
                  callback_context, dash_table, no_update)
from dash.exceptions import PreventUpdate

from store import store

cyto.load_extra_layouts()

# FeatureCloud serves the app UI behind a path prefix (e.g. /app-traffic/<id>/).
# The controller strips it before proxying to port 9001, so the Flask routes
# stay at "/" -- only the URLs the browser requests need the prefix.
PATH_PREFIX = os.getenv("PATH_PREFIX", "")
if PATH_PREFIX and not PATH_PREFIX.endswith("/"):
    PATH_PREFIX += "/"
REQUESTS_PREFIX = PATH_PREFIX or "/"
ENV = "fc" if PATH_PREFIX else "native"

ACCENT = "#5DCAA5"
INK = "#12322a"
MUTED = "#6b7785"

# Pastel fills distinguish the two networks everywhere they appear: the local
# network is blue, the global network green. Every node carries the same thin
# black outline so the shapes stay legible against a pale fill.
LOCAL_FILL = "#CBE2F5"
GLOBAL_FILL = "#C8E9D4"
IDLE_FILL = "#ECEFF1"
# The target variable, wherever it appears in a network.
TARGET_FILL = "#F6C7C7"
NODE_BORDER = "#1f1f1f"
NODE_BORDER_WIDTH = 1

# Chart series follow the same local/global pairing.
NETWORK_COLORS = [LOCAL_FILL, GLOBAL_FILL]

# Which evaluation blocks the tab compares, keyed by the "(n)" marker that
# EvaluationState writes. Change the markers here to compare different variants:
# (1) local DAG + local params, (2) refined local DAG + local params,
# (3) global DAG + local params, (4) global DAG + aggregated global params.
EVAL_PAIR = [("(1)", "Local network"), ("(4)", "Global network")]

# Same idea for the structure-recovery blocks published by FinalState.
STRUCTURE_PAIR = [("Initial local network", "Local network"),
                  ("Final global network", "Global network")]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    requests_pathname_prefix=REQUESTS_PREFIX,
    suppress_callback_exceptions=True,
)

# Dash / Flask: log only errors. The dashboard polls the store every 2 s, and
# without this every poll would print an access-log line.
import logging
import flask.cli

logging.getLogger("werkzeug").setLevel(logging.ERROR)
app.logger.setLevel(logging.ERROR)
flask.cli.show_server_banner = lambda *args, **kwargs: None

# ---------------------------------------------------------------- metrics ---

# Short column headers for the four evaluation blocks. Keys are matched by the
# leading "(n)" marker that EvaluationState writes, so rewording the long label
# upstream will not break this.
SHORT_LABELS = dict(EVAL_PAIR)

# Metrics where a smaller number is better.
LOWER_IS_BETTER = {"LogLoss", "Brier", "shd", "fp", "fn"}

# Structure-recovery metrics, scored against the benchmark network. Available
# whether or not a target variable is configured.
STRUCTURE_METRIC_ORDER = ["shd", "f1", "precision", "recall", "auroc", "aupr",
                          "num_edges", "tp", "fp", "fn"]
STRUCTURE_METRIC_LABELS = {
    "shd": "SHD (structural Hamming distance)",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall (TPR)",
    "auroc": "AUROC",
    "aupr": "AUPR",
    "num_edges": "Edges",
    "tp": "True positives",
    "fp": "False positives",
    "fn": "False negatives",
}
STRUCTURE_COUNTS = {"num_edges", "tp", "fp", "fn", "shd"}
STRUCTURE_CHART_METRICS = ["f1", "precision", "recall", "auroc", "aupr"]

# Not a performance score -- displayed, but never highlighted as "best".
NON_SCORE = {"num_edges"}

# Preferred display order; anything else is appended alphabetically.
METRIC_ORDER = ["Accuracy", "F1", "ROC_AUC", "PR_AUC", "Precision",
                "Precision (weighted)", "Recall", "LogLoss", "Brier",
                "num_edges"]

# The one chart left in the app: global DAG against the local DAGs. Only
# metrics on a shared 0-1 scale, so the bars stay comparable.
CHART_METRICS = ["Accuracy", "F1", "ROC_AUC", "PR_AUC", "Precision", "Recall"]


def short_label(label):
    for marker, short in SHORT_LABELS.items():
        if label.startswith(marker):
            return short
    return label


def field(name, default=None):
    """Read a store field that may not exist in an older store.py.

    visualization.py and store.py are separate files in the image; if one is
    updated without the other, a missing field should quietly fall back rather
    than 500 the callback and blank the dashboard.
    """
    return getattr(store, name, default)


# ---------------------------------------------------------- temporal nodes ---
# Columns carry their time slice in the name: "X(t)" is the current slice,
# "X(tm1)" one step back, "X(tmN)" N steps back. "delay_a_b" columns are the
# irregular-time-series extras and belong to no slice. This mirrors
# Client.get_timepoint_lag, so the UI reads structure exactly as the learner
# wrote it.
TIMEPOINT_RE = re.compile(r"^(.*)\((t|tm(\d+))\)$")

# Pastel fill per slice, current slice first. Anything deeper than the ramp
# reuses the last colour rather than failing.
SLICE_FILLS = ["#C8E9D4", "#CBE2F5", "#DFD6EF", "#FBE3C8", "#F2D9E6", "#D8ECE8"]
DELAY_FILL = "#ECEFF1"
STATIC_FILL = "#ffd6a5"   # same amber as static variables in the 2TBN view


def parse_node(name):
    """(base variable, lag) for a temporal column; (name, None) otherwise.

    Columns are written "phq001_(t)", so the separating underscore comes back
    with the base name; it is trimmed here to keep node labels clean.
    """
    name = str(name)
    if name.startswith("delay_"):
        return name, None
    match = TIMEPOINT_RE.match(name)
    if not match:
        return name, None
    return match.group(1).rstrip("_"), (int(match.group(3)) if match.group(3)
                                        else 0)


def slice_of(name):
    return parse_node(name)[1]


def slice_label(lag):
    if lag is None:
        return "no slice"
    return "t" if lag == 0 else f"t−{lag}"


def slice_fill(lag):
    if lag is None:
        return DELAY_FILL
    return SLICE_FILLS[min(lag, len(SLICE_FILLS) - 1)]


def is_delay(name):
    return str(name).startswith("delay_")


def without_delay(edges):
    """Drop delay columns from a structure before drawing it.

    The delay_* columns carry the irregular-sampling intervals. They stay in
    the model -- parameters, evaluation, the Querying tab -- but as nodes they
    add a fan of edges that says nothing about the variables under study, so
    the network views leave them out.
    """
    return [tuple(e) for e in (edges or [])
            if not is_delay(e[0]) and not is_delay(e[1])]


def delay_count(edges):
    return len({n for edge in (edges or []) for n in edge if is_delay(n)})


def temporal_run():
    """True when the learned structure actually carries time-slice names."""
    edges = (field("global_structure") or []) + (field("local_structure") or [])
    return any(slice_of(n) is not None for edge in edges for n in edge)


def slices_present(edges):
    lags = {slice_of(n) for edge in edges for n in edge}
    return sorted(lag for lag in lags if lag is not None)


def slice_rules():
    """Stylesheet entries colouring each node by its time slice.

    Listed before the target rule so an explicit target colour still wins.
    """
    rules = [{"selector": f"node.slice-{lag}",
              "style": {"background-color": SLICE_FILLS[lag]}}
             for lag in range(len(SLICE_FILLS))]
    rules.append({"selector": "node.static-var",
                  "style": {"background-color": STATIC_FILL}})
    rules.append({"selector": "node.delay-node",
                  "style": {"background-color": DELAY_FILL,
                            "shape": "round-rectangle"}})
    return rules


def slice_legend(edges):
    """Colour key for the slices actually present, plus delay nodes."""
    lags = slices_present(edges)
    if not lags:
        return None
    swatch = lambda colour: {"display": "inline-block", "width": "11px",
                             "height": "11px", "borderRadius": "50%",
                             "background": colour,
                             "border": f"1px solid {NODE_BORDER}",
                             "marginRight": "6px", "verticalAlign": "middle"}
    items = []
    for lag in lags:
        items.append(html.Span([html.Span(style=swatch(slice_fill(lag))),
                                f"slice {slice_label(lag)}"],
                               style={"fontSize": "12px", "color": MUTED,
                                      "marginRight": "16px"}))
    if any(slice_of(n) is None and not is_delay(n)
           for edge in edges for n in edge):
        items.append(html.Span([html.Span(style=swatch(STATIC_FILL)), "static"],
                               style={"fontSize": "12px", "color": MUTED,
                                      "marginRight": "16px"}))
    if any(str(n).startswith("delay_") for edge in edges for n in edge):
        items.append(html.Span([html.Span(style=swatch(DELAY_FILL)), "delay"],
                               style={"fontSize": "12px", "color": MUTED}))
    return html.Div(items, style={"marginBottom": "8px"})


def node_classes(node, *extra):
    """Cytoscape classes for a node, marking the target variable if it is one.

    getattr with a default rather than plain attribute access: the dashboard
    runs in the same container as a possibly older store.py, and a missing
    field must degrade to "no target highlight" rather than 500 every graph.
    """
    classes = [c for c in extra if c]
    if str(node).startswith("delay_"):
        classes.append("delay-node")
    else:
        lag = slice_of(node)
        if lag is not None:
            classes.append(f"slice-{min(lag, len(SLICE_FILLS) - 1)}")
        elif temporal_run():
            # A variable with no slice in a temporal model is static; give it
            # its own colour so it is not read as part of slice t.
            classes.append("static-var")
    target = field("target")
    if field("has_target") and target and node == target:
        classes.append("target")
    return " ".join(classes)


def target_rule():
    """Stylesheet entry for the target node. Listed last so it wins over the
    per-network fill, which is what makes the target stand out in every graph."""
    return {"selector": "node.target", "style": {
        "background-color": TARGET_FILL,
        "border-width": NODE_BORDER_WIDTH, "border-color": NODE_BORDER}}


def selected_evaluation():
    """The two predictive blocks the tab compares, in EVAL_PAIR order."""
    blocks = {label: block for label, block in (store.evaluation or {}).items()
              if block and block.get("mean")}
    chosen = {}
    for marker, _short in EVAL_PAIR:
        for label, block in blocks.items():
            if label.startswith(marker):
                chosen[label] = block
                break
    return chosen


def selected_structure_metrics():
    """The two structure-recovery blocks, relabelled for the comparison."""
    blocks = {label: metrics
              for label, metrics in (field("structure_metrics") or {}).items()
              if metrics}
    return {short: blocks[source] for source, short in STRUCTURE_PAIR
            if source in blocks}


def ordered_metrics(names):
    known = [m for m in METRIC_ORDER if m in names]
    rest = sorted(n for n in names if n not in METRIC_ORDER)
    return known + rest


def evaluation_frame(evaluation):
    """Long-form frame: one row per (model, metric) with mean and std."""
    rows = []
    for label, block in evaluation.items():
        if not block:
            continue
        mean, std = block.get("mean", {}), block.get("std", {})
        for metric, value in mean.items():
            rows.append({
                "model": short_label(label),
                "model_full": label,
                "metric": metric,
                "mean": float(value) if value is not None else np.nan,
                "std": float(std.get(metric, np.nan)),
            })
    return pd.DataFrame(rows)


def fmt(value, metric):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if metric in NON_SCORE:
        return f"{value:.0f}"
    return f"{value:.4f}"


def headline_cards(frame):
    """One card per model showing its accuracy, with the leader marked."""
    acc = frame[frame["metric"] == "Accuracy"]
    if acc.empty:
        return html.Div()
    best = acc["mean"].max()

    cards = []
    for _, row in acc.iterrows():
        is_best = row["mean"] == best
        cards.append(html.Div([
            html.Div(row["model"], style={"fontSize": "12px", "color": MUTED,
                                          "minHeight": "34px"}),
            html.Div(fmt(row["mean"], "Accuracy"),
                     style={"fontSize": "28px", "fontWeight": 600, "color": INK}),
            html.Div(f"± {fmt(row['std'], 'Accuracy')}",
                     style={"fontSize": "12px", "color": MUTED}),
        ], style={
            "flex": "1", "minWidth": "150px", "padding": "14px 16px",
            "borderRadius": "8px", "background": "#fbfcfc",
            "border": f"2px solid {ACCENT if is_best else '#e6e9ec'}",
        }))

    return html.Div([
        html.Div(cards, style={"display": "flex", "gap": "12px",
                               "flexWrap": "wrap", "marginBottom": "6px"}),
        html.Div("Mean accuracy across cross-validation folds. "
                 "The highlighted card is the leader.",
                 style={"fontSize": "12px", "color": MUTED, "marginBottom": "18px"}),
    ])


def metric_table(frame):
    """Metrics down the rows, models across the columns, best value highlighted."""
    models = list(dict.fromkeys(frame["model"]))
    metrics = ordered_metrics(frame["metric"].unique())

    pivot_mean = frame.pivot_table(index="metric", columns="model",
                                   values="mean", aggfunc="first")
    pivot_std = frame.pivot_table(index="metric", columns="model",
                                  values="std", aggfunc="first")

    rows, highlights = [], []
    for index, metric in enumerate(metrics):
        row = {"Metric": metric}
        values = {}
        for model in models:
            mean = pivot_mean.at[metric, model] if model in pivot_mean.columns else np.nan
            std = pivot_std.at[metric, model] if model in pivot_std.columns else np.nan
            values[model] = mean
            row[model] = (fmt(mean, metric) if metric in NON_SCORE
                          else f"{fmt(mean, metric)} ± {fmt(std, metric)}")
        rows.append(row)

        if metric not in NON_SCORE:
            finite = {m: v for m, v in values.items()
                      if v is not None and not np.isnan(v)}
            if finite:
                pick = min if metric in LOWER_IS_BETTER else max
                winner = pick(finite, key=finite.get)
                highlights.append({
                    "if": {"row_index": index, "column_id": winner},
                    "backgroundColor": "#eaf8f2",
                    "fontWeight": "600",
                    "color": INK,
                })

    return html.Div([
        dash_table.DataTable(
            data=rows,
            columns=[{"name": "Metric", "id": "Metric"}] +
                    [{"name": m, "id": m} for m in models],
            style_cell={"textAlign": "right", "fontSize": "13px",
                        "padding": "8px 10px", "fontFamily": "inherit"},
            style_cell_conditional=[{"if": {"column_id": "Metric"},
                                     "textAlign": "left", "fontWeight": "500"}],
            style_header={"backgroundColor": "#f6f8f9", "fontWeight": "600",
                          "borderBottom": "2px solid #e0e5e9",
                          "whiteSpace": "normal", "height": "auto"},
            style_data={"borderBottom": "1px solid #eef1f3"},
            style_data_conditional=highlights,
            style_table={"overflowX": "auto"},
        ),
        html.Div("Mean ± standard deviation across folds. Highlighted cells are "
                 "best in row; LogLoss and Brier are scored lower-is-better, and "
                 "num_edges is a count rather than a score.",
                 style={"fontSize": "12px", "color": MUTED, "marginTop": "10px"}),
    ])


def compare_stylesheet(fill):
    """Node fill differs per network; the black outline is common to both."""
    return [
        {"selector": "node", "style": {
            "label": "data(label)", "background-color": fill,
            "width": "data(size)", "height": "data(size)",
            "border-width": NODE_BORDER_WIDTH, "border-color": NODE_BORDER,
            "font-size": "10px", "text-valign": "center", "color": INK,
        }},
        {"selector": "node.connected", "style": {"background-color": fill}},
        {"selector": "edge", "style": {
            "curve-style": "bezier", "target-arrow-shape": "triangle",
            "width": 1.6}},
        # Shared edges are the agreement between the two networks; unique edges
        # are what changed on the way from the local DAG to the global one.
        {"selector": "edge.shared", "style": {
            "line-color": "#9aa5b1", "target-arrow-color": "#9aa5b1"}},
        {"selector": "edge.unique", "style": {
            "line-color": "#e08a5d", "target-arrow-color": "#e08a5d",
            "line-style": "dashed"}},
        # Same click highlighting as the structure tabs.
        *[rule for rule in structure_stylesheet(fill)
          if rule["selector"] in FLAT_HIGHLIGHT_SELECTORS],
        *slice_rules(),
        target_rule(),
    ]


FLAT_HIGHLIGHT_SELECTORS = {".faded", "node.picked", "edge.parent-edge",
                            "edge.child-edge", "node.parent-node",
                            "node.child-node"}


def comparison_elements(edges, other_edges, _all_nodes=None, picked=None):
    """Cytoscape elements for one network, edges classed shared vs unique.

    Only nodes with at least one edge in THIS network are drawn -- a variable
    isolated here is simply absent rather than floating unconnected.
    """
    shared = set(edges) & set(other_edges)
    connected = {n for edge in edges for n in edge}
    sizes = node_sizes(edges)
    elements = [{"data": {"id": n, "label": node_label(n),
                          "size": sizes.get(n, NODE_MIN_SIZE)},
                 "classes": node_classes(n, "connected")}
                for n in sorted(connected)]
    elements += [{"data": {"source": s, "target": t},
                  "classes": "shared" if (s, t) in shared else "unique"}
                 for s, t in edges]
    if picked:
        mark_neighbourhood(elements, picked)
    return elements


def mark_neighbourhood(elements, picked):
    """Add the click-highlight classes to flat elements, in place."""
    edges = [(e["data"]["source"], e["data"]["target"])
             for e in elements if "source" in e["data"]]
    parents = {s for s, t in edges if t == picked}
    children = {t for s, t in edges if s == picked}
    for element in elements:
        data = element["data"]
        extra = ""
        if "source" in data:
            if data["target"] == picked:
                extra = "parent-edge"
            elif data["source"] == picked:
                extra = "child-edge"
            else:
                extra = "faded"
        elif data["id"] == picked:
            extra = "picked"
        elif data["id"] in parents:
            extra = "parent-node"
        elif data["id"] in children:
            extra = "child-node"
        else:
            extra = "faded"
        element["classes"] = f'{element.get("classes", "")} {extra}'.strip()
    return elements


def panel_elements(edges, other_edges, picked=None):
    """Elements for one evaluation panel, in whichever representation applies.

    Edges missing from the other network are classed "unique" (orange)."""
    other = set(tuple(e) for e in other_edges)
    state = lambda learned: "shared" if tuple(learned) in other else "unique"
    if two_slice_ready(edges):
        return two_slice_elements(edges, picked, edge_state=state)
    if higher_order_ready(edges):
        return higher_order_elements(edges, picked, edge_state=state)
    return comparison_elements(edges, other_edges, None, picked)


def chip(text, color):
    return html.Span(text, style={
        "display": "inline-block", "padding": "3px 10px", "marginRight": "8px",
        "borderRadius": "12px", "fontSize": "12px", "background": color,
        "color": INK})


def two_slice_ready(edges):
    """Which representation every network view uses, from dbn_order alone.

      dbn_order == 1  ->  2TBN: slice (t) and (t+1) side by side, every
                          variable replicated in both.
      dbn_order  > 1  ->  static BN: one node per learned column, no replicas,
                          labelled with its slice ("phq001 (t-2)"), so the
                          single graph shows the higher-order dependencies.

    All views -- structure tabs, evaluation panels, validation editor, click
    handling -- call this, so they switch together. The observed slices are
    only a fallback for a run that never published dbn_order.
    """
    order = field("dbn_order")
    try:
        order = int(order) if order is not None else None
    except (TypeError, ValueError):
        order = None
    if order:
        return order == 1
    return slices_present(edges) == [0, 1]


def node_label(name):
    """Label for the static view: the variable plus the slice it belongs to.

    "phq001_(tm2)" -> "phq001 (t-2)", "phq001_(t)" -> "phq001 (t)". Static
    variables and delay columns keep their name.
    """
    base, lag = parse_node(name)
    if lag is None:
        return str(name)
    return f"{base} ({slice_label(lag)})"


def network_panel(title, subtitle, edges, other_edges, all_nodes, element_id,
                  fill):
    if not edges:
        return html.Div([
            html.H4(title, style={"color": INK, "marginBottom": "2px"}),
            html.Div(subtitle, style={"fontSize": "12px", "color": MUTED}),
            html.P("Not computed yet.", style={"color": MUTED,
                                               "marginTop": "40px"}),
        ], style={"flex": "1", "minWidth": "320px"})

    two_slice = two_slice_ready(edges)
    if two_slice:
        other = set(tuple(e) for e in other_edges)
        # Same 2TBN picture as the structure tabs; edges the other network
        # does not have turn orange so the diff still reads.
        graph = cyto.Cytoscape(
            id=element_id,
            elements=panel_elements(edges, other_edges),
            layout=TWO_SLICE_LAYOUT,
            stylesheet=two_slice_stylesheet(compare_stylesheet(fill),
                                            COMPARE_STATE_RULES_TBN),
            style={"width": "100%", "height": "560px",
                   "border": "1px solid #eef1f3", "borderRadius": "8px"},
        )
        canvas = two_slice_canvas(graph)
    elif higher_order_ready(edges):
        other = set(tuple(e) for e in other_edges)
        canvas = cyto.Cytoscape(
            id=element_id,
            elements=panel_elements(edges, other_edges),
            layout=HIGHER_ORDER_LAYOUT,
            stylesheet=two_slice_stylesheet(compare_stylesheet(fill),
                                            COMPARE_STATE_RULES_TBN),
            style={"width": "100%", "height": "560px",
                   "border": "1px solid #eef1f3", "borderRadius": "8px"},
        )
    else:
        canvas = cyto.Cytoscape(
            id=element_id,
            elements=panel_elements(edges, other_edges),
            layout={"name": "dagre", "rankDir": "LR"},
            stylesheet=compare_stylesheet(fill),
            style={"width": "100%", "height": "560px",
                   "border": "1px solid #eef1f3", "borderRadius": "8px"},
        )

    return html.Div([
        html.H4(title, style={"color": INK, "marginBottom": "2px"}),
        html.Div(subtitle, style={"fontSize": "12px", "color": MUTED,
                                  "marginBottom": "8px"}),
        canvas,
    ], style={"flex": "1", "minWidth": "320px"})


# In the 2TBN view shared edges keep their within/inter-slice colour; only the
# edges unique to one network are recoloured.
COMPARE_STATE_RULES_TBN = [
    {"selector": "edge.unique", "style": {
        "line-color": "#e08a5d", "target-arrow-color": "#e08a5d"}},
]


def structure_comparison():
    """Initial local DAG beside the final global DAG, with the diff summarised.

    This is the part of the evaluation that does not need a target variable --
    it is available as soon as the global network is aggregated, well before any
    predictive metrics exist.
    """
    initial = without_delay(field("first_local_structure"))
    final = without_delay(field("global_structure"))

    if not initial and not final:
        return html.Div([
            html.H4("Network structure", style={"color": INK}),
            html.P("Networks have not been learned yet.", style={"color": MUTED}),
        ])

    shared = set(initial) & set(final)
    added = set(final) - set(initial)
    removed = set(initial) - set(final)

    summary = html.Div([
        chip(f"{len(initial)} edges initially", "#eef1f3"),
        chip(f"{len(final)} edges finally", "#eef1f3"),
        chip(f"{len(shared)} shared", "#e8ecef"),
        chip(f"{len(added)} added by federation", "#eaf8f2"),
        chip(f"{len(removed)} dropped", "#fdf1ee"),
    ], style={"marginBottom": "6px"})

    two_slice = (two_slice_ready(initial or final)
                 or higher_order_ready(initial or final))
    legend = html.Div([
        html.Span("shared edges keep their slice colour"
                  if two_slice else "— shared edge",
                  style={"color": MUTED if two_slice else "#9aa5b1",
                         "fontSize": "12px", "marginRight": "16px"}),
        html.Span("— unique to this network" if two_slice
                  else "--- unique to this network",
                  style={"color": "#e08a5d", "fontSize": "12px"}),
    ], style={"marginBottom": "12px"})

    slices = network_legend(initial + final)

    return html.Div([
        html.H4("Network structure", style={"color": INK, "marginTop": "4px"}),
        summary,
        slices or html.Div(),
        legend,
        html.Div([
            network_panel("Local network",
                          "This client's DAG, before any federated refinement",
                          initial, final, None, "cyto-eval-initial",
                          LOCAL_FILL),
            network_panel("Global network",
                          "Aggregated DAG after federation",
                          final, initial, None, "cyto-eval-final",
                          GLOBAL_FILL),
        ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),
    ])


def data_table(frame, table_id=None, page_size=20):
    kwargs = {"id": table_id} if table_id else {}
    return dash_table.DataTable(
        data=frame.to_dict('records'),
        columns=[{"name": c, "id": c} for c in frame.columns],
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "13px", "padding": "6px",
                    "fontFamily": "inherit", "textAlign": "left"},
        style_header={"backgroundColor": "#f6f8f9", "fontWeight": "600"},
        **kwargs)


DATASET_PAGE_SIZE = 25


def paged_table(frame, table_id, page_size=DATASET_PAGE_SIZE):
    """A table that keeps its rows on the server.

    dash_table normally ships every row to the browser and pages client-side.
    At 10,000 x 25 that is a ~3.7 MB payload the browser has to parse and turn
    into component state before anything appears, which reads as a hung page
    inside the FeatureCloud frame. page_action="custom" sends one page at a
    time instead; page 0 is rendered inline so the table is populated the
    moment the expander opens, without a callback firing for a component that
    does not exist yet.
    """
    pages = max(1, -(-len(frame) // page_size))
    return dash_table.DataTable(
        id=table_id,
        data=frame.iloc[:page_size].to_dict('records'),
        columns=[{"name": c, "id": c} for c in frame.columns],
        page_action="custom", page_current=0, page_size=page_size,
        page_count=pages,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "13px", "padding": "6px",
                    "fontFamily": "inherit", "textAlign": "left"},
        style_header={"backgroundColor": "#f6f8f9", "fontWeight": "600"})


@callback(
    Output('dataset-full', 'data'),
    Input('dataset-full', 'page_current'),
    Input('dataset-full', 'page_size'),
    prevent_initial_call=True,
)
def dataset_page(page_current, page_size):
    frame = field("dataset")
    if frame is None:
        raise PreventUpdate
    start = (page_current or 0) * (page_size or DATASET_PAGE_SIZE)
    return frame.iloc[start:start + (page_size or DATASET_PAGE_SIZE)].to_dict('records')


def expander(summary_text, children, open_by_default=False):
    """Collapsed detail section. Kept as <details> so it needs no callback and
    survives the tab being re-rendered."""
    return html.Details([
        html.Summary(summary_text,
                     style={"cursor": "pointer", "fontWeight": 600,
                            "color": INK, "padding": "8px 0"}),
        html.Div(children, style={"marginTop": "8px"}),
    ], open=open_by_default,
       style={"border": "1px solid #eef1f3", "borderRadius": "8px",
              "padding": "6px 14px", "marginTop": "12px"})


def fact_row(label, value, accent=None):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": MUTED,
                               "textTransform": "uppercase",
                               "letterSpacing": "0.04em"}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": 600,
                               "color": accent or INK, "marginTop": "2px"}),
    ], style={"flex": "1", "minWidth": "150px", "padding": "14px 16px",
              "borderRadius": "8px", "background": "#fbfcfc",
              "border": "1px solid #e6e9ec"})


# Centred single column for the Dataset tab.
DATASET_PAGE = {"maxWidth": "560px", "margin": "24px auto 40px",
                "padding": "0 16px"}


def dataset_view():
    """Dataset summary: size, column list, and where the inputs came from.

    Rows are deliberately not rendered -- at this dataset's size the table
    stalls the browser. A single centred card holds everything.
    """
    frame = field("dataset")
    if frame is None:
        return html.Div([
            html.H3("Dataset", style={"color": INK, "textAlign": "center"}),
            html.P("The dataset has not been loaded yet.",
                   style={"color": MUTED, "textAlign": "center"}),
        ], style=DATASET_PAGE)

    rows, columns = frame.shape

    def stat(value, label):
        return html.Div([
            html.Div(value, style={"fontSize": "34px", "fontWeight": 600,
                                   "color": INK, "lineHeight": "1.1"}),
            html.Div(label, style={"fontSize": "13px", "color": MUTED,
                                   "marginTop": "4px"}),
        ], style={"flex": "1", "textAlign": "center", "padding": "18px 12px"})

    def detail(label, value, mono=True):
        body = (html.Code(value, style={
                    "display": "block", "fontSize": "13px", "color": INK,
                    "background": "#f6f8f9", "border": "1px solid #e6e9ec",
                    "borderRadius": "6px", "padding": "8px 12px",
                    "wordBreak": "break-all", "whiteSpace": "normal"})
                if mono else
                html.Div(value, style={"fontSize": "15px", "color": INK,
                                       "fontWeight": 500}))
        return html.Div([
            html.Div(label, style={"fontSize": "13px", "color": MUTED,
                                   "marginBottom": "6px"}),
            body,
        ], style={"marginBottom": "18px", "textAlign": "left"})

    divider = html.Div(style={"width": "1px", "background": "#e6e9ec",
                              "margin": "14px 0"})

    card = html.Div([
        html.Div([stat(f"{rows:,}", "rows"), divider,
                  stat(f"{columns:,}", "columns")],
                 style={"display": "flex", "border": "1px solid #e6e9ec",
                        "borderRadius": "10px", "marginBottom": "26px",
                        "background": "#fbfcfc"}),

        html.Div([
            html.Div("Columns", style={"fontSize": "13px", "color": MUTED,
                                       "marginBottom": "6px"}),
            dcc.Dropdown(
                id="dataset-columns",
                options=[{"label": str(c), "value": str(c)}
                         for c in frame.columns],
                placeholder=f"Search the {columns} columns",
                clearable=True, searchable=True,
                style={"fontSize": "14px"},
            ),
        ], style={"marginBottom": "18px", "textAlign": "left"}),

        detail("Client ID", str(field("client_id") or "not assigned yet"),
               mono=False),
        detail("Dataset file", str(field("dataset_csv_path")
                                   or field("dataset_path") or "unknown")),
        detail("Config file", str(field("config_path") or "unknown")),
    ], style={"background": "white", "border": "1px solid #eef1f3",
              "borderRadius": "12px", "padding": "28px 28px 10px",
              "boxShadow": "0 1px 4px rgba(18, 50, 42, 0.06)"})

    return html.Div([
        html.H3("Dataset", style={"color": INK, "textAlign": "center",
                                  "marginBottom": "20px"}),
        card,
    ], style=DATASET_PAGE)


def predictions_view():
    """Out-of-fold predictions: every row was held out in exactly one fold."""
    frame = field("predictions")
    if frame is None or len(frame) == 0:
        return None

    accuracy = field("predictions_accuracy")
    model = field("predictions_model") or "the final model"
    wrong = int((~frame["correct"]).sum())

    display = frame.rename(columns={
        "row": "Row", "fold": "Fold", "true": "True class",
        "predicted": "Predicted", "confidence": "Confidence",
        "correct": "Correct"})
    display["Confidence"] = display["Confidence"].map(lambda v: f"{v:.3f}")
    display["Correct"] = display["Correct"].map({True: "yes", False: "no"})

    summary = html.Div([
        chip(f"{len(frame):,} rows scored", "#eef1f3"),
        chip(f"{accuracy:.2%} correct" if accuracy is not None else "—",
             "#eaf8f2"),
        chip(f"{wrong} misclassified", "#fdf1ee"),
    ], style={"marginBottom": "8px"})

    misclassified = display[display["Correct"] == "no"]

    return html.Div([
        html.H4("Predictions on held-out data", style={"color": INK,
                                                       "marginTop": "28px"}),
        html.P(f"Every row is held out in exactly one cross-validation fold, so "
               f"these are out-of-fold predictions from {model} covering the "
               f"whole local dataset.",
               style={"color": MUTED, "fontSize": "13px"}),
        summary,
        expander(f"Show all {len(display):,} predictions",
                 paged_table(display, "predictions-full")),
        expander(f"Show the {len(misclassified)} misclassified rows",
                 paged_table(misclassified, "predictions-wrong"))
        if len(misclassified) else html.Div(),
    ])


def error_banner():
    """Surface a state failure in the dashboard instead of only in the log."""
    if not field("error"):
        return None
    return html.Details([
        html.Summary("A step of the workflow failed — results may be partial. "
                     "Click for details.",
                     style={"cursor": "pointer", "fontWeight": 600,
                            "color": "#8a3d2e"}),
        html.Pre(str(field("error")),
                 style={"whiteSpace": "pre-wrap", "fontSize": "12px",
                        "marginTop": "10px", "maxHeight": "260px",
                        "overflowY": "auto", "color": "#5c2a20"}),
    ], style={"background": "#fdf1ee", "border": "1px solid #f0c8bd",
              "borderRadius": "8px", "padding": "12px 14px",
              "marginBottom": "18px"})


def structure_metrics_table(blocks):
    """Metrics down the rows, networks across the columns, best value marked."""
    networks = list(blocks)
    present = [m for m in STRUCTURE_METRIC_ORDER
               if any(m in blocks[n] for n in networks)]

    rows, highlights = [], []
    for index, metric in enumerate(present):
        label = STRUCTURE_METRIC_LABELS.get(metric, metric)
        row = {"Metric": label}
        values = {}
        for network in networks:
            value = blocks[network].get(metric)
            values[network] = value
            if value is None or (isinstance(value, float) and np.isnan(value)):
                row[network] = "—"
            elif metric in STRUCTURE_COUNTS:
                row[network] = f"{value:.0f}"
            else:
                row[network] = f"{value:.4f}"
        rows.append(row)

        # tp/fp/fn are diagnostic counts, not a verdict -- don't crown a winner
        if metric in {"tp", "fp", "fn", "num_edges"}:
            continue
        finite = {n: val for n, val in values.items()
                  if val is not None and not (isinstance(val, float) and np.isnan(val))}
        if finite:
            pick = min if metric in LOWER_IS_BETTER else max
            highlights.append({
                "if": {"row_index": index,
                       "column_id": pick(finite, key=finite.get)},
                "backgroundColor": "#eaf8f2", "fontWeight": "600", "color": INK,
            })

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": "Metric", "id": "Metric"}] +
                [{"name": n, "id": n} for n in networks],
        style_cell={"textAlign": "right", "fontSize": "13px",
                    "padding": "8px 10px", "fontFamily": "inherit"},
        style_cell_conditional=[{"if": {"column_id": "Metric"},
                                 "textAlign": "left", "fontWeight": "500"}],
        style_header={"backgroundColor": "#f6f8f9", "fontWeight": "600",
                      "borderBottom": "2px solid #e0e5e9",
                      "whiteSpace": "normal", "height": "auto"},
        style_data={"borderBottom": "1px solid #eef1f3"},
        style_data_conditional=highlights,
        style_table={"overflowX": "auto"},
    )


def structure_metrics_section():
    """How well each network recovers the benchmark. Independent of has_target."""
    blocks = selected_structure_metrics()

    if not blocks:
        if field("testing_enabled") is False:
            return html.Div([
                html.H4("Structure recovery", style={"color": INK}),
                html.P("Structure-recovery metrics are scored against a known "
                       "benchmark network and are only computed when "
                       "`testing: true` is set in the config.",
                       style={"color": MUTED}),
            ])
        return html.Div([
            html.H4("Structure recovery", style={"color": INK}),
            html.P("Waiting for the global network to be aggregated...",
                   style={"color": MUTED}),
        ])

    against = (f" against the {field('benchmark')} benchmark network"
               if field("benchmark") else " against the benchmark network")
    shd = {n: b.get("shd") for n, b in blocks.items()}
    delta = None
    if shd.get("Local network") is not None and \
            shd.get("Global network") is not None:
        change = shd["Global network"] - shd["Local network"]
        wording = ("closer to" if change < 0 else
                   "further from" if change > 0 else "the same distance from")
        delta = html.P(f"Federation moved the network {abs(change):.0f} edit"
                       f"{'' if abs(change) == 1 else 's'} {wording} the "
                       f"benchmark (SHD {shd['Local network']:.0f} → "
                       f"{shd['Global network']:.0f}).",
                       style={"fontSize": "13px",
                              "color": INK if change <= 0 else "#8a3d2e"})

    return html.Div([
        html.H4("Structure recovery", style={"color": INK}),
        html.P(f"How closely each learned network matches the true edge set"
               f"{against}. Computed for every run, with or without a target "
               f"variable.", style={"color": MUTED, "fontSize": "13px"}),
        delta,
        structure_metrics_table(blocks),
        html.Div("Lower SHD is better; every other score is higher-is-better. "
                 "True/false positive counts are shown for diagnosis and are "
                 "not ranked.",
                 style={"fontSize": "12px", "color": MUTED, "marginTop": "10px"}),
    ])


def block_means(blocks):
    """{short name: mean-metric dict} for one client's evaluation."""
    chosen = {}
    for marker, short in EVAL_PAIR:
        for label, block in (blocks or {}).items():
            if label.startswith(marker) and block and block.get("mean"):
                chosen[short] = block
                break
    return chosen


def global_vs_local_chart():
    """Global DAG against the local DAGs, averaged over every client.

    Both bars aggregate all clients: the global DAG is one shared structure but
    each client scores it on its own data, and each local DAG is that client's
    own. The error bar is the spread between clients on the local bar.

    Coordinator-only, because it is the only party that holds other clients'
    numbers.
    """
    everyone = field("all_evaluations") or {}
    if not field("is_coordinator") or len(everyone) < 2:
        return html.Div()

    per_network = {short: [] for _marker, short in EVAL_PAIR}
    for blocks in everyone.values():
        for short, block in block_means(blocks).items():
            if short in per_network:
                per_network[short].append(block["mean"])

    rows = []
    for metric in CHART_METRICS:
        for short, means in per_network.items():
            values = [m[metric] for m in means
                      if metric in m and m[metric] is not None
                      and not np.isnan(m[metric])]
            if not values:
                continue
            rows.append({"metric": metric, "network": short,
                         "value": float(np.mean(values)),
                         "error": float(np.std(values))
                                  if short == "Local network" else 0.0})
    if not rows:
        return html.Div()

    figure = px.bar(pd.DataFrame(rows), x="metric", y="value", color="network",
                    barmode="group", error_y="error",
                    color_discrete_sequence=NETWORK_COLORS)
    figure.update_traces(marker_line=dict(width=1, color=NODE_BORDER),
                         hovertemplate="%{x}: %{y:.4f}<extra></extra>")
    figure.update_layout(xaxis_title=None, yaxis_title="Score",
                         legend_title_text="",
                         legend=dict(orientation="h", y=-0.18),
                         margin=dict(l=10, r=10, t=10, b=10), height=400,
                         plot_bgcolor="white", bargap=0.25)
    figure.update_yaxes(gridcolor="#eef1f3", zerolinecolor="#dfe4e8",
                        range=[0, 1.05])
    figure.update_xaxes(showgrid=False)

    return html.Div([
        dcc.Graph(figure=figure, config={"displayModeBar": False}),
        html.Div(f"Each bar averages all {len(everyone)} clients. The error bar "
                 f"on the local network is the spread between them.",
                 style={"fontSize": "12px", "color": MUTED,
                        "marginTop": "-6px", "marginBottom": "6px"}),
    ])


def cross_client_section():
    """Every client's numbers side by side, the coordinator's own included.

    Only the coordinator ever receives these -- participants send their metrics
    upstream and get nothing back.
    """
    if not field("is_coordinator"):
        return None
    everyone = field("all_evaluations") or {}
    if len(everyone) < 2:
        return None

    me = field("client_id")
    per_client = {}
    for client_id, blocks in everyone.items():
        chosen = block_means(blocks)
        if chosen:
            per_client[client_id] = chosen
    if not per_client:
        return None

    # Coordinator first, then the rest sorted, so the numbering is stable for
    # the whole run and Client 1 is always this coordinator.
    order = ([me] if me in per_client else []) + \
            sorted(c for c in per_client if c != me)
    names = {client_id: f"Client {position}: {client_id}"
             for position, client_id in enumerate(order, start=1)}

    sections = []
    for _marker, short in EVAL_PAIR:
        metrics = ordered_metrics({m for c in order
                                   for m in per_client[c].get(short, {})
                                             .get("mean", {})})
        metrics = [m for m in metrics if m != "num_edges"]
        if not metrics:
            continue

        rows, highlights = [], []
        for index, metric in enumerate(metrics):
            row, values = {"Metric": metric}, {}
            for client_id in order:
                block = per_client[client_id].get(short)
                mean = (block or {}).get("mean", {}).get(metric)
                std = (block or {}).get("std", {}).get(metric)
                values[names[client_id]] = mean
                row[names[client_id]] = (
                    "—" if mean is None or np.isnan(mean)
                    else f"{mean:.4f} ± {std:.4f}" if std is not None
                    else f"{mean:.4f}")
            rows.append(row)
            finite = {n: v for n, v in values.items()
                      if v is not None and not np.isnan(v)}
            if finite:
                pick = min if metric in LOWER_IS_BETTER else max
                highlights.append({
                    "if": {"row_index": index,
                           "column_id": pick(finite, key=finite.get)},
                    "backgroundColor": "#eaf8f2", "fontWeight": "600",
                    "color": INK})

        sections.append(html.Div([
            html.H5(short, style={"color": INK, "marginTop": "18px",
                                  "marginBottom": "6px"}),
            dash_table.DataTable(
                data=rows,
                columns=[{"name": "Metric", "id": "Metric"}] +
                        [{"name": names[c], "id": names[c]} for c in order],
                style_cell={"textAlign": "right", "fontSize": "13px",
                            "padding": "8px 10px", "fontFamily": "inherit"},
                style_cell_conditional=[{"if": {"column_id": "Metric"},
                                         "textAlign": "left",
                                         "fontWeight": "500"}],
                style_header={"backgroundColor": "#f6f8f9", "fontWeight": "600",
                              "borderBottom": "2px solid #e0e5e9",
                              "whiteSpace": "normal", "height": "auto"},
                style_data={"borderBottom": "1px solid #eef1f3"},
                style_data_conditional=highlights,
                style_table={"overflowX": "auto"}),
        ]))

    return html.Div([
        html.H4("Across all clients", style={"color": INK, "marginTop": "32px"}),
        html.P(f"All {len(order)} clients, this coordinator included as "
               f"Client 1. Mean ± standard deviation across folds; best in row "
               f"highlighted. Visible to the coordinator only.",
               style={"color": MUTED, "fontSize": "13px"}),
        global_vs_local_chart(),
    ] + sections)


def performance_section():
    """Predictive metrics. Only meaningful when a target variable is configured."""
    if field("has_target") is False:
        return html.Div([
            html.H4("Predictive performance", style={"color": INK}),
            html.P("No target variable is configured for this dataset "
                   "(input.has_target is false), so predictive performance is "
                   "not evaluated. The network structure above is the result.",
                   style={"color": MUTED}),
        ])

    evaluation = selected_evaluation()

    if not evaluation:
        waiting = ("Waiting for the evaluation step..."
                   if field("has_target") else
                   "Waiting for the configuration to be read...")
        return html.Div([
            html.H4("Predictive performance", style={"color": INK}),
            html.P(waiting, style={"color": MUTED}),
        ])

    frame = evaluation_frame(evaluation)
    if frame.empty:
        return html.Div([
            html.H4("Predictive performance", style={"color": INK}),
            html.P("No metrics available.", style={"color": MUTED}),
        ])

    missing = len(EVAL_PAIR) - len(evaluation)
    note = (html.P(f"{missing} of the 2 compared models could not be evaluated; "
                   f"see the app log for the reason.",
                   style={"color": "#8a3d2e", "fontSize": "13px"})
            if missing > 0 else None)

    return html.Div([
        html.H4("Predictive performance", style={"color": INK}),
        note,
        headline_cards(frame),
        html.H4("All metrics", style={"marginTop": "28px", "color": INK}),
        metric_table(frame),
        predictions_view() or html.Div(),
        html.Div("The refined-local and global-DAG-with-local-params variants "
                 "are still computed and written to metrics.json; this tab "
                 "compares the local and global networks only.",
                 style={"fontSize": "12px", "color": MUTED, "marginTop": "14px"}),
    ])


def evaluation_view():
    """Structure first -- it exists long before any metrics do -- then, when a
    target variable is configured, predictive performance as soon as it lands."""
    rule = lambda: html.Hr(style={"border": "none",
                                  "borderTop": "1px solid #eef1f3",
                                  "margin": "32px 0 20px"})
    return html.Div([
        html.H3('Evaluation'),
        structure_comparison(),
        rule(),
        structure_metrics_section(),
        rule(),
        performance_section(),
        cross_client_section() or html.Div(),
    ])


# ------------------------------------------------------------- structures ---

# Node radius scales with connectivity. MIN is the size every node had before,
# so nothing shrinks; the target variable is pushed past the busiest node so it
# always reads as the largest thing on the canvas.
NODE_MIN_SIZE = 40
# Every view sizes nodes the same way: the least-connected node keeps the
# view's base size, the most-connected one is (1 + NODE_GROWTH) times it, and
# everything in between scales linearly with degree.
NODE_GROWTH = 1.0


def degree_scaled_sizes(degrees, base):
    """{node: size} scaled by degree relative to the other nodes on screen."""
    if not degrees:
        return {}
    lo, hi = min(degrees.values()), max(degrees.values())
    span = hi - lo
    return {node: base * (1 + NODE_GROWTH * ((count - lo) / span if span else 0))
            for node, count in degrees.items()}


def structure_stylesheet(fill):
    return [
        {"selector": "node", "style": {
            "label": "data(label)", "background-color": fill,
            "width": "data(size)", "height": "data(size)",
            "border-width": NODE_BORDER_WIDTH, "border-color": NODE_BORDER,
            "font-size": "11px", "text-valign": "center", "color": INK,
        }},
        {"selector": "edge", "style": {
            "curve-style": "bezier", "target-arrow-shape": "triangle",
            "line-color": "#9aa5b1", "target-arrow-color": "#9aa5b1",
            "width": 1.5,
        }},
        # When a node is picked, everything unrelated fades and its parent and
        # child edges are drawn in, so its neighbourhood reads at a glance.
        {"selector": ".faded", "style": {"opacity": 0.12}},
        {"selector": "node.picked", "style": {
            "border-width": 3, "border-color": "#1f1f1f"}},
        {"selector": "edge.parent-edge", "style": {
            "line-color": "#4a7fb5", "target-arrow-color": "#4a7fb5", "width": 3}},
        {"selector": "edge.child-edge", "style": {
            "line-color": "#2f9e76", "target-arrow-color": "#2f9e76", "width": 3}},
        {"selector": "node.parent-node", "style": {
            "border-width": 2, "border-color": "#4a7fb5"}},
        {"selector": "node.child-node", "style": {
            "border-width": 2, "border-color": "#2f9e76"}},
        *slice_rules(),
        target_rule(),
    ]


def node_sizes(edges):
    """Degree-scaled size per node for the flat (non-2TBN) views."""
    degree = {}
    for source, node_target in edges:
        degree[source] = degree.get(source, 0) + 1
        degree[node_target] = degree.get(node_target, 0) + 1
    return degree_scaled_sizes(degree, NODE_MIN_SIZE)


def edges_to_elements(edges, picked=None):
    edges = [tuple(edge) for edge in edges]
    sizes = node_sizes(edges)
    nodes = sorted({n for edge in edges for n in edge})

    parents = {s for s, t in edges if t == picked}
    children = {t for s, t in edges if s == picked}
    related = parents | children | ({picked} if picked else set())

    elements = []
    for node in nodes:
        classes = [node_classes(node)]
        if picked:
            if node == picked:
                classes.append("picked")
            elif node in parents:
                classes.append("parent-node")
            elif node in children:
                classes.append("child-node")
            else:
                classes.append("faded")
        elements.append({"data": {"id": node, "label": node_label(node),
                                  "size": sizes.get(node, NODE_MIN_SIZE)},
                         "classes": " ".join(c for c in classes if c)})

    for source, node_target in edges:
        classes = ""
        if picked:
            if node_target == picked:
                classes = "parent-edge"
            elif source == picked:
                classes = "child-edge"
            else:
                classes = "faded"
        elements.append({"data": {"source": source, "target": node_target},
                         "classes": classes})
    return elements


# ------------------------------------------------------------ 2TBN drawing ---
# The two-slice picture mirrors the Streamlit prototype (visualize_bayesian_
# network_pyvis): a spring layout computed once on the within-slice structure,
# mirrored into a left "(t)" column (the learner's tm1 slice) and a right
# "(t+1)" column (the learner's t slice). Every variable, static ones included,
# appears in both columns.

# Spring layout, then the same scale factors the prototype used: x*3 -/+ 3.5 per
# column, y*5, and 400/250 px per unit. Nodes are 100 px wide with 40 px labels
# underneath, so the proportions match the vis.js rendering exactly.
TBN_SPRING_K = 10
TBN_SPRING_ITERATIONS = 300
TBN_SPRING_SEED = 23
TBN_X_SCALE, TBN_Y_SCALE, TBN_X_OFFSET = 3, 5, 3.5
TBN_PX_X, TBN_PX_Y = 400, 250
TBN_NODE_SIZE = 100
TBN_FONT_SIZE = 40

TBN_PREV_FILL = "#a8d8f0"     # sky blue -- slice (t)
TBN_CURR_FILL = "#a8e6b8"     # mint     -- slice (t+1)
TBN_STATIC_FILL = "#ffd6a5"   # amber    -- variables with no time suffix
TBN_TARGET_FILL = "#ffadad"   # rose     -- the target variable
TBN_BORDER = "#000000"
# rgba(0,0,0,0.45) on white, as a solid colour so it does not fight the
# opacity used for fading unrelated edges.
TBN_INTRA_COLOR = "#8c8c8c"
TBN_INTER_COLOR = "#a888b5"
TBN_INTRA_WIDTH = 4
TBN_INTER_WIDTH = 5


def two_slice_layout(bases, intra):
    """Spring layout on the within-slice structure, shared by both columns."""
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(bases))
    graph.add_edges_from((a, b) for a, b, _ in intra if a != b)
    if graph.number_of_nodes() == 1:
        return {next(iter(graph.nodes)): (0.0, 0.0)}
    return nx.spring_layout(graph, k=TBN_SPRING_K,
                            iterations=TBN_SPRING_ITERATIONS,
                            seed=TBN_SPRING_SEED)


TWO_SLICE_NEIGHBOURS = {}


def slice_of_id(node_id):
    """'prev::Beruf' -> 'Beruf (t)'. Readable name for the summary line."""
    if not node_id or "::" not in node_id:
        return str(node_id)
    column, base = node_id.split("::", 1)
    return f"{base} ({'t+1' if column == 'curr' else 't'})"


def two_slice_elements(edges, picked=None, edge_state=None):
    """The 2TBN picture: slice (t) on the left, slice (t+1) on the right.

    Edge rules follow the Streamlit prototype:
      past -> current/static    inter-slice, (t) -> (t+1)
      current/static -> same    within a slice, mirrored into both columns
      past -> past              within a slice, mirrored into both columns
    Duplicates collapse, and when both orientations of a pair were learned the
    first one seen is kept so neither replica shows a two-way arrow.
    """
    edges = [tuple(e) for e in edges]
    bases, intra, inter = [], [], []
    intra_pairs, inter_pairs = set(), set()

    def remember(name):
        if name not in bases:
            bases.append(name)

    for source, target_node in edges:
        source_base, source_lag = parse_node(source)
        target_base, target_lag = parse_node(target_node)
        remember(source_base); remember(target_base)
        learned = (source, target_node)
        source_past = source_lag is not None and source_lag >= 1
        target_past = target_lag is not None and target_lag >= 1

        if source_past and not target_past:
            pair = (source_base, target_base)
            if pair not in inter_pairs:
                inter_pairs.add(pair)
                inter.append((source_base, target_base, learned, "inter-edge"))
        elif source_lag == 0 and target_past:
            # Backwards in time. define_forbidden_edges rules these out, so one
            # here means a constraint did not apply; flag rather than hide it.
            inter.append((source_base, target_base, learned, "time-violation"))
        else:
            pair = (source_base, target_base)
            if source_base == target_base or pair in intra_pairs \
                    or (target_base, source_base) in intra_pairs:
                continue
            intra_pairs.add(pair)
            intra.append((source_base, target_base, learned))

    if not bases:
        return edges_to_elements(edges, picked)

    layout = two_slice_layout(bases, intra)
    temporal = {parse_node(c)[0] for edge in edges for c in edge
                if parse_node(c)[1] is not None}
    target_name = field("target") if field("has_target") else None

    drawn = []
    for column in ("prev", "curr"):
        for source_base, target_base, learned in intra:
            drawn.append((f"{column}::{source_base}",
                          f"{column}::{target_base}", "intra-edge", learned))
    for source_base, target_base, learned, kind in inter:
        if kind == "time-violation":
            drawn.append((f"curr::{source_base}", f"prev::{target_base}",
                          kind, learned))
        else:
            drawn.append((f"prev::{source_base}", f"curr::{target_base}",
                          kind, learned))

    parents = {u for u, v, *_ in drawn if v == picked}
    children = {v for u, v, *_ in drawn if u == picked}
    related = parents | children | ({picked} if picked else set())
    # Cached for the summary line: it has to read the drawn graph, not the raw
    # edge list, because a static variable appears in both slices under a
    # "prev::"/"curr::" id that no column name matches.
    TWO_SLICE_NEIGHBOURS.clear()
    TWO_SLICE_NEIGHBOURS.update(picked=picked, parents=sorted(parents),
                                children=sorted(children))

    # Degree is counted on the drawn graph, so each replica is sized by the
    # edges it actually shows. Blacklisted edges in the validation editor are
    # still drawn (so they can be undone) but no longer count.
    degree = {f"{column}::{base}": 0
              for column in ("prev", "curr") for base in bases}
    for source_id, target_id, _kind, learned in drawn:
        if edge_state and edge_state(learned) == "blacklisted":
            continue
        degree[source_id] = degree.get(source_id, 0) + 1
        degree[target_id] = degree.get(target_id, 0) + 1
    sizes = degree_scaled_sizes(degree, TBN_NODE_SIZE)

    elements = []
    for column, side in (("prev", -1), ("curr", 1)):
        for base in sorted(bases):
            node_id = f"{column}::{base}"
            x, y = layout[base]
            classes = ["tbn"]
            if base == target_name:
                classes.append("target")
            elif base in temporal:
                classes.append(f"tbn-{column}")
            else:
                classes.append("tbn-static")
            if picked:
                if node_id == picked:
                    classes.append("picked")
                elif node_id in parents:
                    classes.append("parent-node")
                elif node_id in children:
                    classes.append("child-node")
                else:
                    classes.append("faded")
            elements.append({
                "data": {"id": node_id, "label": base,
                         "size": sizes.get(node_id, TBN_NODE_SIZE)},
                "position": {
                    "x": (x * TBN_X_SCALE + side * TBN_X_OFFSET) * TBN_PX_X,
                    "y": y * TBN_Y_SCALE * TBN_PX_Y},
                "classes": " ".join(classes)})

    for source_id, target_id, classes, learned in drawn:
        if edge_state:
            extra = edge_state(learned)
            if extra:
                classes = f"{classes} {extra}"
        if picked:
            if target_id == picked:
                classes = f"{classes} parent-edge"
            elif source_id == picked:
                classes = f"{classes} child-edge"
            elif source_id not in related or target_id not in related:
                classes = f"{classes} faded"
        # The learned endpoints travel with the drawn edge so a click maps back
        # to one real edge: an intra edge is drawn once per slice, and its id
        # alone could not say which learned edge it came from.
        elements.append({"data": {"source": source_id, "target": target_id,
                                  "learned_source": learned[0],
                                  "learned_target": learned[1]},
                         "classes": classes})
    return elements


def column_of(base, lag):
    if lag is None:
        return f"static::{base}"
    return f"curr::{base}" if lag == 0 else f"prev::{base}"


# Arrowhead size relative to the edge width (Cytoscape's arrow-scale).
TBN_ARROW_SCALE = 2.2

_TBN_EDGE = {"curve-style": "unbundled-bezier",
             "control-point-distances": 60, "control-point-weights": 0.5,
             "target-arrow-shape": "triangle", "arrow-scale": TBN_ARROW_SCALE}

# Base look of the 2TBN: node fills per slice, label under the node, thick
# edges with dashed purple inter-slice arcs.
TWO_SLICE_EXTRA = [
    {"selector": "node.tbn", "style": {
        "width": "data(size)", "height": "data(size)",
        "border-width": 2, "border-color": TBN_BORDER,
        "font-size": TBN_FONT_SIZE, "color": "#111111",
        "font-family": "Segoe UI, Helvetica, Arial, sans-serif",
        "text-valign": "bottom", "text-halign": "center", "text-margin-y": 8}},
    {"selector": "node.tbn-prev", "style": {"background-color": TBN_PREV_FILL}},
    {"selector": "node.tbn-curr", "style": {"background-color": TBN_CURR_FILL}},
    {"selector": "node.tbn-static", "style": {"background-color": TBN_STATIC_FILL}},
    {"selector": "node.tbn.target", "style": {"background-color": TBN_TARGET_FILL}},
    {"selector": "edge.intra-edge", "style": dict(
        _TBN_EDGE, **{"line-color": TBN_INTRA_COLOR,
                      "target-arrow-color": TBN_INTRA_COLOR,
                      "width": TBN_INTRA_WIDTH})},
    {"selector": "edge.inter-edge", "style": dict(
        _TBN_EDGE, **{"line-color": TBN_INTER_COLOR,
                      "target-arrow-color": TBN_INTER_COLOR,
                      "width": TBN_INTER_WIDTH, "line-style": "dashed"})},
    # Should never appear: an edge pointing backwards in time.
    {"selector": "edge.time-violation", "style": dict(
        _TBN_EDGE, **{"line-color": "#c0392b", "target-arrow-color": "#c0392b",
                      "width": 6, "line-style": "dotted"})},
]

# Click highlighting, scaled to the 2TBN's larger nodes and edges. Listed after
# any per-view edge colouring so a picked neighbourhood always reads.
TWO_SLICE_HIGHLIGHT = [
    {"selector": ".faded", "style": {"opacity": 0.08}},
    {"selector": "node.picked", "style": {"border-width": 7,
                                          "border-color": "#1f1f1f"}},
    {"selector": "node.parent-node", "style": {"border-width": 6,
                                               "border-color": "#4a7fb5"}},
    {"selector": "node.child-node", "style": {"border-width": 6,
                                              "border-color": "#2f9e76"}},
    {"selector": "edge.parent-edge", "style": {
        "line-color": "#4a7fb5", "target-arrow-color": "#4a7fb5", "width": 9,
        "opacity": 1}},
    {"selector": "edge.child-edge", "style": {
        "line-color": "#2f9e76", "target-arrow-color": "#2f9e76", "width": 9,
        "opacity": 1}},
]


# --------------------------------------------------- higher-order drawing ---
# dbn_order > 1 uses the 2TBN template -- same node sizes and degree scaling,
# fonts, borders, colours and edge styles -- on a single network: one node per
# learned column, labelled with its slice, laid out left to right in time.

# Current slice keeps the 2TBN mint; past slices get successively deeper
# shades of the 2TBN sky blue, so t-1 still looks like the 2TBN "(t)" slice.
TBN_LAG_FILLS = ["#a8d8f0", "#8ec5ea", "#76b2e2", "#629fd8"]


def tbn_lag_fill(lag):
    if lag == 0:
        return TBN_CURR_FILL
    return TBN_LAG_FILLS[min(lag, len(TBN_LAG_FILLS)) - 1]


HIGHER_ORDER_EXTRA = [
    {"selector": f"node.tbn-lag-{lag}",
     "style": {"background-color": tbn_lag_fill(lag)}}
    for lag in range(1, len(TBN_LAG_FILLS) + 1)
] + [
    # Re-assert the target colour after the lag fills.
    {"selector": "node.tbn.target", "style": {"background-color": TBN_TARGET_FILL}},
]

# Hierarchical, left to right: edges run from older slices towards t. The
# spacing is scaled to the 2TBN's 100-200 px nodes and 40 px labels.
HIGHER_ORDER_LAYOUT = {"name": "dagre", "rankDir": "LR", "nodeSep": 90,
                       "rankSep": 320, "edgeSep": 30, "padding": 60,
                       "fit": True}


def higher_order_ready(edges):
    """dbn_order > 1: a temporal model that is not drawn as a 2TBN."""
    return not two_slice_ready(edges) and bool(slices_present(edges))


def higher_order_elements(edges, picked=None, edge_state=None):
    """The higher-order DBN as one network, styled like the 2TBN.

    Edge kinds follow the 2TBN rules, with a static variable counted as part
    of whichever slice it connects to:
      same slice             -> within-slice edge (grey)
      older -> newer slice   -> inter-slice edge (dashed purple)
      newer -> older slice   -> time violation (red dotted; should not occur)
    """
    edges = [tuple(e) for e in edges]
    target_name = field("target") if field("has_target") else None
    nodes = sorted({n for edge in edges for n in edge})

    drawn = []
    for source, node_target in edges:
        source_lag, target_lag = slice_of(source), slice_of(node_target)
        effective_target = target_lag if target_lag is not None else 0
        effective_source = source_lag if source_lag is not None else effective_target
        if effective_source == effective_target:
            kind = "intra-edge"
        elif effective_source > effective_target:
            kind = "inter-edge"
        else:
            kind = "time-violation"
        if edge_state:
            extra = edge_state((source, node_target))
            if extra:
                kind = f"{kind} {extra}"
        drawn.append((source, node_target, kind))

    # Degree on the drawn graph; blacklisted edges in the editor stay visible
    # (so they can be undone) but no longer count, exactly as in the 2TBN.
    degree = {n: 0 for n in nodes}
    for source, node_target, kind in drawn:
        if "blacklisted" in kind.split():
            continue
        degree[source] += 1
        degree[node_target] += 1
    sizes = degree_scaled_sizes(degree, TBN_NODE_SIZE)

    parents = {u for u, v, _ in drawn if v == picked}
    children = {v for u, v, _ in drawn if u == picked}
    related = parents | children | ({picked} if picked else set())

    elements = []
    for node in nodes:
        lag = slice_of(node)
        classes = ["tbn"]
        if node == target_name:
            classes.append("target")
        elif lag is None:
            classes.append("tbn-static")
        elif lag == 0:
            classes.append("tbn-curr")
        else:
            classes.append(f"tbn-lag-{min(lag, len(TBN_LAG_FILLS))}")
        if picked:
            if node == picked:
                classes.append("picked")
            elif node in parents:
                classes.append("parent-node")
            elif node in children:
                classes.append("child-node")
            else:
                classes.append("faded")
        elements.append({"data": {"id": node, "label": node_label(node),
                                  "size": sizes.get(node, TBN_NODE_SIZE)},
                         "classes": " ".join(classes)})

    for source, node_target, kind in drawn:
        classes = kind
        if picked:
            if node_target == picked:
                classes += " parent-edge"
            elif source == picked:
                classes += " child-edge"
            elif source not in related or node_target not in related:
                classes += " faded"
        elements.append({"data": {"source": source, "target": node_target,
                                  "id": f"{source}|{node_target}"},
                         "classes": classes})
    return elements


def higher_order_legend(edges):
    """Colour key for the higher-order view, in the 2TBN legend's style."""
    def swatch(colour):
        return html.Span(style={"display": "inline-block", "width": "11px",
                                "height": "11px", "borderRadius": "50%",
                                "background": colour,
                                "border": f"1px solid {TBN_BORDER}",
                                "marginRight": "6px",
                                "verticalAlign": "middle"})

    def item(colour, text):
        return html.Span([swatch(colour), text],
                         style={"fontSize": "12px", "color": MUTED,
                                "marginRight": "16px"})

    lags = slices_present(edges)
    items = [item(tbn_lag_fill(lag), f"slice {slice_label(lag)}")
             for lag in sorted(lags, reverse=True)]
    if any(slice_of(n) is None for edge in edges for n in edge):
        items.append(item(TBN_STATIC_FILL, "static"))
    if field("has_target") and field("target"):
        items.append(item(TBN_TARGET_FILL, "target"))
    items += [
        html.Span("— within-slice edge",
                  style={"fontSize": "12px", "color": TBN_INTRA_COLOR,
                         "marginRight": "16px"}),
        html.Span("--- inter-slice edge",
                  style={"fontSize": "12px", "color": TBN_INTER_COLOR}),
    ]
    return html.Div(items, style={"marginBottom": "8px"})


def network_legend(edges):
    """The legend matching whichever representation the edges are drawn in."""
    if two_slice_ready(edges):
        return two_slice_legend()
    if higher_order_ready(edges):
        return higher_order_legend(edges)
    return slice_legend(edges) or html.Div()


def two_slice_stylesheet(base, state_rules=()):
    """Cytoscape applies rules in order, later ones winning, so: the view's own
    base rules, then the 2TBN look, then per-view edge states (shared/unique,
    blacklisted/whitelisted), then highlighting."""
    return (list(base) + TWO_SLICE_EXTRA + HIGHER_ORDER_EXTRA
            + list(state_rules) + TWO_SLICE_HIGHLIGHT)


TWO_SLICE_LAYOUT = {"name": "preset", "fit": True, "padding": 70}


def two_slice_canvas(graph):
    """Wrap a Cytoscape graph with the "Slice (t)" / "Slice (t+1)" banners.

    The banners float over the top of the canvas, one per half, exactly as the
    prototype's overlay did. The layout is symmetric about x = 0 and the preset
    layout fits it to the canvas, so each banner sits over its slice.
    """
    def banner(text, colour, fill, border):
        return html.Div(text, style={
            "width": "50%", "textAlign": "center",
            "fontFamily": "Segoe UI, Helvetica, Arial, sans-serif",
            "fontSize": "20px", "fontWeight": 700, "color": colour,
            "letterSpacing": "0.02em", "background": fill,
            "borderRadius": "8px", "margin": "0 6px", "padding": "5px 0",
            "border": f"1.5px solid {border}"})

    overlay = html.Div([
        banner("Slice (t)", "#1a5f8a", "rgba(168,216,240,0.55)",
               "rgba(26,122,181,0.35)"),
        banner("Slice (t+1)", "#1a6e36", "rgba(168,230,184,0.55)",
               "rgba(26,140,69,0.35)"),
    ], style={"position": "absolute", "top": "12px", "left": 0,
              "width": "100%", "display": "flex", "pointerEvents": "none",
              "zIndex": 10, "boxSizing": "border-box"})
    return html.Div([graph, overlay], style={"position": "relative"})


def two_slice_legend():
    """Colour key for the 2TBN view."""
    def swatch(colour):
        return html.Span(style={"display": "inline-block", "width": "11px",
                                "height": "11px", "borderRadius": "50%",
                                "background": colour,
                                "border": f"1px solid {TBN_BORDER}",
                                "marginRight": "6px",
                                "verticalAlign": "middle"})

    def item(colour, text):
        return html.Span([swatch(colour), text],
                         style={"fontSize": "12px", "color": MUTED,
                                "marginRight": "16px"})

    items = [item(TBN_PREV_FILL, "slice (t)"),
             item(TBN_CURR_FILL, "slice (t+1)"),
             item(TBN_STATIC_FILL, "static")]
    if field("has_target") and field("target"):
        items.append(item(TBN_TARGET_FILL, "target"))
    items += [
        html.Span("— within-slice edge",
                  style={"fontSize": "12px", "color": TBN_INTRA_COLOR,
                         "marginRight": "16px"}),
        html.Span("--- inter-slice edge",
                  style={"fontSize": "12px", "color": TBN_INTER_COLOR}),
    ]
    return html.Div(items, style={"marginBottom": "8px"})


# ------------------------------------------------------------ node sizing ---
# Each node's degree-scaled size is also written as a per-element style
# bypass. Cytoscape applies a bypass after every stylesheet rule, so no
# stylesheet (current or future) can pin nodes back to one fixed size.
VIZ_VERSION = "click-toggle-6"


def _with_size_bypass(builder):
    def wrapped(*args, **kwargs):
        elements = builder(*args, **kwargs)
        for element in elements:
            data = element.get("data", {})
            if "source" not in data and data.get("size") is not None:
                element["style"] = {"width": data["size"],
                                    "height": data["size"]}
        return elements
    wrapped.__name__ = builder.__name__
    wrapped.__doc__ = builder.__doc__
    return wrapped


edges_to_elements = _with_size_bypass(edges_to_elements)
comparison_elements = _with_size_bypass(comparison_elements)
two_slice_elements = _with_size_bypass(two_slice_elements)
higher_order_elements = _with_size_bypass(higher_order_elements)


def size_range_text(elements):
    """'100–200 px' for the nodes in elements, for the caption under a graph."""
    sizes = [e["data"]["size"] for e in elements
             if "source" not in e.get("data", {}) and "size" in e.get("data", {})]
    if not sizes:
        return ""
    return f" ({min(sizes):.0f}–{max(sizes):.0f} px)"


print(f"[viz] visualization.py loaded, version {VIZ_VERSION}", flush=True)


def structure_view(title, edges, element_id, fill, which):
    learned = [tuple(edge) for edge in (edges or [])]
    hidden_delay = delay_count(learned)
    edges = without_delay(learned)
    if not edges:
        return html.Div([html.H3(title),
                         html.P("Not available yet.", style={"color": MUTED})])

    nodes = {n for edge in edges for n in edge}
    iteration = field("iteration")
    if which == "local" and iteration:
        title = f"{title} (iteration {iteration})"

    order = field("dbn_order")
    lags = slices_present(edges)
    two_slice = two_slice_ready(edges)

    if two_slice:
        body_note = ("Slice (t) on the left, slice (t+1) on the right; every "
                     "variable appears in both. Dashed purple edges are the "
                     "inter-slice (temporal) dependencies. Click a node to "
                     "highlight its parents and children.")
    elif lags:
        deepest = slice_label(max(lags))
        body_note = (f"Higher-order DBN: one node per variable and time slice "
                     f"({deepest} … t), coloured by slice; static variables "
                     f"appear once. Dashed purple edges are the inter-slice "
                     f"(temporal) dependencies. Click a node to highlight its "
                     f"parents and children.")
    else:
        body_note = ("Click a node to highlight it with its parents and "
                     "children.")

    if two_slice:
        initial = two_slice_elements(edges)
        size_note = ("Node size grows with the number of connections"
                     + size_range_text(initial) + ".")
        graph = two_slice_canvas(cyto.Cytoscape(
            id=element_id,
            elements=initial,
            layout=TWO_SLICE_LAYOUT,
            stylesheet=two_slice_stylesheet(structure_stylesheet(fill)),
            style={"width": "100%", "height": "1000px",
                   "border": "1px solid #eef1f3", "borderRadius": "8px"},
        ))
        footer = html.Div(size_note,
                          style={"fontSize": "12px", "color": MUTED,
                                 "marginTop": "8px"})
    elif higher_order_ready(edges):
        initial = higher_order_elements(edges)
        size_note = ("Node size grows with the number of connections"
                     + size_range_text(initial) + ".")
        graph = cyto.Cytoscape(
            id=element_id,
            elements=initial,
            layout=HIGHER_ORDER_LAYOUT,
            stylesheet=two_slice_stylesheet(structure_stylesheet(fill)),
            style={"width": "100%", "height": "1000px",
                   "border": "1px solid #eef1f3", "borderRadius": "8px"},
        )
        footer = html.Div(size_note,
                          style={"fontSize": "12px", "color": MUTED,
                                 "marginTop": "8px"})
    else:
        initial = edges_to_elements(edges)
        size_note = ("Node size grows with the number of connections"
                     + size_range_text(initial) + ".")
        graph = cyto.Cytoscape(
            id=element_id,
            elements=initial,
            layout={"name": "dagre", "rankDir": "LR"},
            stylesheet=structure_stylesheet(fill),
            style={"width": "100%", "height": "820px",
                   "border": "1px solid #eef1f3", "borderRadius": "8px"},
        )
        footer = html.Div([
            html.Div([
                html.Span("— parent edge", style={"color": "#4a7fb5",
                                                  "fontSize": "12px",
                                                  "marginRight": "16px"}),
                html.Span("— child edge", style={"color": "#2f9e76",
                                                 "fontSize": "12px"}),
            ], style={"marginTop": "8px"}),
            html.Div(size_note,
                     style={"fontSize": "12px", "color": MUTED,
                            "marginTop": "6px"}),
        ])

    return html.Div([
        html.H3(title),
        html.Div([
            chip(f"{len(nodes)} nodes", "#eef1f3"),
            chip(f"{len(edges)} edges", "#eef1f3"),
            chip(f"{len(lags)} time slice{'s' if len(lags) != 1 else ''}",
                 "#eef1f3") if lags else None,
            chip(f"DBN order {order}", "#eef1f3") if order else None,
            chip(f"{hidden_delay} delay columns hidden", "#eef1f3")
            if hidden_delay else None,
        ], style={"marginBottom": "8px"}),
        network_legend(edges),
        html.Div([
            html.Span(body_note, style={"color": MUTED, "fontSize": "13px",
                                        "marginRight": "12px"}),
            html.Button("Clear selection", id=f"clear-{which}", n_clicks=0,
                        style={"fontSize": "12px", "height": "30px"}),
        ], style={"display": "flex", "alignItems": "center",
                  "marginBottom": "10px"}),
        html.Div(id=f"neighbourhood-{which}",
                 style={"fontSize": "13px", "minHeight": "24px",
                        "marginBottom": "8px"}),
        graph,
        footer,
    ])


def neighbourhood_summary(edges, picked):
    if not picked:
        return ""
    parents = sorted(node_label(s) for s, t in edges if t == picked)
    children = sorted(node_label(t) for s, t in edges if s == picked)
    return html.Span([
        html.Strong(node_label(picked)), " — ",
        html.Span(f"parents: {', '.join(parents) if parents else 'none'}",
                  style={"color": "#4a7fb5"}),
        "  |  ",
        html.Span(f"children: {', '.join(children) if children else 'none'}",
                  style={"color": "#2f9e76"}),
    ])


# ------------------------------------------------------------- what-if tab ---

# The Querying tab is a single centred column rather than a full-width sheet:
# it is a small form and a short answer, not a dashboard.
CENTRED = {"maxWidth": "760px", "margin": "0 auto", "padding": "0 12px"}


def model_ready():
    return (field("global_params") is not None
            and field("category_levels") is not None
            and field("dataset") is not None)


def levels_for(column):
    levels = (field("category_levels") or {}).get(column)
    if levels:
        return [str(level) for level in levels]
    frame = field("dataset")
    if frame is not None and column in frame.columns:
        return sorted(str(v) for v in frame[column].dropna().unique())
    return []


def predictable_nodes():
    """Nodes the aggregated model can actually score."""
    params = field("global_params") or {}
    return sorted(params.keys())


def baseline_row():
    """Most frequent value per column -- the starting point for a what-if."""
    frame = field("dataset")
    row = {}
    for column in frame.columns:
        values = frame[column].dropna()
        row[column] = str(values.mode().iloc[0]) if len(values) else None
    return row


def predict_distribution(node, evidence):
    """P(node | the chosen values) from the aggregated global parameters.

    Only the node's own parents enter the computation, so setting a variable
    that is not a parent leaves the answer unchanged -- the UI says so rather
    than letting it look broken.
    """
    import client as fedpam_client

    frame = field("dataset")
    params = field("global_params")
    row = baseline_row()
    row.update({k: v for k, v in evidence.items() if v is not None})

    single = pd.DataFrame([row])[list(frame.columns)]
    for column in single.columns:
        single[column] = pd.Categorical(
            single[column].astype(str), categories=levels_for(column))

    probabilities = fedpam_client.Client().predict_node_probability_from_beta(
        single, node, params)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1 - probabilities, probabilities])
    classes = [str(c) for c in params[node]["classes"]]
    return classes, probabilities[0]


def parents_of(node):
    return sorted({s for s, t in
                   [tuple(e) for e in (field("global_structure") or [])]
                   if t == node})


def whatif_view():
    if not model_ready():
        return html.Div([
            html.H3('Querying'),
            html.P("Available once the global network and its parameters have "
                   "been aggregated.", style={"color": MUTED}),
        ], style=CENTRED)

    nodes = predictable_nodes()
    default_target = field("target") if field("target") in nodes else (
        nodes[0] if nodes else None)
    frame = field("dataset")
    variables = [c for c in frame.columns if c != default_target]
    initial_result, initial_note = whatif_answer(default_target, {})

    return html.Div([
        html.H3('Querying', style={"textAlign": "center"}),
        html.P("Fix the values you care about and see how the predicted "
               "distribution of the chosen variable responds. Anything left "
               "blank is held at its most common value in this client's data.",
               style={"color": MUTED, "fontSize": "13px",
                      "textAlign": "center"}),

        html.Div([
            html.Div([
                html.Label("Predict", style={"fontSize": "12px",
                                             "color": MUTED}),
                dcc.Dropdown(id="whatif-node", options=nodes,
                             value=default_target, clearable=False),
            ], style={"flex": "1", "minWidth": "200px"}),
            html.Div([
                html.Label("Set values for", style={"fontSize": "12px",
                                                    "color": MUTED}),
                dcc.Dropdown(id="whatif-vars", options=variables, value=[],
                             multi=True, placeholder="pick any variables"),
            ], style={"flex": "2", "minWidth": "260px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "marginBottom": "14px"}),

        html.Div(id="whatif-controls",
                 style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                        "justifyContent": "center", "marginBottom": "16px"}),
        html.Div(id="whatif-note", children=initial_note,
                 style={"fontSize": "13px", "marginBottom": "16px",
                        "textAlign": "center"}),
        html.Div(id="whatif-result", children=initial_result),
    ], style=CENTRED)


@callback(
    Output("whatif-controls", "children"),
    Input("whatif-vars", "value"),
    prevent_initial_call=True,
)
def whatif_controls(selected):
    if not selected:
        return []
    return [html.Div([
        html.Label(variable, style={"fontSize": "12px", "color": MUTED}),
        dcc.Dropdown(id={"type": "whatif-value", "variable": variable},
                     options=levels_for(variable),
                     placeholder="any", clearable=True),
    ], style={"minWidth": "170px"}) for variable in selected]


@callback(
    Output("whatif-result", "children"),
    Output("whatif-note", "children"),
    Input("whatif-node", "value"),
    Input({"type": "whatif-value", "variable": ALL}, "value"),
    State({"type": "whatif-value", "variable": ALL}, "id"),
    # Must stay True: "whatif-node" only exists once the Querying tab has been
    # rendered. With prevent_initial_call=False this callback fires on page
    # load against a missing Input, the renderer never resolves it, and the
    # whole dashboard sits at "Updating..." with no tab content at all.
    prevent_initial_call=True,
)
def whatif_predict(node, values, ids):
    if not node or not model_ready():
        raise PreventUpdate
    evidence = {identifier["variable"]: value
                for identifier, value in zip(ids or [], values or [])
                if value is not None}
    return whatif_answer(node, evidence)


def whatif_answer(node, evidence):
    """Shared by the callback and the tab's first render.

    The tab renders its own initial answer rather than relying on a callback
    firing for a component that does not exist yet -- see whatif_view.
    """
    if not node or not model_ready():
        return html.Div(), ""
    try:
        classes, probabilities = predict_distribution(node, evidence)
    except Exception as error:
        return (html.Div(f"Could not score this configuration: {error}",
                         style={"color": "#8a3d2e"}), "")

    order = np.argsort(probabilities)[::-1]
    top = classes[order[0]]

    rows = []
    for rank, index in enumerate(order):
        probability = float(probabilities[index])
        leading = rank == 0
        rows.append(html.Div([
            html.Div(classes[index],
                     style={"fontSize": "14px", "fontWeight": 600 if leading else 400,
                            "color": INK if leading else MUTED}),
            html.Div(f"{probability:.4f}",
                     style={"fontSize": "15px", "fontFamily": "monospace",
                            "color": INK, "textAlign": "right",
                            "minWidth": "90px"}),
            html.Div(f"{probability:.1%}",
                     style={"fontSize": "13px", "color": MUTED,
                            "textAlign": "right", "minWidth": "70px"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "gap": "16px",
                  "padding": "12px 16px",
                  "borderBottom": "1px solid #eef1f3",
                  "background": "#f6fbf9" if leading else "transparent"}))

    cards = html.Div(rows, style={"border": "1px solid #eef1f3",
                                  "borderRadius": "8px", "overflow": "hidden",
                                  "marginBottom": "6px"})

    # Only the node's parents can move the answer; say so when the user has set
    # something that cannot.
    relevant = set(parents_of(node))
    inert = [v for v in evidence if v not in relevant]
    if inert:
        note = html.Span(
            f"{', '.join(inert)} {'is' if len(inert) == 1 else 'are'} not a "
            f"parent of {node} in the global network, so {'it has' if len(inert) == 1 else 'they have'} "
            f"no effect on this prediction. Parents of {node}: "
            f"{', '.join(relevant) if relevant else 'none'}.",
            style={"color": "#8a3d2e"})
    else:
        note = html.Span(
            f"Parents of {node}: {', '.join(relevant) if relevant else 'none'}."
            + ("" if relevant else " With no parents, the prediction is the "
                                   "marginal distribution and will not change."),
            style={"color": MUTED})

    return html.Div([
        html.Div([
            html.Div("Most likely", style={"fontSize": "11px", "color": MUTED,
                                           "textTransform": "uppercase",
                                           "letterSpacing": "0.04em"}),
            html.Div(top, style={"fontSize": "32px", "fontWeight": 600,
                                 "color": INK}),
            html.Div(f"{float(probabilities[order[0]]):.1%} probability",
                     style={"fontSize": "13px", "color": MUTED}),
        ], style={"textAlign": "center", "padding": "18px",
                  "border": f"2px solid {ACCENT}", "borderRadius": "8px",
                  "background": "#fbfcfc", "marginBottom": "18px"}),
        cards,
    ]), note


# ------------------------------------------------------- validation editor ---

EDITOR_STYLESHEET = [
    {"selector": "node", "style": {
        "label": "data(label)", "background-color": "#ffffff",
        "width": "data(size)", "height": "data(size)",
        "font-size": "11px", "text-valign": "center", "color": INK,
        "border-width": NODE_BORDER_WIDTH, "border-color": NODE_BORDER,
    }},
    {"selector": "node.pending", "style": {
        "background-color": "#ffffff", "border-width": 3,
        "border-color": "#e0a458"}},
    {"selector": "edge", "style": {
        "curve-style": "bezier", "target-arrow-shape": "triangle", "width": 2}},
    # Edges from the learned network that the user has kept.
    {"selector": "edge.kept", "style": {
        "line-color": "#9aa5b1", "target-arrow-color": "#9aa5b1"}},
    # Removed -> blacklist. Kept visible and clickable so a removal is undoable.
    {"selector": "edge.blacklisted", "style": {
        "line-color": "#d98b7a", "target-arrow-color": "#d98b7a",
        "line-style": "dotted", "width": 3}},
    # Added by hand -> whitelist.
    {"selector": "edge.whitelisted", "style": {
        "line-color": ACCENT, "target-arrow-color": ACCENT,
        "line-style": "dashed", "width": 3}},
    *slice_rules(),
    {"selector": "node.target", "style": {"background-color": TARGET_FILL}},
]


def as_pairs(items):
    """Normalise stored edges to a list of (source, target) tuples."""
    return [tuple(item) for item in (items or [])]


def resulting_edges(base, blacklist, whitelist):
    kept = [edge for edge in base if edge not in set(blacklist)]
    return kept + [edge for edge in whitelist if edge not in set(kept)]


def constraints_document(base, blacklist, whitelist):
    """The JSON the user downloads: the two constraint lists plus the network
    they produce, so a follow-up run can be seeded directly from this file."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_network": "final global DAG",
        "blacklist": [{"source": s, "target": t} for s, t in blacklist],
        "whitelist": [{"source": s, "target": t} for s, t in whitelist],
        "resulting_network": [
            {"source": s, "target": t}
            for s, t in resulting_edges(base, blacklist, whitelist)
        ],
        "counts": {
            "original_edges": len(base),
            "blacklisted": len(blacklist),
            "whitelisted": len(whitelist),
            "resulting_edges": len(resulting_edges(base, blacklist, whitelist)),
        },
    }


def editor_elements(base, blacklist, whitelist, pending):
    if two_slice_ready(base):
        # Draw the 2TBN and colour each edge by its curation state. The learned
        # endpoints ride along in the edge data, so a tap still resolves to one
        # blacklist entry even though an intra edge appears in both slices.
        blacklist_set, whitelist_set = set(blacklist), set(whitelist)
        shown = list(base) + [e for e in whitelist if e not in set(base)]

        def state(learned):
            pair = tuple(learned)
            if pair in blacklist_set:
                return "blacklisted"
            if pair in whitelist_set:
                return "whitelisted"
            return "kept"

        return two_slice_elements(shown, pending, edge_state=state)

    if higher_order_ready(base):
        # Same template as the structure tabs. Node ids are the learned
        # column names, so node clicks and edge clicks map straight back to
        # the blacklist/whitelist entries.
        blacklist_set, whitelist_set = set(blacklist), set(whitelist)
        shown = list(base) + [e for e in whitelist if e not in set(base)]

        def state(learned):
            pair = tuple(learned)
            if pair in blacklist_set:
                return "blacklisted"
            if pair in whitelist_set:
                return "whitelisted"
            return "kept"

        return higher_order_elements(shown, pending, edge_state=state)

    """Every edge stays on the canvas -- removed ones just change class -- so
    the graph doubles as the undo surface."""
    blacklist_set, whitelist_set = set(blacklist), set(whitelist)
    nodes = {n for edge in list(base) + list(whitelist) for n in edge}

    # Sized on the curated network rather than the drawn one, so blacklisting an
    # edge visibly shrinks the nodes it connected and whitelisting grows them.
    sizes = node_sizes(resulting_edges(base, blacklist, whitelist))
    elements = [{"data": {"id": n, "label": node_label(n),
                          "size": sizes.get(n, NODE_MIN_SIZE)},
                 "classes": node_classes(n, "pending" if n == pending else "")}
                for n in sorted(nodes)]

    for source, target in base:
        elements.append({
            "data": {"source": source, "target": target,
                     "id": f"{source}|{target}"},
            "classes": "blacklisted" if (source, target) in blacklist_set else "kept",
        })
    for source, target in whitelist:
        if (source, target) not in set(base):
            elements.append({
                "data": {"source": source, "target": target,
                         "id": f"{source}|{target}"},
                "classes": "whitelisted",
            })
    return elements


editor_elements = _with_size_bypass(editor_elements)


# Curation colours for the 2TBN editor: kept edges keep their slice colour,
# removed and added ones are recoloured, widened to match the 2TBN's scale.
EDITOR_STATE_RULES_TBN = [
    {"selector": "edge.blacklisted", "style": {
        "line-color": "#d98b7a", "target-arrow-color": "#d98b7a",
        "line-style": "dotted", "width": 7}},
    {"selector": "edge.whitelisted", "style": {
        "line-color": ACCENT, "target-arrow-color": ACCENT,
        "line-style": "dashed", "width": 7}},
]


def validation_canvas(base):
    if higher_order_ready(base):
        return cyto.Cytoscape(
            id='validation-graph',
            elements=editor_elements(base, [], [], None),
            layout=HIGHER_ORDER_LAYOUT,
            stylesheet=two_slice_stylesheet(EDITOR_STYLESHEET,
                                            EDITOR_STATE_RULES_TBN),
            style={"width": "100%", "height": "700px",
                   "border": "1px solid #eef1f3", "borderRadius": "8px"},
        )
    if two_slice_ready(base):
        return two_slice_canvas(cyto.Cytoscape(
            id='validation-graph',
            elements=editor_elements(base, [], [], None),
            layout=TWO_SLICE_LAYOUT,
            stylesheet=two_slice_stylesheet(EDITOR_STYLESHEET,
                                            EDITOR_STATE_RULES_TBN),
            style={"width": "100%", "height": "700px",
                   "border": "1px solid #eef1f3", "borderRadius": "8px"},
        ))
    return cyto.Cytoscape(
        id='validation-graph',
        elements=editor_elements(base, [], [], None),
        layout={"name": "dagre", "rankDir": "LR"},
        stylesheet=EDITOR_STYLESHEET + [target_rule()],
        style={"width": "100%", "height": "520px",
               "border": "1px solid #eef1f3", "borderRadius": "8px"},
    )


def validation_view():
    base = without_delay(as_pairs(field("global_structure")))
    if not base:
        return html.Div([
            html.H3('Validation'),
            html.P("The final global network has not been aggregated yet.",
                   style={"color": MUTED}),
        ])

    nodes = sorted({n for edge in base for n in edge})
    return html.Div([
        html.H3('Validation'),
        html.P("Curate the learned network with domain knowledge. Click an edge "
               "to remove it; removed edges become the blacklist. Add edges that "
               "should have been found; those become the whitelist. Click a "
               "removed or added edge again to undo it.",
               style={"color": MUTED, "fontSize": "13px"}),

        html.Div([
            html.Span("— kept", style={"color": "#9aa5b1", "marginRight": "16px",
                                       "fontSize": "12px"}),
            html.Span("··· blacklisted (removed)",
                      style={"color": "#d98b7a", "marginRight": "16px",
                             "fontSize": "12px"}),
            html.Span("--- whitelisted (added)",
                      style={"color": ACCENT, "fontSize": "12px",
                             "marginRight": "16px"}),
            html.Span("Node size follows connectivity in the curated network.",
                      style={"color": MUTED, "fontSize": "12px"}),
        ], style={"marginBottom": "10px"}),
        # The editor stays on the learned node names: in the unrolled view an
        # intra-slice edge appears twice, so a click would not map back to one
        # blacklist entry. Slice colour carries the temporal information here.
        network_legend(base),

        validation_canvas(base),

        html.Div([
            html.Div([
                html.Label("From", style={"fontSize": "12px", "color": MUTED}),
                dcc.Dropdown(id='validation-source', options=nodes,
                             placeholder="source"),
            ], style={"flex": "1", "minWidth": "160px"}),
            html.Div([
                html.Label("To", style={"fontSize": "12px", "color": MUTED}),
                dcc.Dropdown(id='validation-target', options=nodes,
                             placeholder="target"),
            ], style={"flex": "1", "minWidth": "160px"}),
            html.Button("Add edge", id='validation-add-btn', n_clicks=0,
                        style={"backgroundColor": ACCENT, "color": "white",
                               "border": "none", "borderRadius": "6px",
                               "height": "38px", "marginTop": "18px"}),
            html.Button("Reset", id='validation-reset-btn', n_clicks=0,
                        style={"height": "38px", "marginTop": "18px"}),
        ], style={"display": "flex", "gap": "12px", "alignItems": "flex-start",
                  "marginTop": "16px", "flexWrap": "wrap"}),

        html.Div(id='validation-message',
                 style={"minHeight": "22px", "marginTop": "10px",
                        "fontSize": "13px"}),

        html.H4("Constraints", style={"color": INK, "marginTop": "22px"}),
        html.Div(id='validation-summary', style={"marginBottom": "10px"}),
        html.Pre(id='validation-json',
                 style={"background": "#f8fafb", "border": "1px solid #eef1f3",
                        "borderRadius": "8px", "padding": "14px",
                        "fontSize": "12px", "maxHeight": "320px",
                        "overflowY": "auto", "whiteSpace": "pre-wrap"}),

        html.Div([
            html.Button("Download JSON", id='validation-download-btn', n_clicks=0,
                        style={"backgroundColor": ACCENT, "color": "white",
                               "border": "none", "borderRadius": "6px",
                               "padding": "0 20px", "height": "38px"}),
            html.Button("Save to /mnt/output", id='validation-save-btn',
                        n_clicks=0, style={"height": "38px"}),
        ], style={"display": "flex", "gap": "12px", "marginTop": "14px"}),
        html.Div(id='validation-save-msg',
                 style={"fontSize": "12px", "color": MUTED, "marginTop": "8px"}),
    ])


# ------------------------------------------------------------------ layout ---

app.layout = html.Div([
    html.Div([
        html.H1('FedDy-PAM Dashboard',
                style={"display": "inline-block", "color": INK}),
        html.Div([
            html.Span(id='finish-status',
                      style={"fontSize": "13px", "color": MUTED,
                             "marginRight": "12px"}),
            html.Button('Finish', id='finish-btn', n_clicks=0),
        ], style={"float": "right", "display": "flex", "alignItems": "center",
                  "marginTop": "26px"}),
    ], style={"overflow": "hidden"}),

    dcc.Tabs(id="results-tabs", value='dataset', children=[
        dcc.Tab(label='Dataset',          value='dataset'),
        dcc.Tab(label='Local Structure',  value='local-structure'),
        dcc.Tab(label='Global Structure', value='global-structure'),
        dcc.Tab(label='Evaluation',       value='evaluation'),
        dcc.Tab(label='Validation',       value='validation'),
        dcc.Tab(label='Querying',         value='querying'),
    ]),
    dcc.Store(id="last-clicked-node", data=None),
    # Change tokens written by the poller. Both live OUTSIDE dcc.Loading, so
    # updating them never puts the spinner on screen.
    dcc.Store(id="content-token", data=None),
    dcc.Store(id="finish-token", data=None),
    dcc.Store(id="finish-sink", data=None),
    # Validation edits live at the top level so switching tabs does not discard
    # them. They are per-browser, not shared with the state machine.
    dcc.Store(id="validation-edits", data={"blacklist": [], "whitelist": []}),
    dcc.Store(id="validation-pending", data=None),
    dcc.Download(id="validation-download"),
    # Polls the store so tabs fill in as the federated run progresses instead of
    # staying empty until a manual refresh.
    dcc.Interval(id="refresh", interval=2000, n_intervals=0),
    dcc.Loading(
        id="loading-tab-content",
        type="circle",
        color=ACCENT,
        # Only show the spinner if a render genuinely takes a while. Without
        # this, every sub-second callback flashes the whole panel.
        delay_show=500,
        delay_hide=0,
        style={"marginTop": "80px"},
        children=html.Div(id='results-tabs-content')
    )
])


# Which store fields each tab actually displays. The poller uses this to decide
# whether the visible tab needs rebuilding at all.
TAB_FIELDS = {
    'dataset': ("dataset", "dataset_path", "dataset_csv_path",
                "config_path", "client_id"),
    'local-structure': ("local_structure", "iteration", "target"),
    'querying': ("global_params", "category_levels", "target", "dataset"),
    'global-structure': ("global_structure", "target"),
    'evaluation': ("evaluation", "all_evaluations", "predictions",
                   "structure_metrics", "first_local_structure",
                   "global_structure", "has_target", "testing_enabled",
                   "target", "is_coordinator"),
    'validation': ("validation", "global_structure", "target"),
}


# ---------------------------------------------------------------- callbacks ---

@callback(
    Output('content-token', 'data'),
    Output('finish-token', 'data'),
    Input('refresh', 'n_intervals'),
    Input('results-tabs', 'value'),
    State('content-token', 'data'),
    State('finish-token', 'data'),
)
def poll_store(_n_intervals, tab, content_token, finish_token):
    """The only thing the timer touches.

    It compares cheap change tokens and writes nothing when nothing moved, so
    on an idle tick no downstream callback runs at all -- which is what stops
    the dashboard flashing every few seconds. Neither output sits inside
    dcc.Loading, so writing them never shows a spinner either.
    """
    new_content = [tab] + list(store.revision(*TAB_FIELDS.get(tab, ()), "error"))
    # The finish controls must never be able to break tab switching: this one
    # callback is the only thing that moves the content token, so if it raises,
    # the whole dashboard freezes on whatever tab is open.
    try:
        new_finish = list(finish_signature())
    except Exception:
        new_finish = finish_token

    content_out = no_update if new_content == content_token else new_content
    finish_out = no_update if new_finish == finish_token else new_finish

    if content_out is no_update and finish_out is no_update:
        raise PreventUpdate
    return content_out, finish_out


@callback(
    Output('results-tabs-content', 'children'),
    Input('content-token', 'data'),
    State('results-tabs', 'value'),
    prevent_initial_call=True,
)
def render_content(_token, tab):
    body = _tab_body(tab)
    banner = error_banner()
    return html.Div([banner, body]) if banner else body


def _tab_body(tab):

    if tab == 'dataset':
        return dataset_view()

    if tab == 'local-structure':
        return structure_view('Local Structure', field("local_structure"),
                              "cyto-local", LOCAL_FILL, "local")

    if tab == 'global-structure':
        return structure_view('Global Structure', field("global_structure"),
                              "cyto-global", GLOBAL_FILL, "global")

    if tab == 'evaluation':
        return evaluation_view()

    if tab == 'validation':
        return validation_view()

    if tab == 'querying':
        return whatif_view()

    return html.Div()


def finish_signature():
    """Everything render_finish_controls depends on, as a comparable tuple.

    poll_store calls this on every tick, so it must exist for the poller -- and
    therefore every tab switch -- to work at all.
    """
    return store.revision("is_coordinator", "current_state",
                          "finish_clicked", "finish_signalled")


@callback(
    Output('finish-btn', 'style'),
    Output('finish-btn', 'disabled'),
    Output('finish-status', 'children'),
    Input('finish-token', 'data'),
)
def render_finish_controls(_token):
    """Only the coordinator gets a Finish button; it ends the run for everyone."""
    hidden = {"display": "none"}
    shown = {"backgroundColor": ACCENT, "color": "white", "border": "none",
             "borderRadius": "6px", "padding": "0 22px", "height": "38px",
             "fontSize": "14px", "fontWeight": 600, "letterSpacing": "0.01em",
             "textTransform": "none", "cursor": "pointer",
             "boxShadow": "0 1px 3px rgba(18, 50, 42, 0.25)",
             "transition": "filter 120ms ease"}
    # Greyed out, flat and with a no-entry cursor while there is nothing to do.
    faded = dict(shown, backgroundColor="#c8d0d6", color="#f4f6f7",
                 cursor="not-allowed", boxShadow="none", fontWeight=500)

    if ENV != "fc" or store.is_coordinator is None:
        return hidden, True, ""

    if not store.is_coordinator:
        if store.finish_signalled:
            return hidden, True, "Coordinator ended the workflow. Shutting down."
        if store.current_state == 'visualize':
            return hidden, True, ("Results are final. Waiting for the coordinator "
                                  "to end the workflow.")
        return hidden, True, "Workflow running."

    if store.finish_clicked:
        return faded, True, "Ending the workflow for all clients…"
    if store.current_state != 'visualize':
        return faded, True, "Results still computing."
    return shown, False, "Ends the workflow for every client."


def _structure_click(which, edges, picked):
    edges = without_delay(edges)
    if not edges:
        raise PreventUpdate
    trigger = callback_context.triggered_id
    if trigger == f"clear-{which}":
        picked = None
    node = picked.get("id") if isinstance(picked, dict) else picked
    if two_slice_ready(edges):
        elements = two_slice_elements(edges, node)
        if not node:
            return elements, ""
        parents = [slice_of_id(n) for n in TWO_SLICE_NEIGHBOURS.get("parents", [])]
        children = [slice_of_id(n) for n in TWO_SLICE_NEIGHBOURS.get("children", [])]
        summary = html.Span([
            html.Strong(slice_of_id(node)), " — ",
            html.Span(f"parents: {', '.join(parents) if parents else 'none'}",
                      style={"color": "#4a7fb5"}),
            "  |  ",
            html.Span(f"children: {', '.join(children) if children else 'none'}",
                      style={"color": "#2f9e76"}),
        ])
        return elements, summary
    if higher_order_ready(edges):
        return (higher_order_elements(edges, node),
                neighbourhood_summary(edges, node))
    return edges_to_elements(edges, node), neighbourhood_summary(edges, node)


# Clicks come in through `tapNode` rather than `tapNodeData`: tapNode carries
# a timestamp, so tapping the same node twice is two distinct events. With
# tapNodeData the second tap on a node sends identical data, Dash sees no
# change, and the node could never be clicked off again.

def _classes_of(value):
    if isinstance(value, str):
        return value.split()
    return list(value or [])


def toggled_pick(tap, elements=None):
    """The node to highlight after a tap: the tapped node, or none if it was
    already the highlighted one (a second click unselects it).

    "Already highlighted" is read from the tap event itself -- it reports the
    tapped node's classes at the moment of the click. The graph's `elements`
    prop is only a fallback: after a callback redraws a preset-position graph
    (the 2TBN), Dash can hand the elements back without their classes.
    """
    tap = tap or {}
    node = (tap.get("data") or {}).get("id")
    if not node:
        return None
    if "classes" in tap:
        return None if "picked" in _classes_of(tap.get("classes")) else node
    for element in elements or []:
        if element.get("data", {}).get("id") == node:
            return None if "picked" in _classes_of(element.get("classes")) else node
    return node


@callback(
    Output('cyto-local', 'elements'),
    Output('neighbourhood-local', 'children'),
    Input('cyto-local', 'tapNode'),
    Input('clear-local', 'n_clicks'),
    State('cyto-local', 'elements'),
    prevent_initial_call=True,
)
def highlight_local(tap, _clear, elements):
    return _structure_click("local", field("local_structure"),
                            toggled_pick(tap, elements))


@callback(
    Output('cyto-global', 'elements'),
    Output('neighbourhood-global', 'children'),
    Input('cyto-global', 'tapNode'),
    Input('clear-global', 'n_clicks'),
    State('cyto-global', 'elements'),
    prevent_initial_call=True,
)
def highlight_global(tap, _clear, elements):
    return _structure_click("global", field("global_structure"),
                            toggled_pick(tap, elements))


@callback(
    Output('cyto-eval-initial', 'elements'),
    Input('cyto-eval-initial', 'tapNode'),
    State('cyto-eval-initial', 'elements'),
    prevent_initial_call=True,
)
def highlight_eval_initial(tap, elements):
    initial = without_delay(field("first_local_structure"))
    final = without_delay(field("global_structure"))
    if not initial:
        raise PreventUpdate
    return panel_elements(initial, final, toggled_pick(tap, elements))


@callback(
    Output('cyto-eval-final', 'elements'),
    Input('cyto-eval-final', 'tapNode'),
    State('cyto-eval-final', 'elements'),
    prevent_initial_call=True,
)
def highlight_eval_final(tap, elements):
    initial = without_delay(field("first_local_structure"))
    final = without_delay(field("global_structure"))
    if not final:
        raise PreventUpdate
    return panel_elements(final, initial, toggled_pick(tap, elements))


@callback(
    Output('validation-edits', 'data'),
    Output('validation-pending', 'data'),
    Output('validation-message', 'children'),
    Input('validation-graph', 'tapEdge'),
    Input('validation-graph', 'tapNode'),
    Input('validation-add-btn', 'n_clicks'),
    Input('validation-reset-btn', 'n_clicks'),
    State('validation-source', 'value'),
    State('validation-target', 'value'),
    State('validation-edits', 'data'),
    State('validation-pending', 'data'),
    prevent_initial_call=True,
)
def edit_network(edge_tap, node_tap, _add, _reset, source, target,
                 edits, pending):
    # tapEdge/tapNode (not *Data) so a second click on the same edge or node
    # is a new event: that is what makes click-again-to-undo and
    # click-again-to-unselect work.
    trigger_prop = (callback_context.triggered[0]["prop_id"].split(".")[-1]
                    if callback_context.triggered else "")
    edge_data = (edge_tap or {}).get("data") if trigger_prop == "tapEdge" else None
    node_data = (node_tap or {}).get("data") if trigger_prop == "tapNode" else None
    trigger = callback_context.triggered_id
    base = without_delay(as_pairs(field("global_structure")))
    blacklist = as_pairs((edits or {}).get("blacklist"))
    whitelist = as_pairs((edits or {}).get("whitelist"))

    def ok(message):
        return ({"blacklist": [list(e) for e in blacklist],
                 "whitelist": [list(e) for e in whitelist]},
                None,
                html.Span(message, style={"color": INK}))

    def reject(message):
        return no_update, no_update, html.Span(message, style={"color": "#8a3d2e"})

    if trigger == 'validation-reset-btn':
        return ({"blacklist": [], "whitelist": []}, None,
                html.Span("Reset to the learned network.", style={"color": MUTED}))

    # --- click an edge: remove it, or undo a previous edit ------------------
    if trigger == 'validation-graph' and edge_data:
        edge = (edge_data.get("learned_source") or edge_data.get("source"),
                edge_data.get("learned_target") or edge_data.get("target"))
        if edge in whitelist:
            whitelist.remove(edge)
            return ok(f"Removed the added edge {edge[0]} → {edge[1]}.")
        if edge in blacklist:
            blacklist.remove(edge)
            return ok(f"Restored {edge[0]} → {edge[1]}.")
        if edge in base:
            blacklist.append(edge)
            return ok(f"Blacklisted {edge[0]} → {edge[1]}. Click it again to undo.")
        return no_update, no_update, no_update

    # --- click two nodes: add an edge between them --------------------------
    if trigger == 'validation-graph' and node_data:
        node = node_data.get("id")
        if pending is None:
            return no_update, node, html.Span(
                f"{node} selected as source. Click a target node, or click "
                f"{node} again to cancel.", style={"color": MUTED})
        if pending == node:
            return no_update, None, html.Span("Selection cancelled.",
                                              style={"color": MUTED})
        candidate = (pending, node)
        message = add_edge(base, blacklist, whitelist, candidate)
        if message.startswith("Added") or message.startswith("Restored"):
            return ok(message)
        return reject(message)

    # --- add via the dropdowns ---------------------------------------------
    if trigger == 'validation-add-btn':
        if not source or not target:
            return reject("Pick both a source and a target first.")
        message = add_edge(base, blacklist, whitelist, (source, target))
        if message.startswith("Added") or message.startswith("Restored"):
            return ok(message)
        return reject(message)

    raise PreventUpdate


def add_edge(base, blacklist, whitelist, candidate):
    """Validate and apply an edge addition. Mutates blacklist/whitelist."""
    source, target = candidate
    if source == target:
        return "An edge cannot start and end at the same node."
    if candidate in whitelist:
        return f"{source} → {target} has already been added."
    if candidate in base and candidate not in blacklist:
        return f"{source} → {target} is already in the network."
    if (target, source) in resulting_edges(base, blacklist, whitelist):
        return (f"{target} → {source} already exists; a Bayesian network cannot "
                f"hold both directions.")

    # Time only runs one way: a node in slice t cannot cause one at t-1.
    source_lag, target_lag = slice_of(source), slice_of(target)
    if source_lag is not None and target_lag is not None and source_lag < target_lag:
        return (f"{source} is in slice {slice_label(source_lag)} and {target} in "
                f"{slice_label(target_lag)}; an edge cannot point backwards in "
                f"time.")

    # A DAG has to stay acyclic, so reject anything that closes a loop.
    graph = nx.DiGraph(resulting_edges(base, blacklist, whitelist))
    graph.add_edge(source, target)
    if not nx.is_directed_acyclic_graph(graph):
        return (f"{source} → {target} would create a cycle, which is not "
                f"allowed in a DAG.")

    if candidate in blacklist:
        blacklist.remove(candidate)
        return f"Restored {source} → {target}."
    whitelist.append(candidate)
    return f"Added {source} → {target} to the whitelist."


@callback(
    Output('validation-graph', 'elements'),
    Output('validation-json', 'children'),
    Output('validation-summary', 'children'),
    Output('validation-download-btn', 'disabled'),
    Output('validation-save-btn', 'disabled'),
    Input('validation-edits', 'data'),
    Input('validation-pending', 'data'),
)
def render_editor(edits, pending):
    base = without_delay(as_pairs(field("global_structure")))
    blacklist = as_pairs((edits or {}).get("blacklist"))
    whitelist = as_pairs((edits or {}).get("whitelist"))

    document = constraints_document(base, blacklist, whitelist)
    counts = document["counts"]
    summary = html.Div([
        chip(f"{counts['original_edges']} learned", "#eef1f3"),
        chip(f"{counts['blacklisted']} blacklisted", "#fdf1ee"),
        chip(f"{counts['whitelisted']} whitelisted", "#eaf8f2"),
        chip(f"{counts['resulting_edges']} in the curated network", "#e8ecef"),
    ])
    unedited = not blacklist and not whitelist
    return (editor_elements(base, blacklist, whitelist, pending),
            json.dumps(document, indent=2),
            summary,
            unedited, unedited)


@callback(
    Output('validation-download', 'data'),
    Input('validation-download-btn', 'n_clicks'),
    State('validation-edits', 'data'),
    prevent_initial_call=True,
)
def download_constraints(n_clicks, edits):
    # Dash re-invokes callbacks for components newly added to the layout, so
    # merely opening the tab fired this one and popped a save dialog. Nothing
    # is sent until the button is really pressed and there is an edit to save.
    if not n_clicks:
        raise PreventUpdate
    edits = edits or {}
    if not edits.get("blacklist") and not edits.get("whitelist"):
        raise PreventUpdate

    base = without_delay(as_pairs(field("global_structure")))
    document = constraints_document(base,
                                    as_pairs((edits or {}).get("blacklist")),
                                    as_pairs((edits or {}).get("whitelist")))
    return dict(content=json.dumps(document, indent=2),
                filename="edge_constraints.json")


@callback(
    Output('validation-save-msg', 'children'),
    Input('validation-save-btn', 'n_clicks'),
    State('validation-edits', 'data'),
    prevent_initial_call=True,
)
def save_constraints(n_clicks, edits):
    """Fallback for when the browser blocks downloads from the embedded frame.

    /mnt/output is collected by FeatureCloud when the container stops, so the
    file comes back with the rest of the results.
    """
    if not n_clicks:
        raise PreventUpdate
    edits = edits or {}
    if not edits.get("blacklist") and not edits.get("whitelist"):
        raise PreventUpdate

    base = without_delay(as_pairs(field("global_structure")))
    document = constraints_document(base,
                                    as_pairs((edits or {}).get("blacklist")),
                                    as_pairs((edits or {}).get("whitelist")))
    path = os.path.join(os.getenv("OUTPUT_DIR", "/mnt/output"),
                        "edge_constraints.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(document, handle, indent=2)
    except Exception as error:
        return html.Span(f"Could not write {path}: {error}",
                         style={"color": "#8a3d2e"})
    return f"Saved to {path} — collected with the run's results."


@callback(
    Output('finish-sink', 'data'),
    Input('finish-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def finish_workflow(n_clicks):
    # Guarded server-side as well as in the UI: only the coordinator's click
    # may end the run, whatever a participant's browser sends.
    if n_clicks and store.is_coordinator:
        store.update(finish_clicked=True)
    return n_clicks

@callback(
    Output('predictions-full', 'data'),
    Input('predictions-full', 'page_current'),
    Input('predictions-full', 'page_size'),
    prevent_initial_call=True,
)
def predictions_full_page(page_current, page_size):
    frame = field("predictions")
    if frame is None or len(frame) == 0:
        raise PreventUpdate
    display = frame.rename(columns={
        "row": "Row", "fold": "Fold", "true": "True class",
        "predicted": "Predicted", "confidence": "Confidence",
        "correct": "Correct"})
    display["Confidence"] = display["Confidence"].map(lambda v: f"{v:.3f}")
    display["Correct"] = display["Correct"].map({True: "yes", False: "no"})
    if "predictions" == "misclassified":
        display = display[display["Correct"] == "no"]
    size = page_size or DATASET_PAGE_SIZE
    start = (page_current or 0) * size
    return display.iloc[start:start + size].to_dict('records')


@callback(
    Output('predictions-wrong', 'data'),
    Input('predictions-wrong', 'page_current'),
    Input('predictions-wrong', 'page_size'),
    prevent_initial_call=True,
)
def predictions_wrong_page(page_current, page_size):
    frame = field("predictions")
    if frame is None or len(frame) == 0:
        raise PreventUpdate
    display = frame.rename(columns={
        "row": "Row", "fold": "Fold", "true": "True class",
        "predicted": "Predicted", "confidence": "Confidence",
        "correct": "Correct"})
    display["Confidence"] = display["Confidence"].map(lambda v: f"{v:.3f}")
    display["Correct"] = display["Correct"].map({True: "yes", False: "no"})
    if "misclassified" == "misclassified":
        display = display[display["Correct"] == "no"]
    size = page_size or DATASET_PAGE_SIZE
    start = (page_current or 0) * size
    return display.iloc[start:start + size].to_dict('records')