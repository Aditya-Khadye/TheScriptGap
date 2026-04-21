import pandas as pd
import numpy as np
from pathlib import Path
import re
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
from sklearn.preprocessing import QuantileTransformer


def cleanup_font_name(font_name: str) -> str:
    """Remove font variation tags like [wdth,wght] or [wght] from font names."""
    return re.sub(r'\[[\w,]+\]$', '', font_name).strip()


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and clean font names."""
    df = pd.read_csv(csv_path)
    df['source'] = df['source'].apply(cleanup_font_name)
    df['target'] = df['target'].apply(cleanup_font_name)
    return df


def create_network_figure(df: pd.DataFrame, threshold_val: float, pos: dict) -> go.Figure:
    """Create a network figure for a given threshold with fixed node positions."""
    G = nx.Graph()

    for _, row in df.iterrows():
        weight = row['similarity_normalized']
        G.add_edge(row['source'], row['target'], weight=weight)

    # Filter edges based on threshold
    filtered_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= threshold_val]
    G_filtered = G.edge_subgraph(filtered_edges).copy()
    G_filtered.add_nodes_from(G.nodes())

    if G_filtered.number_of_nodes() == 0:
        return go.Figure().add_annotation(text="No edges above this threshold")

    # Compute global 25th/75th percentile cutoffs from the full edge set
    all_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    q25 = np.percentile(all_weights, 25)
    q75 = np.percentile(all_weights, 75)

    def edge_color(w):
        if w < q25:
            return 'rgba(220, 50, 50, 0.5)'    # red — bottom 25%
        elif w < q75:
            return 'rgba(230, 150, 30, 0.5)'   # orange — middle 50%
        else:
            return 'rgba(50, 180, 80, 0.5)'    # green — top 25%

    def edge_label(w):
        if w < q25:
            return 'Low'
        elif w < q75:
            return 'Medium'
        else:
            return 'High'

    def node_color(w):
        if w >= q75:
            return 'rgba(50, 180, 80, 0.9)'
        elif w >= q25:
            return 'rgba(230, 150, 30, 0.9)'
        else:
            return 'rgba(220, 50, 50, 0.9)'

    fig = go.Figure()

    # --- Binned colored edges ---
    edge_traces = [
        (pos[u][0], pos[u][1], pos[v][0], pos[v][1], d['weight'])
        for u, v, d in G_filtered.edges(data=True)
    ]

    for x0, y0, x1, y1, w in edge_traces:
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=1.0 + w * 2.0, color=edge_color(w)),
            hoverinfo='none',
            showlegend=False
        ))

    # --- Invisible midpoint markers for hover info ---
    if edge_traces:
        fig.add_trace(go.Scatter(
            x=[(x0 + x1) / 2 for x0, y0, x1, y1, w in edge_traces],
            y=[(y0 + y1) / 2 for x0, y0, x1, y1, w in edge_traces],
            mode='markers',
            marker=dict(size=6, color='rgba(0,0,0,0)'),
            hovertext=[
                f'Similarity: {w:.3f} ({edge_label(w)})<br>p25={q25:.3f}, p75={q75:.3f}'
                for *_, w in edge_traces
            ],
            hoverinfo='text',
            showlegend=False
        ))

    # --- Legend entries (one dummy trace per bin) ---
    for label, color in [
        (f'High  (≥ {q75:.2f})',            'rgba(50, 180, 80, 0.9)'),
        (f'Medium ({q25:.2f} – {q75:.2f})', 'rgba(230, 150, 30, 0.9)'),
        (f'Low   (< {q25:.2f})',             'rgba(220, 50, 50, 0.9)'),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=3, color=color),
            name=label,
            showlegend=True
        ))

    # --- Compute per-node average similarity across ALL edges (not just filtered) ---
    node_avg_sim = {node: 0.0 for node in G_filtered.nodes()}
    node_edge_count = {node: 0 for node in G_filtered.nodes()}

    for u, v, d in G.edges(data=True):
        if u in node_avg_sim:
            node_avg_sim[u] += d['weight']
            node_edge_count[u] += 1
        if v in node_avg_sim:
            node_avg_sim[v] += d['weight']
            node_edge_count[v] += 1

    node_avg_sim = {
        n: (node_avg_sim[n] / node_edge_count[n] if node_edge_count[n] > 0 else 0.0)
        for n in node_avg_sim
    }

    node_colors = [node_color(node_avg_sim[n]) for n in G_filtered.nodes()]

    # --- Nodes ---
    node_x = [pos[node][0] for node in G_filtered.nodes()]
    node_y = [pos[node][1] for node in G_filtered.nodes()]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G_filtered.nodes()],
        textposition="top center",
        textfont=dict(size=13, color='#111111', family='Arial Black, sans-serif'),
        hovertext=[
            f'{node}<br>Avg similarity: {node_avg_sim[node]:.3f} ({edge_label(node_avg_sim[node])})'
            for node in G_filtered.nodes()
        ],
        hoverinfo='text',
        marker=dict(
            size=14,
            color=node_colors,
            line=dict(width=2, color='#000000')
        ),
        showlegend=False
    ))

    fig.update_layout(
        title=f'{language} Font Similarity Network (Threshold: {threshold_val:.2f})',
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(title='Similarity Bin', x=1.02, y=1, xanchor='left'),
        height=700
    )
    return fig


app = Dash(__name__)

df = None           # Will be loaded on app start
node_positions = None  # Will be calculated on app start

app.layout = html.Div([
    html.Div([
        html.H2(f'Script Network Analysis'),
        html.Hr(),
        dcc.Graph(id='network-graph'),
        dcc.Slider(0, 1, 0.05, value=0.75, id='threshold-slider',
                   marks={i / 20: f'{i / 20:.2f}' for i in range(21)})
    ])
])


@app.callback(
    Output('network-graph', 'figure'),
    Input('threshold-slider', 'value')
)
def update_graph(threshold_val):
    global node_positions
    return create_network_figure(df, threshold_val, node_positions)


if __name__ == "__main__":
    FONT_SIMILARITY_DIR = Path("./font_similarity_outputs/full_font_similarity_pairs")
    language = "Arabic"  # Change this to visualize a different language's network
    csv_path = FONT_SIMILARITY_DIR / f"font_similarity_pairs_{language}.csv"

    df = load_and_prepare_data(csv_path)

    # Calculate node positions once from the full graph
    G_full = nx.Graph()
    for _, row in df.iterrows():
        weight = row['similarity_normalized']
        G_full.add_edge(row['source'], row['target'], weight=weight)

    node_positions = nx.spring_layout(G_full, k=0.5, iterations=50, seed=42)

    app.run(debug=True)