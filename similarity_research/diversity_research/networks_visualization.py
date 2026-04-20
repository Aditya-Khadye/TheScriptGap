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


def create_network_figure(df: pd.DataFrame, threshold_val: float) -> go.Figure:
    """Create a network figure for a given threshold."""
    G = nx.Graph()
    
    for _, row in df.iterrows():
        weight = row['similarity_normalized']
        G.add_edge(row['source'], row['target'], weight=weight)
    
    # Filter edges based on threshold
    filtered_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= threshold_val]
    G_filtered = G.edge_subgraph(filtered_edges).copy()
    
    if G_filtered.number_of_nodes() == 0:
        return go.Figure().add_annotation(text="No edges above this threshold")
    
    pos = nx.spring_layout(G_filtered, k=0.5, iterations=50, seed=42)
    
    edge_x, edge_y = [], []
    for edge in G_filtered.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = [pos[node][0] for node in G_filtered.nodes()]
    node_y = [pos[node][1] for node in G_filtered.nodes()]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
        line=dict(width=0.5, color='#888'), hoverinfo='none', showlegend=False))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
        text=[node for node in G_filtered.nodes()], textposition="top center",
        hoverinfo='text', marker=dict(size=8, color='#0066cc'), showlegend=False))
    
    fig.update_layout(
        title=f'Font Similarity Network (Threshold: {threshold_val:.2f})',
        hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700
    )
    return fig


app = Dash(__name__)

df = None  # Will be loaded on app start

app.layout = html.Div([
    html.Div([
        html.H2('Script Network Analysis'),
        html.Hr(),
        dcc.Graph(id='network-graph'),
        dcc.Slider(0, 1, 0.05, value=0.75, id='threshold-slider',
            marks={i/20: f'{i/20:.2f}' for i in range(21)})
    ])
])

@app.callback(
    Output('network-graph', 'figure'),
    Input('threshold-slider', 'value')
)
def update_graph(threshold_val):
    return create_network_figure(df, threshold_val)


if __name__ == "__main__":
    FONT_SIMILARITY_DIR = Path("./font_similarity_outputs/full_font_similarity_pairs")
    language = "Tamil"
    csv_path = FONT_SIMILARITY_DIR / f"font_similarity_pairs_{language}.csv"
    
    df = load_and_prepare_data(csv_path)
    
    app.run(debug=True)