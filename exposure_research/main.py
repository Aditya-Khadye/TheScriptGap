import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils.utils_formatting as utils

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, ALL
from dash import callback_context, State
from plotly.express.colors import qualitative

# Load data globally so it's available to callbacks
font_script_df = None
scripts_list = [
    "devanagari",
    "arabic",
    "bengali",
    "cyrillic",
    "katakana",
    "telugu",
    "tamil",
    "latin"
]

# Assign colors to scripts using a palette
color_palette = qualitative.Pastel  # or try: Set1, Bold, Pastel, Light, Dark, etc.
script_colors = {script: color_palette[i % len(color_palette)] for i, script in enumerate(scripts_list)}

def create_treemap(df):
    """Create a treemap visualization of font usage by supported scripts."""
    fig = px.treemap(
        df,
        path=['script', 'font_name'],
        values='font_count',
        color='script',
        color_discrete_map=script_colors,
        title='Font Usage by Supported Scripts'
    )
    fig.update_layout(
        transition=dict(duration=0),
        margin=dict(t=50, l=10, r=10, b=10),
        autosize=True,
    )
    return fig


def main():
    global font_script_df, scripts_list, script_colors
    
    print("Loading data...")
    
    # Load and process data
    big_query_df = pd.read_csv('../support_research/output/big_query_data.csv')
    big_query_df = big_query_df.rename(columns={'scripts': 'supported_scripts'})
    big_query_df = utils.standardize_font_names(big_query_df)
    big_query_df['supported_scripts'] = big_query_df['supported_scripts'].apply(utils.safe_literal_eval)
    
    exploded_result = big_query_df.explode('supported_scripts')
    exploded_result = exploded_result.rename(columns={'supported_scripts': 'script'})
    
    font_script_df = (
        exploded_result[['font_name', 'script', 'font_count']]
        .groupby(['script', 'font_name'], as_index=False)['font_count'].sum()
        .sort_values('font_count', ascending=False)
        .reset_index(drop=True)
    )
    
    font_script_df['font_name'] = font_script_df['font_name'].where(font_script_df['font_count'] >= 5000, 'other')
    font_script_df = utils.filter_null_scripts(font_script_df)
    
    # Filter to only include scripts in scripts_list
    font_script_df = font_script_df[font_script_df['script'].isin(scripts_list)]
    
    # Sort scripts by total font_count (descending)
    script_totals = font_script_df.groupby('script')['font_count'].sum().sort_values(ascending=False)
    scripts_list = script_totals.index.tolist()
    
    # Reassign colors based on sorted order
    script_colors = {script: color_palette[i % len(color_palette)] for i, script in enumerate(scripts_list)}
    
    # Create Dash app
    app = Dash(__name__)
    
    app.layout = html.Div([
    html.H1("Script Exposure based on font requests"),
    
    html.Div([
        html.Label("Toggle Scripts:"),
        html.Div([
            html.Button(
                script.capitalize(),
                id={'type': 'script-button', 'index': script},
                n_clicks=0,
                style={
                    'margin': '5px',
                    'padding': '10px 15px',
                    'backgroundColor': '#1f77b4',
                    'color': 'black',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold'
                }
            )
            for script in scripts_list
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '20px'})
    ], style={'padding': '20px'}),
    
    dcc.Store(id='selected-scripts-store', data=scripts_list),
    dcc.Graph(
        id='treemap-graph',
        style={'height': '85vh', 'width': '100%'},
        config={'responsive': True},
    )
], style={'width': '100%', 'margin': '0', 'padding': '0'})
    
    @app.callback(
        Output('selected-scripts-store', 'data'),
        Input({'type': 'script-button', 'index': ALL}, 'n_clicks'),
        State('selected-scripts-store', 'data'),
        prevent_initial_call=True
    )
    def update_selected_scripts(n_clicks, current_selected):
        """Toggle the clicked script on/off, keeping others unchanged."""
        ctx = callback_context
        if not ctx.triggered or not current_selected:
            return scripts_list
        
        # Get which button was clicked - triggered_id returns the dict directly
        clicked_script = ctx.triggered_id['index']
        
        # Toggle the clicked script
        if clicked_script in current_selected:
            return [s for s in current_selected if s != clicked_script]
        else:
            return current_selected + [clicked_script]
    
    @app.callback(
        Output('treemap-graph', 'figure'),
        Input('selected-scripts-store', 'data')
    )
    def update_treemap(selected_scripts):
        """Update treemap based on selected scripts."""
        if not selected_scripts:
            selected_scripts = scripts_list
        
        filtered_df = font_script_df[font_script_df['script'].isin(selected_scripts)]
        return create_treemap(filtered_df)
    
    @app.callback(
        Output({'type': 'script-button', 'index': ALL}, 'style'),
        Input('selected-scripts-store', 'data')
    )
    def update_button_styles(selected_scripts):
        """Update button appearance based on selection state."""
        return [
            {
                'margin': '5px',
                'padding': '10px 15px',
                'backgroundColor': script_colors[script] if script in selected_scripts else '#cccccc',
                'color': 'black',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontWeight': 'bold'
            }
            for script in scripts_list
        ]
    
    print("Starting Dash app on http://localhost:8050")
    app.run(debug=True)


if __name__ == "__main__":
    main()
