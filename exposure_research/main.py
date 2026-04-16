import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils.utils_formatting as utils

import pandas as pd
import plotly.express as px


# Tree map visualization function
def create_treemap(df):
    """Create a treemap visualization of font usage by supported scripts."""
    fig = px.treemap(
        df,
        path=['script', 'font_name'],
        values='font_count',
        color='script',
        title='Font Usage by Supported Scripts'
    )
    fig.show()



def main():
    print("Hello from exposure-research!")

    # Importing big query data (from our condenced http archive database)
    # Just using csv for now to save on costs
    # PATH TO SUPPORT RESEARCH DATA, CHANGE THIS WHEN REFACTORING
    big_query_df = pd.read_csv('../support_research/output/big_query_data.csv')

    big_query_df = big_query_df.rename(columns={'scripts': 'supported_scripts'})

    big_query_df = utils.standardize_font_names(big_query_df) # already standardized but still needs to remove - and _ for " "


    big_query_df['supported_scripts'] = big_query_df['supported_scripts'].apply(utils.safe_literal_eval)


    exploded_result = big_query_df.explode('supported_scripts')
    exploded_result = exploded_result.rename(columns={'supported_scripts': 'script'})


    # 5. Group by script and aggregate
    font_script_df = (
        exploded_result[['font_name', 'script', 'font_count']]
        .groupby(['script', 'font_name'], as_index=False)['font_count'].sum()
        .sort_values('font_count', ascending=False)
        .reset_index(drop=True)
    )


    font_script_df['font_name'] = font_script_df['font_name'].where(font_script_df['font_count'] >= 5000, 'other')


    print(font_script_df.head())

    print(font_script_df['script'].value_counts().to_string())

    create_treemap(utils.filter_scripts(utils.filter_null_scripts(font_script_df), scripts_kept=[
        "devanagari",
        "arabic",
        "bengali",
        "cyrillic",
        "katakana",
        "telugu",
        "tamil",
        "latin"
    ]))



if __name__ == "__main__":
    main()
