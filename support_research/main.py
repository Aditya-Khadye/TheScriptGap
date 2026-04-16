import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils.utils_formatting as utils

import pandas as pd
import google_public as gp_data
import plotly.express as px

import time

from rapidfuzz import fuzz, process

def fuzzy_match_fonts(df, column, threshold=95):
    font_names = [f for f in df[column].unique().tolist() if pd.notna(f)]
    mapping = {}

    for name in font_names:
        if name in mapping:
            continue

        for other in font_names:
            if other in mapping:
                continue
            # Gate 1: first character must match
            if name[0].lower() != other[0].lower():
                continue
            # Gate 2: length cant be too different
            if abs(len(name) - len(other)) > 3:
                continue
            # Gate 3: fuzzy score
            score = fuzz.token_sort_ratio(name, other)
            if score >= threshold:
                print(f"Mapping '{other}' to '{name}' with score {score}")
                mapping[other] = name

    df["font_clean"] = df[column].map(mapping)
    return df


def main():
    print("Hello from support-research!")

    # Importing google font data
    google_fonts = gp_data.google_font_script_matches() # UNCOMMENT THIS WHEN DONE TESTING

    # make into df
    google_fonts_df = pd.DataFrame(list(google_fonts.items()), columns=['font', 'subsets']) # UNCOMMENT THIS WHEN DONE TESTING

    google_fonts_df = pd.read_csv('output/google_font_scripts.csv')


    # renaming columns to match
    google_fonts_df = google_fonts_df.rename(columns={'font': 'font_name', 'subsets': 'supported_scripts'})

    # standardize formatting
    google_fonts_df = utils.standardize_font_names(google_fonts_df)

    # Importing big query data (from our condenced http archive database)
    # Just using csv for now to save on costs
    big_query_df = pd.read_csv('output/big_query_data.csv')

    big_query_df = big_query_df.rename(columns={'scripts': 'supported_scripts'})

    big_query_df = utils.standardize_font_names(big_query_df) # already standardized but still needs to remove - and _ for " "

    big_query_df['supported_scripts'] = big_query_df['supported_scripts'].apply(utils.safe_literal_eval)
    google_fonts_df['supported_scripts'] = google_fonts_df['supported_scripts'].apply(utils.safe_literal_eval)


    
    # Combining the data into a single dataframe for analysis using fuzzy matching
    result = pd.concat([google_fonts_df, big_query_df], join='outer', ignore_index=True)

    print(f'before: {result["font_name"].nunique()}')

    start_time = time.time()

    result = fuzzy_match_fonts(result, "font_name", threshold=95)

    end_time = time.time()
    print(f"Fuzzy matching took {end_time - start_time:.2f} seconds")
    print(f'after: {result["font_clean"].nunique()}')

    print(result.describe())

    result.to_csv('output/combined_google_bigquery.csv', index=False)

    # result = result.sort_values(by='font_count', ascending=False).reset_index(drop=True)
    # print(result.head())
    # print(result.describe())



if __name__ == "__main__":
    main()
