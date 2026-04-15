import pandas as pd
import google_public as gp_data
import plotly.express as px

def standardize_font_names(fonts_df):
    """Standardize font names by converting to lowercase and replacing spaces with hyphens."""
    fonts_df['font_name'] = fonts_df['font_name'].str.lower().str.replace(' ', '-')
    return fonts_df

def standardize_script_names(fonts_df):
    """Standardize script names by converting to lowercase and replacing spaces with hyphens."""
    fonts_df['supported_scripts'] = fonts_df['supported_scripts'].str.lower()
    return fonts_df


# def fuzzy_match_fonts(df1, df2, threshold=80):
#     """Fuzzy match font names between two dataframes and combine them based on the best matches."""
#     # from fuzzywuzzy import process

#     # Create a new dataframe to store the results
#     matched_df = pd.DataFrame(columns=['font_name', 'supported_scripts', 'font_count'])

#     for index, row in df1.iterrows():
#         font_name = row['font_name']
#         supported_scripts = row['supported_scripts']
#         font_count = row.get('font_count', 0)

#         # Find the best match in df2
#         best_match = process.extractOne(font_name, df2['font_name'], score_cutoff=threshold)

#         if best_match:
#             matched_font_name = best_match[0]
#             matched_row = df2[df2['font_name'] == matched_font_name].iloc[0]
#             matched_supported_scripts = matched_row['supported_scripts']
#             matched_font_count = matched_row.get('font_count', 0)

#             # Combine the data
#             combined_supported_scripts = list(set(supported_scripts.split(',') + matched_supported_scripts.split(',')))
#             combined_font_count = font_count + matched_font_count

#             # Append to the matched dataframe
#             matched_df = matched_df.append({
#                 'font_name': font_name,
#                 'supported_scripts': ','.join(combined_supported_scripts),
#                 'font_count': combined_font_count
#             }, ignore_index=True)
#         else:
#             # If no match is found, keep the original data
#             matched_df = matched_df.append({
#                 'font_name': font_name,
#                 'supported_scripts': supported_scripts,
#                 'font_count': font_count
#             }, ignore_index=True)

#     return matched_df


# Tree map visualization function
def create_treemap(df):
    """Create a treemap visualization of font usage by supported scripts."""
    fig = px.treemap(df, path=['supported_scripts', 'font_name'], values='font_count',
                     title='Font Usage by Supported Scripts')
    fig.show()

# filter out null scripts
def filter_null_scripts(df):
    """Filter out rows with null or empty supported scripts."""
    return df[df['supported_scripts'].notnull() & (df['supported_scripts'] != '') & (df['supported_scripts'] != '[]')]

# filter out scripts that are not in the set list
def filter_scripts(df, scripts_kept):
    """Filter the dataframe to keep only rows with supported scripts in the specified list."""
    return df[df['supported_scripts'].isin(scripts_kept)]





def main():
    print("Hello from support-research!")

    # Importing google font data
    # google_fonts = gp_data.google_font_script_matches() # UNCOMMENT THIS WHEN DONE TESTING

    # make into df
    # google_fonts_df = pd.DataFrame(list(google_fonts.items()), columns=['font', 'subsets']) # UNCOMMENT THIS WHEN DONE TESTING

    google_fonts_df = pd.read_csv('output/google_font_scripts.csv')


    # renaming columns to match
    google_fonts_df = google_fonts_df.rename(columns={'font': 'font_name', 'subsets': 'supported_scripts'})

    # standardize formatting
    google_fonts_df = standardize_font_names(google_fonts_df)

    # Importing big query data (from our condenced http archive database)
    # Just using csv for now to save on costs
    big_query_df = pd.read_csv('output/big_query_data.csv')

    big_query_df = big_query_df.rename(columns={'scripts': 'supported_scripts'})

    big_query_df = standardize_font_names(big_query_df) # already standardized but still needs to remove - and _ for " "

    # Combining the data into a single dataframe for analysis using fuzzy matching
    result = pd.concat([google_fonts_df, big_query_df], join='outer', ignore_index=True)
    
    result = result.sort_values(by='font_count', ascending=False).reset_index(drop=True)
    print(result.head())
    print(result.describe())


    exploded_result = result.explode('supported_scripts')

    print(exploded_result.head())



    # filter_scripts(exploded_result, ['latin', 'latin-ext', 
    #                                  'cyrillic', 'cyrillic-ext', 
    #                                  'greek', 'greek-ext', 
    #                                  'japanese', 'katakana',
    #                                  'devangari', 'bengali', 'tamil', 'telugu',
    #                                  'arabic', 'Han'
                                     
    #                                  ]) # filter to only keep rows with these scripts

    # # combine script extensions with their base scripts (e.g. latin-ext with latin) and remove the extension rows
    # exploded_result['supported_scripts'] = exploded_result['supported_scripts'].replace({
    #     'latin-ext': 'latin',
    #     'cyrillic-ext': 'cyrillic',
    #     'greek-ext': 'greek'
    # })

    # create_treemap(filter_null_scripts(exploded_result))






if __name__ == "__main__":
    main()
