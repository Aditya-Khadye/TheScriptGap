import os
from dotenv import load_dotenv
import requests
import json
import pandas as pd

### SETUP ###
load_dotenv()

google_font_api = os.getenv("GOOGLE_FONTS_API")

url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={google_font_api}"

### HELPER FUNCTIONS ###
def dict_to_pd(support_dict):
  support_df = pd.DataFrame.from_dict(support_dict, orient='index', columns=['count'])
  support_df = support_df.reset_index().rename(columns={'index': 'script'})
  support_df = support_df.sort_values(by='count', ascending=False)

  return support_df

### MAIN FUNCTIONS ###
def google_top_list():
    # Note if we end up using this at all we might wanna
    # filter out certain subsets like symbol sets
    response = requests.get(url)
    data = response.json()

    support_dictionary = {}
    for font in data['items']:
        for subset in font['subsets']:
            support_dictionary[subset] = support_dictionary.get(subset, 0) + 1

    return support_dictionary


def google_script_filtered():

    support_dictionary = {}

    # Filter for  the top 8 scripts
    # Want the number of occurances of script subset

    # Han, Devangari, Arabic, Bengali, Cyrillic, Kana, Telugu, Tamil
    subsets = ["chinese-simplified", "chinese-traditional", "devanagari", "arabic", "bengali",
               "cyrillic", "japanese", "telugu", "tamil"]

    for subset in subsets:
      response = requests.get(url, params={'subset': subset})
      data = response.json()

      support_dictionary[subset] = len(data['items'])
    return support_dictionary

def google_script_relations():
    response = requests.get(url)
    data = response.json()

    # print(data)

    filter_scripts = ["chinese-simplified", "chinese-traditional", "devanagari", "arabic", "bengali",
                      "cyrillic", "japanese", "telugu", "tamil"]

    filter_scripts = None # Comment out to use filter
    
    combinations = {}
    
    for font in data['items']:
        subsets = font['subsets']
        
        # Filter subsets if specified
        if filter_scripts:
            subsets = [s for s in subsets if s in filter_scripts]
        
        # Generate all unique pairs for this font
        if len(subsets) >= 2:
            for i in range(len(subsets)):
                for j in range(i + 1, len(subsets)):
                    pair = tuple(sorted([subsets[i], subsets[j]]))
                    combinations[pair] = combinations.get(pair, 0) + 1
    
    return combinations


def google_font_script_matches(filter_scripts=None):
    """Return fonts and the filtered scripts they support."""
    if filter_scripts is None:
        filter_scripts = [
            "chinese-simplified",
            "chinese-traditional",
            "devanagari",
            "arabic",
            "bengali",
            "cyrillic",
            "japanese",
            "telugu",
            "tamil",
        ]

    filter_scripts = set(filter_scripts)

    response = requests.get(url)
    data = response.json()

    matching_fonts = {}
    for font in data["items"]:
        matching_subsets = [subset for subset in font["subsets"] if subset in filter_scripts]
        if matching_subsets:
            matching_fonts[font.get("family", font.get("id"))] = matching_subsets

    return matching_fonts


# Temporary main function for google data
def main():
    print("Hello from support-research!")

    # google_script_relations_dict = google_script_relations()
    # google_script_relations_df = pd.DataFrame(list(google_script_relations_dict.items()), columns=['script_pair', 'count'])
    # google_script_relations_df = google_script_relations_df.sort_values(by='count', ascending=False)
    # google_script_relations_df.to_csv('output/google_script_relations.csv', index=False)



    filter_scripts = ["chinese-simplified", "chinese-traditional", "devanagari", "arabic", "bengali",
                      "cyrillic", "japanese", "telugu", "tamil"]
    matching_fonts = google_font_script_matches(filter_scripts=filter_scripts)
    print(matching_fonts)
    matching_fonts_df = pd.DataFrame(list(matching_fonts.items()), columns=['font', 'subsets'])
    matching_fonts_df.to_csv('output/google_font_scripts.csv', index=False)

    # print("\nSupport Full List")
    # support_dict = google_top_list()
    # support_df = dict_to_pd(support_dict)
    # print(support_df.head())
    # support_df.to_csv('output/google_support_toplist.csv', index=False)

    # print("\nSupport Filtered by top 8 Scripts (population)")

    # support_dict = google_script_filtered()
    # support_df = dict_to_pd(support_dict)
    # print(support_df)
    # support_df.to_csv('output/google_support_filtered.csv', index=False)
    # print("Google Support Completed")



if __name__ == "__main__":
    main()