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

# Temporary main function for google data
def main():
    print("Hello from support-research!")

    print("\nSupport Full List")
    support_dict = google_top_list()
    support_df = dict_to_pd(support_dict)
    print(support_df.head())
    support_df.to_csv('output/google_support_toplist.csv', index=False)

    print("\nSupport Filtered by top 8 Scripts (population)")

    support_dict = google_script_filtered()
    support_df = dict_to_pd(support_dict)
    print(support_df)
    support_df.to_csv('output/google_support_filtered.csv', index=False)
    print("Google Support Completed")



if __name__ == "__main__":
    main()