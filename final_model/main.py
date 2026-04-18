import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd


def main():
    print("Hello from final-model!")

    # Numbe of requests for each script (how utilized is this script on the web globally?)
    exposure = pd.read_csv("../exposure_research/output/exposure_filtered_results.csv")

    exposure = exposure.rename(columns={"count": "exposure"})

    exposure['script'] = exposure['script'].str.lower()

    exposure = exposure[["script", "exposure"]]
    
    print(exposure.head())





    # Number of distinct fonts supporting each script
    # TODO: CHANGE THIS OUT LATER FOR UPDATED METRICS THAT USE GLOBAL WEB DATA TO SUPPORT DISTINCT FONT COUNT
    support = pd.read_csv("../support_research/output/script_font_counts.csv")

    # Reformat to group by script and count the number of distinct fonts supporting each script

    support = support.rename(columns={"distinct_font_count": "support"})
    support = support.rename(columns={"supported_scripts": "script"})
    support = support[["script", "support"]]
    support['script'] = support['script'].str.lower()

    print(support.head())

    # Since this is using google data we need to rename some things to match the other data we have
    # It's not perfect yet though could improve upon this further
    
    support['script'] = support['script'].str.replace(r'chinese-simplified|chinese-traditional', 'han', regex=True)
    support['script'] = support['script'].str.replace(r'japanese', 'katakana', regex=True)


    # TODO: Currently this is backwards deal with it later abs(1 - diversity_index) or something like that 
    similarity = pd.read_csv("../similarity_research/diversity_research/vit100_outputs/diversity_index_summary.csv")

    similarity = similarity.rename(columns={"diversity_index": "similarity"})
    similarity = similarity[["script", "similarity"]]
    similarity['script'] = similarity['script'].str.lower()


    print(similarity.head())

    # merge only keeping the main indexes
    final_df = exposure[["script", "exposure"]].merge(support[["script", "support"]], on="script", how="outer").merge(similarity[["script", "similarity"]], on="script", how="outer")

    print(final_df.head())

    

if __name__ == "__main__":
    main()
