import pandas as pd
import ast

def standardize_font_names(fonts_df):
    """Standardize font names by converting to lowercase and replacing spaces with hyphens."""
    fonts_df['font_name'] = fonts_df['font_name'].str.lower().str.replace(' ', '-')
    return fonts_df

def standardize_script_names(fonts_df):
    """Standardize script names by converting to lowercase and replacing spaces with hyphens."""
    fonts_df['supported_scripts'] = fonts_df['supported_scripts'].str.lower()
    return fonts_df


def filter_null_scripts(df):
    """Filter out rows with null or empty scripts."""
    return df[df['script'].notnull() & (df['script'] != '') & (df['script'] != '[]')]

def filter_scripts(df, scripts_kept):
    """Filter the dataframe to keep only rows with supported scripts in the specified list."""
    return df[df['script'].isin(scripts_kept)]

def safe_literal_eval(val):
    """
    Safely evaluate a string representation of a list, or return an empty list if the value is null or empty.
    Big query data outputs lists as strings, so we need to convert them back to actual lists. 
    This function handles various edge cases.
    """

    if pd.isna(val) or val == '':
        return []

    if isinstance(val, list):
        scripts = val
    elif isinstance(val, str):
        val = val.strip()
        if val.startswith('[') and val.endswith(']'):
            inner = val[1:-1].strip()
            if inner == '':
                return []
            try:
                scripts = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                scripts = [item.strip() for item in inner.split(',') if item.strip()]
        else:
            scripts = [val]
    else:
        return []

    return [str(script).lower().strip().replace(' ', '_') for script in scripts if str(script).strip()]
