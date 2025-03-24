"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline
import pandas as pd


def check_and_remove_duplicates(df, key='id'):
    """
    Check and remove duplicate entries based on a key column.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    key (str): The column to check for duplicates (default is 'id')

    Returns:
    pd.DataFrame: A new DataFrame with duplicates removed
    """
    duplicate_ids = df[df.duplicated(key, keep=False)][key].unique()

    if len(duplicate_ids) > 0:
        print(f"Found {len(duplicate_ids)} duplicated '{key}' values:")
        print(duplicate_ids)
    else:
        print(f"No duplicates found in column '{key}'.")

    # Drop duplicates, keeping the first occurrence
    df_cleaned = df.drop_duplicates(subset=key, keep='first')

    return df_cleaned


def drop_unwanted_columns(x: pd.DataFrame, columns_to_drop=None) -> pd.DataFrame:
    """
    Drops unwanted columns from the DataFrame.

    Parameters:
        :param columns_to_drop: List of column names to drop. Defaults to ['Unnamed: 0.1', 'Unnamed: 0'].
        :param x: df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns dropped.

    """
    if columns_to_drop is None:
        columns_to_drop = ['Unnamed: 0.1', 'Unnamed: 0']

    return x.drop(columns=[col for col in columns_to_drop if col in x.columns], axis=1)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])
