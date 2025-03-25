"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import check_and_remove_duplicates, drop_unwanted_columns


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=check_and_remove_duplicates,
            inputs="raw_data",
            outputs="deduped_data",
            name="deduplicate_by_id"
        ),
        node(
            func=drop_unwanted_columns,
            inputs=dict(x="deduped_data", drop_list="params:admin_columns_to_drop"),
            outputs="clean_data",
            name="drop_admin_columns"
        ),
    ])
