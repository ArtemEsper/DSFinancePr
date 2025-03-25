"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    check_and_remove_duplicates,
    drop_unwanted_columns,
    normalize_column_names,
    fix_column_types,
    remove_invalid_rows,
    filter_loan_status,
    clean_string_fields,
    cap_outliers,
)


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
        node(
            func=normalize_column_names,
            inputs="dropped_columns",
            outputs="normalized_columns",
            name="normalize_column_names"),
        node(
            func=fix_column_types,
            inputs="normalized_columns",
            outputs="typed_columns",
            name="fix_column_types"),
        node(
            func=remove_invalid_rows,
            inputs="typed_columns",
            outputs="cleaned_rows",
            name="remove_invalid_rows"),
        node(
            func=filter_loan_status,
            inputs="cleaned_rows",
            outputs="filtered_status",
            name="filter_loan_status"),
        node(
            func=clean_string_fields,
            inputs="filtered_status",
            outputs="cleaned_strings",
            name="clean_string_fields"),
        node(
            func=cap_outliers,
            inputs="cleaned_strings",
            outputs="data_processing_output",
            name="cap_outliers"),
    ])
