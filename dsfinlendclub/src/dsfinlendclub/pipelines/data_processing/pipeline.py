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
    filter_and_flag_loan_status,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=check_and_remove_duplicates,
                inputs="sample",  # raw_data dataset in the tests
                outputs=["deduped_data", "dedup_flag"],
                name="deduplicate_by_id",
            ),
            node(
                func=fix_column_types,
                inputs="deduped_data",
                outputs="typed_columns",
                name="fix_column_types",
            ),
            node(
                func=filter_and_flag_loan_status,
                inputs="typed_columns",
                outputs="processed_loan_stat",
                name="encode_loan_status",
            ),
            node(
                func=drop_unwanted_columns,
                inputs={"x": "processed_loan_stat", "drop_list": "params:admin_columns_to_drop"},
                outputs="clean_data",
                name="drop_admin_columns"
            ),
            node(
                func=normalize_column_names,
                inputs="clean_data",
                outputs="normalized_columns",
                name="normalize_column_names",
            ),
            node(
                func=remove_invalid_rows,
                inputs="normalized_columns",
                outputs="cleaned_rows",
                name="remove_invalid_rows",
            ),
            node(
                func=filter_loan_status,
                inputs="cleaned_rows",
                outputs="filtered_status",
                name="filter_loan_status",
            ),
            node(
                func=clean_string_fields,
                inputs="filtered_status",
                outputs="cleaned_strings",
                name="clean_string_fields",
            ),
            node(
                func=cap_outliers,
                inputs="cleaned_strings",
                outputs="intermediate_data",
                name="cap_outliers",
            ),
        ]
    )
