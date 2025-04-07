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
    clean_string_fields,
    cap_outliers,
    filter_and_encode_loan_status,
    encode_joint_application_flag,
    clean_remaining_object_columns,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=check_and_remove_duplicates,
                inputs="lendingclub",  # lendingclub for transformation and raw_data dataset for the tests
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
                func=clean_remaining_object_columns,
                inputs="typed_columns",
                outputs="object_columns_converted",
                name="clean_remaining_object_columns",
            ),
            node(
                func=encode_joint_application_flag,
                inputs="object_columns_converted",
                outputs="encoded_app_type",
                name="encode_joint_application_flag",
            ),
            node(
                func=filter_and_encode_loan_status,
                inputs="encoded_app_type",
                outputs="processed_loan_stat",
                name="encode_loan_status",
            ),
            node(
                func=normalize_column_names,
                inputs="processed_loan_stat",
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
                func=clean_string_fields,
                inputs="cleaned_rows",
                outputs="cleaned_strings",
                name="clean_string_fields",
            ),
            node(
                func=drop_unwanted_columns,
                inputs={
                    "x": "cleaned_strings",
                    "drop_list": "params:admin_columns_to_drop",
                },
                outputs="clean_data",
                name="drop_admin_columns",
            ),
            node(
                func=cap_outliers,
                inputs="clean_data",
                outputs="intermediate_data",  # intermediate_mock_data or intermediate_sample_data for tests and
                name="cap_outliers",
            ),
        ]
    )
