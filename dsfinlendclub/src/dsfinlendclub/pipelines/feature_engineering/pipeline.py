"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    create_has_hardship_flag,
    create_was_late_before_hardship,
    create_hardship_features,
    engineer_emp_length_features,
    encode_interest_and_grade_fields,
    create_joint_income_feature,
    create_joint_dti_feature,
    create_joint_verification_feature,
    create_joint_revol_bal_feature,
    create_credit_age_feature,
    create_mths_since_last_record_feature,
    handle_pub_rec_missing,
    create_revol_util_features,
    create_initial_list_status_flag,
    encode_purpose_field,
    create_fico_score_features,
    create_loan_amount_features,
    create_credit_history_features,
    evaluate_feature_engineering,
    drop_unwanted_columns,
    create_credit_history_model_features,
    create_fico_model_features,
    create_dti_model_features,
    create_income_model_features,
    create_term_model_features,
    create_utilization_model_features,
    create_model_specific_datasets,
)


def create_pipeline(**kwargs) -> Pipeline:  # feature engineering pipeline
    return pipeline(
        [
            # Hardship features
            node(
                func=create_has_hardship_flag,
                inputs="processed_data",
                outputs="data_with_hardship_flag",
                name="create_hardship_flag",
            ),
            node(
                func=create_was_late_before_hardship,
                inputs="data_with_hardship_flag",
                outputs="data_with_late_hardship",
                name="create_late_hardship",
            ),
            node(
                func=create_hardship_features,
                inputs="data_with_late_hardship",
                outputs="data_with_hardship_features",
                name="create_hardship_features",
            ),
            # Employment features
            node(
                func=engineer_emp_length_features,
                inputs="data_with_hardship_features",
                outputs="data_with_emp_features",
                name="create_emp_features",
            ),
            # Interest and grade features
            node(
                func=encode_interest_and_grade_fields,
                inputs="data_with_emp_features",
                outputs="data_with_grade_features",
                name="encode_grades",
            ),
            # Joint application features
            node(
                func=create_joint_income_feature,
                inputs="data_with_grade_features",
                outputs="data_with_joint_income",
                name="create_joint_income",
            ),
            node(
                func=create_joint_dti_feature,
                inputs="data_with_joint_income",
                outputs="data_with_joint_dti",
                name="create_joint_dti",
            ),
            node(
                func=create_joint_verification_feature,
                inputs="data_with_joint_dti",
                outputs="data_with_joint_verification",
                name="create_joint_verification",
            ),
            node(
                func=create_joint_revol_bal_feature,
                inputs="data_with_joint_verification",
                outputs="data_with_joint_revol",
                name="create_joint_revol",
            ),
            # Credit history features
            node(
                func=create_credit_age_feature,
                inputs="data_with_joint_revol",
                outputs="data_with_credit_age",
                name="create_credit_age",
            ),
            node(
                func=create_mths_since_last_record_feature,
                inputs="data_with_credit_age",
                outputs="data_with_last_record",
                name="create_last_record",
            ),
            node(
                func=handle_pub_rec_missing,
                inputs=["data_with_last_record", "params:pub_rec_strategy"],
                outputs="data_with_pub_rec",
                name="handle_pub_rec",
            ),
            # Revolving utilization features
            node(
                func=create_revol_util_features,
                inputs="data_with_pub_rec",
                outputs="data_with_revol_util",
                name="create_revol_util",
            ),
            # List status features
            node(
                func=create_initial_list_status_flag,
                inputs="data_with_revol_util",
                outputs="data_with_list_status",
                name="create_list_status",
            ),
            # Purpose encoding
            node(
                func=encode_purpose_field,
                inputs="data_with_list_status",
                outputs="data_with_purpose",
                name="encode_purpose",
            ),
            # FICO score features
            node(
                func=create_fico_score_features,
                inputs="data_with_purpose",
                outputs="data_with_fico",
                name="create_fico_features",
            ),
            # Loan amount features
            node(
                func=create_loan_amount_features,
                inputs="data_with_fico",
                outputs="data_with_loan_features",
                name="create_loan_features",
            ),
            # Credit history composite features
            node(
                func=create_credit_history_features,
                inputs="data_with_loan_features",
                outputs="data_with_credit_history",
                name="create_credit_history",
            ),
            # Feature evaluation
            node(
                func=evaluate_feature_engineering,
                inputs="data_with_credit_history",
                outputs="feature_evaluation_metrics",
                name="evaluate_features",
            ),
            # Final column cleanup
            node(
                func=create_credit_history_model_features,
                inputs="feature_evaluation_metrics",
                outputs="data_with_credit_model_features",
                name="create_credit_model_features",
            ),
            node(
                func=create_fico_model_features,
                inputs="data_with_credit_model_features",
                outputs="data_with_fico_model_features",
                name="create_fico_model_features",
            ),
            node(
                func=create_dti_model_features,
                inputs="data_with_fico_model_features",
                outputs="data_with_dti_model_features",
                name="create_dti_model_features",
            ),
            node(
                func=create_income_model_features,
                inputs="data_with_dti_model_features",
                outputs="data_with_income_model_features",
                name="create_income_model_features",
            ),
            node(
                func=create_term_model_features,
                inputs="data_with_income_model_features",
                outputs="data_with_term_model_features",
                name="create_term_model_features",
            ),
            node(
                func=create_utilization_model_features,
                inputs="data_with_term_model_features",
                outputs="data_with_model_features",
                name="create_utilization_model_features",
            ),
            node(
                func=drop_unwanted_columns,
                inputs=["data_with_model_features", "params:columns_to_drop"],
                outputs="engineered_features",
                name="drop_columns",
            ),
            node(
                func=create_model_specific_datasets,
                inputs="engineered_features",
                outputs=["tree_features", "regression_features"],
                name="create_model_datasets",
            ),
            # Evaluate model-specific features
            node(
                func=evaluate_feature_engineering,
                inputs="tree_features",
                outputs="tree_features_evaluation",
                name="evaluate_tree_features",
            ),
            node(
                func=evaluate_feature_engineering,
                inputs="regression_features",
                outputs="regression_features_evaluation",
                name="evaluate_regression_features",
            ),
        ]
    )
