"""
Feature engineering pipeline for the LendingClub credit risk modeling project.
"""

from kedro.pipeline import Pipeline, node, pipeline
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
    create_home_ownership_ordinal,
    create_time_based_features,
    create_payment_features,
    create_credit_inquiry_features,
    create_delinquency_features,
    create_debt_composition_features,
    create_account_activity_features,
    create_loan_purpose_risk_features,
    print_columns,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the feature engineering pipeline that transforms preprocessed LendingClub data
    into model-ready features. The pipeline is organized into logical groups of related
    transformations, building more complex features as it progresses.

    Returns:
        Pipeline: A Kedro pipeline for feature engineering
    """
    return pipeline(
        [
            # Group 1: Categorical field handling
            node(
                func=create_home_ownership_ordinal,
                inputs="intermediate_data",
                outputs="data_with_home_ownership_ordinal",
                name="create_home_ownership_ordinal",
            ),
            # Group: Purpose and loan characteristics
            node(
                func=encode_purpose_field,
                inputs="data_with_home_ownership_ordinal",
                outputs="data_with_purpose",
                name="encode_purpose",
            ),
            node(
                func=create_loan_purpose_risk_features,
                inputs="data_with_purpose",
                outputs="data_with_purpose_risk",
                name="create_purpose_risk",
            ),
            # Group: hardship indicators
            node(
                func=create_has_hardship_flag,
                inputs="data_with_purpose_risk",
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
            # Group 2: Employment and income features
            node(
                func=engineer_emp_length_features,
                inputs="data_with_hardship_features",
                outputs="data_with_emp_features",
                name="create_emp_features",
            ),
            # Group 3: Loan terms and interest features
            node(
                func=encode_interest_and_grade_fields,
                inputs="data_with_emp_features",
                outputs="data_with_grade_features",
                name="encode_grades",
            ),
            # Group 4: Joint application features
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
            # Group 5: Credit history and timeline features
            node(
                func=create_credit_age_feature,
                inputs="data_with_joint_revol",
                outputs="data_with_credit_age",
                name="create_credit_age",
            ),
            node(
                func=create_time_based_features,
                inputs="data_with_credit_age",
                outputs="data_with_time_features",
                name="create_time_features",
            ),
            node(
                func=create_mths_since_last_record_feature,
                inputs="data_with_time_features",
                outputs="data_with_last_record",
                name="create_last_record",
            ),
            node(
                func=handle_pub_rec_missing,
                inputs=["data_with_last_record", "params:pub_rec_strategy"],
                outputs="data_with_pub_rec",
                name="handle_pub_rec",
            ),
            # Group 6: Credit utilization features
            node(
                func=create_revol_util_features,
                inputs="data_with_pub_rec",
                outputs="data_with_revol_util",
                name="create_revol_util",
            ),
            # Group 7: Loan details and payment features
            node(
                func=create_initial_list_status_flag,
                inputs="data_with_revol_util",
                outputs="data_with_list_status",
                name="create_list_status",
            ),
            node(
                func=create_payment_features,
                inputs="data_with_list_status",
                outputs="data_with_payment_features",
                name="create_payment_features",
            ),
            # Group 9: Credit profile features
            node(
                func=create_fico_score_features,
                inputs="data_with_payment_features",
                outputs="data_with_fico",
                name="create_fico_features",
            ),
            node(
                func=create_loan_amount_features,
                inputs="data_with_fico",
                outputs="data_with_loan_features",
                name="create_loan_features",
            ),
            node(
                func=create_credit_history_features,
                inputs="data_with_loan_features",
                outputs="data_with_credit_history",
                name="create_credit_history",
            ),
            node(
                func=create_credit_inquiry_features,
                inputs="data_with_credit_history",
                outputs="data_with_inquiry_features",
                name="create_inquiry_features",
            ),
            node(
                func=create_delinquency_features,
                inputs="data_with_inquiry_features",
                outputs="data_with_delinquency_features",
                name="create_delinquency_features",
            ),
            node(
                func=create_debt_composition_features,
                inputs="data_with_delinquency_features",
                outputs="data_with_debt_composition",
                name="create_debt_composition",
            ),
            node(
                func=create_account_activity_features,
                inputs="data_with_debt_composition",
                outputs="data_with_account_activity",
                name="create_account_activity",
            ),
            # Group 11: Model-specific feature creation
            node(
                func=create_credit_history_model_features,
                inputs="data_with_account_activity",
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
            # Group 12: Final processing and dataset preparation
            node(
                func=drop_unwanted_columns,
                inputs=["data_with_model_features", "params:columns_to_drop"],
                outputs="engineered_features",
                name="drop_columns",
            ),
            # node(
            #     func=lambda df: print_columns(df, "drop_columns"),
            #     inputs="engineered_features",
            #     outputs="engineered_features_debug",
            #     name="debug_engineered_features",
            # ),
            node(
                func=create_model_specific_datasets,
                inputs="engineered_features",
                outputs=["tree_features", "regression_features"],
                name="create_model_datasets",
            ),
            node(
                func=lambda df: print_columns(df, "create_model_datasets"),
                inputs="tree_features",
                outputs="tree_features_debug",
                name="debug_tree_features",
            ),
            node(
                func=lambda df: print_columns(df, "create_model_datasets"),
                inputs="regression_features",
                outputs="regression_features_debug",
                name="debug_regression_features",
            ),
            # Group 13: Evaluate model-specific datasets
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

# # Group 10: Feature evaluation
# node(
#     func=evaluate_feature_engineering,
#     inputs="data_with_account_activity",
#     outputs="feature_evaluation_metrics",
#     name="evaluate_features",
# ),
