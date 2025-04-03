import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from dsfinlendclub.pipelines.data_processing.pipeline import create_pipeline
from ydata_profiling import ProfileReport
from kedro.config import OmegaConfigLoader
from pathlib import Path

conf_path = Path("conf")
conf_loader = OmegaConfigLoader(conf_source=str(conf_path), env="local")

params = conf_loader["parameters"]
columns_to_drop = params["admin_columns_to_drop"]


def test_data_processing_pipeline_runs():
    raw_data = pd.DataFrame({
        "id": [1, 2, 3],  # ⚠️ has to be deduplicated and ❌ deleted
        "member_id": [1, 3, 2],  # ❌ has to be deleted
        "url": ["https://lendingclub.com/browse/loanDetail.action?loan_id=167338079",
                "https://lendingclub.com/browse/loanDetail.action?loan_id=71016917",
                "https://lendingclub.com/browse/loanDetail.action?loan_id=71016917"],  # ❌ has to be deleted
        "title": ["Credit card refinancing", "Debt consolidation", "Home improvement"],  # ❌ has to be deleted
        "zip_code": ["115xx", "116xx", "117xx"],  # ❌ has to be deleted
        "policy_code": [1.0, 1.0, 1.0],  # ❌ has to be deleted
        "pymnt_plan": ["n", "n", "n"],  # ❌ has to be deleted
        "funded_amnt_inv": [1000, 1000, 2000],  # ❌ has to be deleted
        "next_pymnt_d": ["Jan-2019", "Jan-2019", "Feb-2020"],  # ❌ has to be deleted
        "Unnamed: 0.1": [123, 124, 125],  # ❌ has to be deleted
        "Unnamed: 0": [126, 126, 130],  # ❌ has to be deleted
        "loan_status": ["Fully Paid", "Charged Off", "Default"],  # ⚠️ ❌ converted to binary 'loan_status_binary'
        "addr_state": ["CA", "TX", "FL"],  # ❌ has to be deleted
        "last_pymnt_d": ["Jan-2019", "Jan-2019", "Feb-2020"],  # ❌ has to be deleted
        "last_credit_pull_d": ["Jan-2019", "Jan-2019", "Feb-2020"],  # ❌ has to be deleted
        "hardship_type": ["ST0650PV01", "ST0650PV02", "ST0650PV03"],  # ❌ has to be deleted
        "hardship_reason": ["Job loss", None, "Medical expenses"],  # ❌ has to be deleted
        "hardship_start_date": ["Jan-2020", None, "Mar-2020"],  # ❌ has to be deleted
        "hardship_end_date": ["Apr-2020", None, "Jun-2020"],  # ❌ has to be deleted
        "hardship_amount": [1000.0, None, 1500.0],  # ❌ has to be deleted
        "hardship_length": [3, None, 4],  # ❌ has to be deleted
        "deferral_term": [2, None, 3],  # ❌ has to be deleted
        "hardship_loan_status": ["Late (31-120 days)", None, "Current"],  # ⚠️ need feature extraction 'late'
        "hardship_payoff_balance_amount": [5000.0, None, 3200.0],  # ❌ has to be deleted
        "hardship_last_payment_amount": [150.0, None, 200.0],  # ❌ has to be deleted
        "orig_projected_additional_accrued_interest": [75.3, None, 45.0],  # ❌ has to be deleted
        "payment_plan_start_date": ["2020-03-01", None, "2020-06-15"],  # ❌ has to be deleted
        "debt_settlement_flag": ["N", "Y", "Y"],  # ❌ has to be deleted
        "sec_app_fico_range_low": [670, None, 720],  # ❌ has to be deleted
        "sec_app_fico_range_high": [690, None, 740],  # ❌ has to be deleted
        "sec_app_inq_last_6mths": [1, None, 0],  # ❌ has to be deleted
        "sec_app_mort_acc": [2, None, 1],  # ❌ has to be deleted
        "sec_app_open_acc": [5, None, 7],  # ❌ has to be deleted
        "sec_app_revol_util": ["45%", None, "30%"],  # ❌ has to be deleted
        "sec_app_open_act_il": [2, None, 3],  # ❌ has to be deleted
        "sec_app_num_rev_accts": [8, None, 10],  # ❌ has to be deleted
        "sec_app_chargeoff_within_12_mths": [0, None, 0],  # ❌ has to be deleted
        "sec_app_collections_12_mths_ex_med": [0, None, 1],  # ❌ has to be deleted
        "out_prncp": [5000, 4000, 0],  # ❌ has to be deleted
        "out_prncp_inv": [4900, 3900, 0],  # ❌ has to be deleted
        "total_pymnt": [7000, 8000, 15000],  # ❌ has to be deleted
        "total_pymnt_inv": [6900, 7900, 15000],  # ❌ has to be deleted
        "total_rec_prncp": [5000, 6000, 15000],  # ❌ has to be deleted
        "total_rec_int": [2000, 1800, 0],  # ❌ has to be deleted
        "total_rec_late_fee": [0.0, 10.5, 0.0],  # ❌ has to be deleted
        "recoveries": [0.0, 100.0, 0.0],  # ❌ has to be deleted
        "collection_recovery_fee": [0.0, 20.0, 0.0],  # ❌ has to be deleted
        "last_pymnt_amnt": [500.0, 400.0, 0.0],  # ❌ has to be deleted
        "last_fico_range_high": [690, 705, 730],  # ❌ has to be deleted
        "last_fico_range_low": [685, 700, 725],  # ❌ has to be deleted
        "collections_12_mths_ex_med": [0, 0, 1],  # ❌ has to be deleted
        "open_il_24m": [2, 1, 1],  # ❌ has to be deleted
        "mths_since_rcnt_il": [6, 12, 3],  # ❌ has to be deleted
        "total_bal_il": [5000, 4000, 6000],  # ❌ has to be deleted
        "il_util": [60.0, 55.0, 65.0],  # ❌ has to be deleted
        "open_rv_12m": [3, 2, 1],  # ❌ has to be deleted
        "open_rv_24m": [5, 3, 4],  # ❌ has to be deleted
        "max_bal_bc": [2000, 1500, 3000],  # ❌ has to be deleted
        "all_util": [35.0, 40.0, 30.0],  # ❌ has to be deleted
        "total_rev_hi_lim": [15000, 13000, 16000],  # ❌ has to be deleted
        "inq_fi": [1, 0, 1],  # ❌ has to be deleted
        "total_cu_tl": [4, 3, 5],  # ❌ has to be deleted
        "inq_last_12m": [2, 1, 3],  # ❌ has to be deleted
        "acc_open_past_24mths": [6, 5, 7],  # ❌ has to be deleted
        "bc_open_to_buy": [2000, 1500, 2500],  # ❌ has to be deleted
        "bc_util": [28.0, 32.0, 25.0],  # ❌ has to be deleted
        "chargeoff_within_12_mths": [0, 0, 0],  # ❌ has to be deleted
        "delinq_amnt": [0, 100, 0],  # ❌ has to be deleted
        "mo_sin_old_il_acct": [100, 120, 80],  # ❌ has to be deleted
        "mo_sin_old_rev_tl_op": [90, 110, 85],  # ❌ has to be deleted
        "mo_sin_rcnt_rev_tl_op": [10, 15, 5],  # ❌ has to be deleted
        "mo_sin_rcnt_tl": [5, 8, 3],  # ❌ has to be deleted
        "mort_acc": [1, 0, 2],  # ❌ has to be deleted
        "mths_since_recent_bc": [6, 9, 4],  # ❌ has to be deleted
        "num_accts_ever_120_pd": [0, 1, 0],  # ❌ has to be deleted
        "num_actv_bc_tl": [3, 2, 4],  # ❌ has to be deleted
        "num_actv_rev_tl": [5, 4, 6],  # ❌ has to be deleted
        "num_bc_sats": [4, 3, 5],  # ❌ has to be deleted
        "num_bc_tl": [6, 5, 7],  # ❌ has to be deleted
        "num_il_tl": [7, 6, 8],  # ❌ has to be deleted
        "num_op_rev_tl": [6, 5, 7],  # ❌ has to be deleted
        "num_rev_accts": [12, 10, 14],  # ❌ has to be deleted
        "num_rev_tl_bal_gt_0": [8, 7, 9],  # ❌ has to be deleted
        "num_sats": [10, 9, 11],  # ❌ has to be deleted
        "num_tl_120dpd_2m": [0, 0, 0],  # ❌ has to be deleted
        "num_tl_30dpd": [0, 1, 0],  # ❌ has to be deleted
        "num_tl_90g_dpd_24m": [0, 1, 0],  # ❌ has to be deleted
        "pct_tl_nvr_dlq": [100.0, 95.0, 98.0],  # ❌ has to be deleted
        "percent_bc_gt_75": [20.0, 30.0, 15.0],  # ❌ has to be deleted
        "tot_hi_cred_lim": [25000, 22000, 27000],  # ❌ has to be deleted
        "total_bal_ex_mort": [20000, 18000, 23000],  # ❌ has to be deleted
        "total_bc_limit": [8000, 7000, 9000],  # ❌ has to be deleted
        "total_il_high_credit_limit": [12000, 10000, 14000],  # ❌ has to be deleted
        "hardship_status": [None, "COMPLETED", None],  # ❌ has to be deleted
        "loan_amnt": [1000, 1000, 2000],  # ✅ retain => positive
        "annual_inc": [50000, 50000, 60000],  # ✅ retain => caped to 0.99 quantile
        "annual_inc_joint": [95000, None, 110000],  # ✅ retain => caped to 0.99 quantile
        "dti": [15.0, 15.0, 20.0],  # ✅ retain => caped max=100
        "dti_joint": [20.0, 50.0, 20.0],  # ✅ retain => caped max=100
        "int_rate": ["10.5%", "10.5%", "12.3%"],  # ✅ retain => convert to float
        "term": ["36 months", "36 months", "60 months"],  # ✅ retain => convert to int
        "emp_length": ["1 year", "n/a", "10+ years"],  # ✅ retain
        "issue_d": ["Jan-2019", "Jan-2019", "Feb-2020"],  # ✅ retain
        "earliest_cr_line": ["Jan-2019", "Jan-2019", "Feb-2020"],  # ✅ retain
        "purpose": ["credit_card", "debt_consolidation", "home_improvement"],  # ✅ retain
        "home_ownership": ["MORTGAGE", "ANY", "NONE"],  # ✅ 🧠 remain and processed at this stage
        "revol_util": ["47%", "45%", "49%"],  # ✅ retain
        "initial_list_status": ["w", "W", "f"],  # ✅ retain
        "is_joint_app": [0, 0, 1],  # ✅ retain => bool
        "verification_status": ["Not Verified", "Source Verified", "Verified"],  # ✅ retain => lower case
        "verification_status_joint": ["Not Verified", "Source Verified", "Verified"],  # ✅ retain => lower case
        "sec_app_earliest_cr_line": ["Jan-2019", "Jan-2019", "Feb-2020"],  # ✅ retain => lower case
        "hardship_flag": ["Y", "N", "N"],  # ✅ retain => lower case
        "revol_bal": [6244.0, 6244.0, 100000.0],  # ✅ retain => float
        "revol_bal_joint": [2244.0, 99244.0, 100300.0],  # ✅ retain => float
        "hardship_dpd": [10.0, None, 35.0],  # ✅ retain => float
        "mths_since_last_record": [70.0, 30.0, None],  # ✅ retain => float
        "funded_amnt": [10000, 12000, 15000],  # ✅ retain => int
        "installment": [300.45, 350.12, 400.00],  # ✅ retain => int
        "grade": ["B", "C", "A"],  # ✅ retain
        "sub_grade": ["B3", "C2", "A5"],  # ✅ retain
        "delinq_2yrs": [0, 1, 0],  # ✅ retain => int
        "fico_range_low": [680, 700, 720],  # ✅ retain
        "fico_range_high": [684, 704, 724],  # ✅ retain
        "inq_last_6mths": [1, 0, 2],  # ✅ retain
        "mths_since_last_delinq": [10, None, 24],  # ✅ retain => float
        "open_acc": [12, 9, 15],  # ✅ retain => int
        "pub_rec": [0, 0, 1],  # ✅ retain => int
        "total_acc": [25, 20, 30],  # ✅ retain => int
        "acc_now_delinq": [0.0, 1.0, 10.0],  # ✅ retain => float
        "tot_coll_amt": [0, 200, 0],  # ✅ retain => int
        "tot_cur_bal": [15000, 12000, 18000],  # ✅ retain => int
        "open_act_il": [1, 2, 3],  # ✅ retain => int
        "open_il_12m": [1.0, 5.0, 7.0],  # ✅ retain => int
        "open_acc_6m": [2, 3, 1],  # ✅ retain => int
        "pub_rec_bankruptcies": [0, 1, 0],  # ✅ retain
        "tax_liens": [0, 0, 0],  # ✅ retain
        "mths_since_recent_bc_dlq": [None, 12, 6],  # ✅ retain
        "mths_since_recent_inq": [4, 7, 2],  # ✅ retain
        "mths_since_recent_revol_delinq": [None, 18, None],  # ✅ retain
        "avg_cur_bal": [5000, 4000, 6000],  # ✅ retain => float
        "num_tl_op_past_12m": [4, 3, 5],  # ✅ retain
        "mths_since_last_major_derog": [None, 60, 12],  # ✅ retain
    })

    catalog = DataCatalog({
        "raw_data": MemoryDataset(raw_data),  # change MemoryDataset(raw_data) to 'sample' or 'lendingclub'
        "params:admin_columns_to_drop": MemoryDataset(columns_to_drop),
        "intermediate_data": MemoryDataset(),
        "dedup_flag": MemoryDataset(),  # capture intermediate deduplication result
    })

    pipeline = create_pipeline()
    runner = SequentialRunner()
    output = runner.run(pipeline, catalog)

    # Check outputs exist
    assert "intermediate_data" in output

    processed = output["intermediate_data"]  # resulting dataset of the preprocessing pipeline in the /base/catalog
    assert isinstance(processed, pd.DataFrame)

    # Get deduplication flag
    deduped_success = output["dedup_flag"]
    # Check deduplication BEFORE 'id' is dropped
    assert deduped_success is True

    # check if fields exist
    for col in ["loan_amnt", "annual_inc", "dti", "term", "int_rate", "emp_length", "issue_d",
                "purpose", "home_ownership", "verification_status", "verification_status_joint",
                "hardship_flag", "sec_app_earliest_cr_line", "earliest_cr_line", "hardship_loan_status",
                "initial_list_status", "revol_util", "annual_inc_joint", "dti_joint", "revol_bal", "revol_bal_joint",
                "hardship_dpd", "mths_since_last_record", "is_joint_app", "funded_amnt", "installment", "grade",
                "sub_grade", "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
                "mths_since_last_delinq", "open_acc", "pub_rec", "total_acc", "open_acc_6m",
                "mths_since_last_major_derog"]:
        assert col in processed.columns, f"Expected column {col} is missing"

    # fields to delete
    for col in columns_to_drop:
        assert col not in processed.columns, f"{col} was not dropped"

    # loan_amnt
    assert pd.api.types.is_numeric_dtype(processed['loan_amnt'])
    assert (processed['loan_amnt'] > 0).all()
    print(processed["loan_amnt"].head())

    # pub_rec_bankruptcies
    assert "pub_rec_bankruptcies" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["pub_rec_bankruptcies"])
    assert (processed["pub_rec_bankruptcies"].dropna() >= 0).all()

    # tax_liens
    assert "tax_liens" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["tax_liens"])
    assert (processed["tax_liens"].dropna() >= 0).all()

    # mths_since_recent_bc_dlq
    assert "mths_since_recent_bc_dlq" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["mths_since_recent_bc_dlq"])
    # allow NaNs for this field
    assert processed["mths_since_recent_bc_dlq"].dropna().between(0, 300).all()

    # mths_since_recent_inq
    assert "mths_since_recent_inq" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["mths_since_recent_inq"])
    assert (processed["mths_since_recent_inq"].dropna() >= 0).all()

    # mths_since_recent_revol_delinq
    assert "mths_since_recent_revol_delinq" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["mths_since_recent_revol_delinq"])
    # allow NaNs, check value range if present
    assert processed["mths_since_recent_revol_delinq"].dropna().between(0, 300).all()

    # avg_cur_bal
    assert "avg_cur_bal" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["avg_cur_bal"])
    assert (processed["avg_cur_bal"].dropna() >= 0).all()

    # num_tl_op_past_12m
    assert "num_tl_op_past_12m" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["num_tl_op_past_12m"])
    assert (processed["num_tl_op_past_12m"].dropna() >= 0).all()

    # acc_now_delinq
    assert "acc_now_delinq" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["acc_now_delinq"])
    assert (processed["acc_now_delinq"].dropna() >= 0).all()

    # tot_coll_amt
    assert "tot_coll_amt" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["tot_coll_amt"])
    assert (processed["tot_coll_amt"].dropna() >= 0).all()

    # tot_cur_bal
    assert "tot_cur_bal" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["tot_cur_bal"])
    assert (processed["tot_cur_bal"].dropna() >= 0).all()

    # open_act_il
    assert "open_act_il" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["open_act_il"])
    assert (processed["open_act_il"].dropna() >= 0).all()

    # open_il_12m
    assert "open_il_12m" in processed.columns
    assert pd.api.types.is_numeric_dtype(processed["open_il_12m"])
    assert (processed["open_il_12m"].dropna() >= 0).all()

    # open_acc_6m
    assert pd.api.types.is_numeric_dtype(processed["open_acc_6m"])
    assert (processed["open_acc_6m"].dropna() >= 0).all()

    # is_joint_app
    assert pd.api.types.is_integer_dtype(processed["is_joint_app"]), "'is_joint_app' should be an integer (0/1)"
    # Values are only 0 or 1
    assert set(processed["is_joint_app"].dropna().unique()) <= {0, 1}, "Unexpected values in 'is_joint_app'"

    # annual_inc
    assert pd.api.types.is_numeric_dtype(processed['annual_inc'])
    assert (processed['annual_inc'] > 0).all()

    # annual_inc_joint
    assert pd.api.types.is_numeric_dtype(processed["annual_inc_joint"])
    assert (processed['annual_inc_joint'].dropna() >= 0).all()

    # # Check column exists before validating its values
    # if "annual_inc" in processed.columns and not processed["annual_inc"].empty:
    #     annual_inc_cap = raw_data["annual_inc"].quantile(0.99)
    #     assert processed["annual_inc"].max() <= annual_inc_cap
    #
    # if "annual_inc_joint" in processed.columns and not processed["annual_inc_joint"].dropna().empty:
    #     annual_inc_joint_cap = raw_data["annual_inc_joint"].quantile(0.99)
    #     assert processed["annual_inc_joint"].dropna().max() <= annual_inc_joint_cap

    # dti
    assert pd.api.types.is_numeric_dtype(processed["dti"])
    assert (processed["dti"] >= 0).all()
    assert (processed["dti"] <= 100).all()  # Adjust threshold if needed

    # int_rate
    assert processed["int_rate"].dtype == float

    # term
    assert pd.api.types.is_integer_dtype(processed["term"])
    assert set(processed["term"].unique()) <= {36, 60}, "Unexpected values in 'term'"

    # emp_length
    assert "n/a" not in processed["emp_length"].fillna("").values

    # issue_d
    assert pd.api.types.is_datetime64_any_dtype(processed['issue_d'])
    assert processed['issue_d'].notnull().all()  # or .any() if you expect nulls

    # earliest_cr_line
    assert pd.api.types.is_datetime64_any_dtype(processed['earliest_cr_line'])
    assert processed['earliest_cr_line'].notnull().all()  # or .any() if you expect nulls

    # loan_status
    assert "loan_status_binary" in processed.columns
    assert processed["loan_status_binary"].dropna().isin([0, 1]).all()  # Check binary target created

    # purpose
    assert processed["purpose"].str.islower().all()
    assert processed["purpose"].str.contains("_").sum() == 0

    # home_ownership
    assert processed["home_ownership"].str.islower().all()
    assert processed["home_ownership"].str.strip().eq(processed["home_ownership"]).all()
    assert not processed["home_ownership"].isin(["none", "any", "unknown"]).any()
    expected_categories = {"rent", "own", "mortgage", "other"}
    assert set(processed["home_ownership"].dropna().unique()).issubset(expected_categories)

    # revol_util
    if "revol_util" in processed.columns:
        print(processed["revol_util"].head())  # Debug
        assert pd.api.types.is_numeric_dtype(processed["revol_util"])
        assert processed["revol_util"].dropna().between(0, 100).all()
    else:
        raise AssertionError("Expected 'revol_util' column is missing in processed dataset")

    assert processed["revol_util"].dropna().between(0, 100).all()

    # initial_list_status
    assert processed["initial_list_status"].str.strip().eq(processed["initial_list_status"]).all()
    assert processed["initial_list_status"].isin(["f", "w"]).all()

    # sec_app_earliest_cr_line
    if "sec_app_earliest_cr_line" in processed.columns:
        assert pd.api.types.is_datetime64_any_dtype(processed["sec_app_earliest_cr_line"])

    # dti_joint
    assert pd.api.types.is_numeric_dtype(processed["dti_joint"])
    assert processed["dti_joint"].dropna().between(0, 100).all()

    # verification_status and verification_status_joint
    expected_values = {"not verified", "source verified", "verified"}
    for col in ["verification_status", "verification_status_joint"]:
        if col in processed.columns:
            assert processed[col].dropna().isin(expected_values).all()

    # revol_bal and revol_bal_joint
    for col in ["revol_bal", "revol_bal_joint"]:
        if col in processed.columns:
            assert pd.api.types.is_numeric_dtype(processed[col])
            assert processed[col].dropna().between(0, processed[col].quantile(0.99)).all()

    # hardship_dpd
    assert pd.api.types.is_numeric_dtype(processed["hardship_dpd"])

    # mths_since_last_record
    assert pd.api.types.is_numeric_dtype(processed["mths_since_last_record"])
    # Just check that no exceptions occur due to mixed types
    processed["mths_since_last_record"].dropna().apply(lambda x: isinstance(x, (int, float)))
    if "mths_since_last_record" in processed.columns and not processed["mths_since_last_record"].empty:
        # Validate type and expect NaNs only if column exists and has rows
        assert pd.api.types.is_numeric_dtype(processed["mths_since_last_record"])
        assert processed["mths_since_last_record"].isna().any(), "Expected some NaNs in mths_since_last_record"
    else:
        print("Column 'mths_since_last_record' is missing or empty — skipping NaN test.")

    # assert processed["mths_since_last_record"].isna().any()  # Expect NaNs at this point

    # funded_amnt
    assert pd.api.types.is_numeric_dtype(processed["funded_amnt"])
    assert (processed["funded_amnt"] > 0).all(), "'funded_amnt' should be positive"

    # installment
    assert pd.api.types.is_numeric_dtype(processed["installment"])
    assert (processed["installment"] > 0).all(), "'installment' should be positive"

    # grade
    assert processed["grade"].dropna().str.isalpha().all(), "'grade' should contain alphabetic values"
    assert processed["grade"].str.len().eq(1).all(), "'grade' should be a single letter"

    # sub_grade
    assert processed["sub_grade"].dropna().str.len().between(2, 3).all(), "'sub_grade' should be 2-3 characters long"
    assert processed["sub_grade"].str[0].isin(processed["grade"]).all(), "'sub_grade' prefix should match 'grade'"

    # delinq_2yrs
    assert pd.api.types.is_numeric_dtype(processed["delinq_2yrs"])
    assert (processed["delinq_2yrs"] >= 0).all(), "'delinq_2yrs' should be non-negative"

    # fico_range_low
    assert pd.api.types.is_numeric_dtype(processed["fico_range_low"])
    assert (processed["fico_range_low"] >= 300).all()

    # fico_range_high
    assert pd.api.types.is_numeric_dtype(processed["fico_range_high"])
    assert (processed["fico_range_high"] >= processed["fico_range_low"]).all()

    # inq_last_6mths
    assert pd.api.types.is_numeric_dtype(processed["inq_last_6mths"])
    assert (processed["inq_last_6mths"] >= 0).all()

    # mths_since_last_delinq
    assert pd.api.types.is_float_dtype(processed["mths_since_last_delinq"])
    assert processed["mths_since_last_delinq"].isna().sum() >= 0  # allow missing

    # open_acc
    assert pd.api.types.is_numeric_dtype(processed["open_acc"])
    assert (processed["open_acc"] >= 0).all()

    # pub_rec
    assert pd.api.types.is_numeric_dtype(processed["pub_rec"])
    assert processed["pub_rec"].isin([0, 1]).all(), "'pub_rec' should be binary"

    # total_acc
    assert pd.api.types.is_numeric_dtype(processed["total_acc"])
    assert (processed["total_acc"] >= 0).all()

    # mths_since_last_major_derog
    assert pd.api.types.is_numeric_dtype(processed["mths_since_last_major_derog"])
    assert processed["mths_since_last_major_derog"].isna().sum() == 0
    assert processed["mths_since_last_major_derog"].between(0, 300).all()

    print(processed.head())

    ProfileReport(raw_data, title="Raw Data").to_file("data/08_reporting/raw_report.html")
    ProfileReport(processed, title="Processed Data").to_file("data/08_reporting/processed_report.html")
