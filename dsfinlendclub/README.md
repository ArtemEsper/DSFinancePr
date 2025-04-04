# LendingClub Credit Risk Modeling Project

This project is built using [Kedro](https://github.com/kedro-org/kedro), an opinionated Python framework for reproducible, maintainable, and modular data science code. It focuses on analyzing and modeling LendingClub loan data, with pipelines structured from initial data processing through feature engineering.

> 🤖 **Note**: The development of this pipeline was assisted by [Claude AI](https://claude.ai), with design patterns and implementation strategies adapted from the excellent guide in [this Medium blog post](https://medium.com/@DaveThackeray/claude-3-7-sonnet-the-first-ai-model-that-understands-your-entire-codebase-560915c6a703).

---
---

## ✅ Testing

Robust unit tests have been written to ensure correctness and pipeline integrity:

- `tests/pipelines/data_processing/test_pipeline.py`  
  Verifies preprocessing logic, data cleaning, and column consistency.

- `tests/pipelines/feature_engineering/test_pipeline.py`  
  Includes comprehensive checks for:
  - Feature creation (joint features, hardship flags, encodings)
  - Model-specific transformations (tree vs regression)
  - No data leakage or invalid types
  - Consistent outputs from intermediate to final datasets

Run tests via:

```bash
pytest tests/
```
---

---

## 🔄 Data Flow

| Stage                  | Input                   | Output                                 | Location                |
|------------------------|--------------------------|----------------------------------------|------------------------|
| `01_raw`               | Raw LendingClub data    | `raw_data: Loan_status_2007-2020Q3`    | `data/01_raw/`          |
| `02_intermediate`      | Deduplicated + cleaned  | `preprocessed_lc_data`                 | `data/02_intermediate/` |
| `04_feature`           | Feature-engineered set  | `tree_features`, `regression_features` | `data/04_feature/`      |

---

## ⚙️ Data Processing Pipeline

Defined in `pipelines/data_processing`. This pipeline prepares the raw LendingClub data for modeling.

### 🔧 Steps:
1. **Remove Duplicates**
   - Drops duplicate entries based on the `id` column.
2. **Drop Administrative Columns**
   - Removes irrelevant fields like `member_id`, `url`, `desc`, `zip_code`, etc.
   - Column list managed via `parameters_data_processing.yml`
3. **Normalize Column Names**
   - Lowercases and trims all column names for consistency.
4. **Fix Column Types**
   - Converts `term` and `int_rate` to numeric; parses dates.
5. **Remove Invalid Rows**
   - Drops rows with invalid values (e.g., negative income or loan amount).
6. **Filter Loan Status**
   - Keeps only rows with known final outcomes: `Fully Paid`, `Charged Off`, or `Default`.
7. **Clean String Fields**
   - Cleans `emp_length`, `purpose`, and other key strings.
8. **Cap Outliers**
   - Caps extreme values (e.g., `annual_inc`) at the 99th percentile.

### 📤 Output:
- Cleaned data is stored in `02_intermediate/` as `data_processing_output`

---

## 🛠 Feature Engineering Pipeline

Defined in `pipelines/feature_engineering`. This pipeline generates model-ready features from the cleaned dataset.

### 🔧 Example Steps:
- **Employment Length Encoding**
  - Creates both `emp_length_clean_tree` (`-1` fill) and `emp_length_clean_reg` (median imputed) for tree vs regression models.
- **Term Encoding**
  - Converts term from `"36 months"`/`"60 months"` to numeric or binary.
- **Purpose Encoding**
  - One-hot or target encodes the `purpose` field, with the option to use both versions in different models.
- **Loan Status Target Creation**
  - Generates binary target `loan_status_binary` for supervised learning.

### 📤 Output:
- Feature matrices are stored in `04_feature/` as:
  - `feature_data_tree`
  - `feature_data_reg`

---

## 📦 Parameters

Located in `conf/base/`:

- `parameters_data_processing.yml`: defines columns to drop, deduplication rules
- `parameters_feature_engineering.yml`: controls encoding strategies, mappings, etc.

---

## 🚀 Running the Pipelines

To run only data processing:

```bash
kedro run --pipeline data_processing
kedro run --pipeline feature_engineering
```

