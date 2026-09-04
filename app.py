import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Decision Tree Model Builder",
    layout="wide"
)

st.title("Decision Tree Model Builder")

st.write(
    """
    Upload a dataset, select the outcome and predictor variables,
    use cross-validation to select the optimal tree depth,
    and evaluate the final decision tree.
    """
)


# ============================================================
# HELPER FUNCTION: CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, cutoff):

    predictions = (probabilities >= cutoff).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    misclassification = 1 - accuracy

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )
    except ValueError:
        roc_auc = np.nan

    metrics = {
        "Accuracy": accuracy,
        "Misclassification Error": misclassification,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "False Positive Rate": false_positive_rate,
        "False Negative Rate": false_negative_rate,
        "ROC AUC": roc_auc
    }

    return metrics, predictions


# ============================================================
# HELPER FUNCTION: GET MODEL-SELECTION METRIC
# ============================================================

def get_selected_metric(
    y_true,
    probabilities,
    metric_name
):

    # Cross-validation uses cutoff = 0.50 for
    # cutoff-dependent metrics.

    metrics, _ = calculate_metrics(
        y_true,
        probabilities,
        cutoff=0.50
    )

    return metrics[metric_name]


# ============================================================
# 1. UPLOAD DATA
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write(
        f"Rows: **{df.shape[0]}** | "
        f"Columns: **{df.shape[1]}**"
    )


    # ========================================================
    # 2. SELECT OUTCOME
    # ========================================================

    st.subheader("1. Select Outcome Variable")

    target = st.selectbox(
        "Which column are you predicting?",
        options=df.columns
    )


    # ========================================================
    # 3. SELECT PREDICTORS
    # ========================================================

    st.subheader("2. Select Predictor Variables")

    available_predictors = [
        col
        for col in df.columns
        if col != target
    ]

    predictors = st.multiselect(
        "Select the columns to use as predictors:",
        options=available_predictors
    )


    # ========================================================
    # 4. TRAIN / TEST SPLIT
    # ========================================================

    st.subheader("3. Select Training and Testing Split")

    test_percent = st.selectbox(
        "Percentage of observations used for testing:",
        options=[20, 30, 40],
        index=1
    )

    test_size = test_percent / 100


    # ========================================================
    # 5. TREE DEPTH
    # ========================================================

    st.subheader("4. Select Tree Depth Range")

    col1, col2 = st.columns(2)

    with col1:

        min_depth = st.number_input(
            "Minimum tree depth",
            min_value=1,
            max_value=50,
            value=1,
            step=1
        )

    with col2:

        max_depth = st.number_input(
            "Maximum tree depth",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )


    # ========================================================
    # 6. MODEL-SELECTION METRIC
    # ========================================================

    st.subheader("5. Select Model-Selection Metric")

    metric_name = st.selectbox(
        "Metric used to select the optimal tree depth:",
        options=[
            "Accuracy",
            "Misclassification Error",
            "F1 Score",
            "ROC AUC",
            "False Positive Rate",
            "False Negative Rate",
            "Recall",
            "Precision"
        ]
    )


    cv_folds = st.selectbox(
        "Number of cross-validation folds:",
        options=[3, 5, 10],
        index=1
    )


    st.info(
        """
        Cross-validation is performed only on the training data.

        For Accuracy, Misclassification Error, F1 Score,
        False Positive Rate, False Negative Rate, Recall,
        and Precision, a cutoff of 0.50 is used during
        cross-validation.

        ROC AUC does not depend on a classification cutoff.
        """
    )


    # ========================================================
    # BUILD MODEL BUTTON
    # ========================================================

    if st.button("Build Decision Tree"):

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if len(predictors) == 0:

            st.error(
                "Please select at least one predictor variable."
            )

            st.stop()


        if min_depth > max_depth:

            st.error(
                "Minimum tree depth cannot be larger "
                "than maximum tree depth."
            )

            st.stop()


        y_original = df[target]

        valid_rows = y_original.notna()

        model_df = df.loc[
            valid_rows
        ].copy()

        X = model_df[predictors].copy()

        y = model_df[target].copy()


        # Outcome must be exactly 0 and 1

        unique_values = set(
            y.unique()
        )

        if unique_values != {0, 1}:

            st.error(
                """
                The outcome variable must contain exactly
                two classes coded as 0 and 1.

                0 = Negative class

                1 = Positive class
                """
            )

            st.stop()


        # ====================================================
        # TRAIN / TEST SPLIT
        # ====================================================

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=y,
                random_state=42
            )
        )


        st.subheader("Training and Testing Samples")

        split_col1, split_col2 = st.columns(2)

        with split_col1:

            st.metric(
                "Training Observations",
                len(X_train)
            )

        with split_col2:

            st.metric(
                "Testing Observations",
                len(X_test)
            )


        # Check CV feasibility

        smallest_class = (
            y_train.value_counts().min()
        )

        if smallest_class < cv_folds:

            st.error(
                f"""
                There are not enough observations in the
                smaller outcome class for {cv_folds}-fold
                cross-validation.

                Please choose fewer folds.
                """
            )

            st.stop()


        # ====================================================
        # PREPROCESSING
        # ====================================================

        numeric_features = (
            X_train
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        categorical_features = [
            col
            for col in X_train.columns
            if col not in numeric_features
        ]


        numeric_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]
        )


        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )


        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_transformer,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_transformer,
                    categorical_features
                )
            ]
        )


        # ====================================================
        # CROSS-VALIDATION
        # ====================================================

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42
        )

        results = []


        for depth in range(
            int(min_depth),
            int(max_depth) + 1
        ):

            fold_scores = []


            for train_index, validation_index in cv.split(
                X_train,
                y_train
            ):

                X_cv_train = X_train.iloc[
                    train_index
                ]

                X_cv_validation = X_train.iloc[
                    validation_index
                ]

                y_cv_train = y_train.iloc[
                    train_index
                ]

                y_cv_validation = y_train.iloc[
                    validation_index
                ]


                model = DecisionTreeClassifier(
                    max_depth=depth,
                    random_state=42
                )


                pipeline = Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            preprocessor
                        ),
                        (
                            "model",
                            model
                        )
                    ]
                )


                pipeline.fit(
                    X_cv_train,
                    y_cv_train
                )


                validation_probabilities = (
                    pipeline.predict_proba(
                        X_cv_validation
                    )[:, 1]
                )


                fold_metric = get_selected_metric(
                    y_cv_validation,
                    validation_probabilities,
                    metric_name
                )


                fold_scores.append(
                    fold_metric
                )


            results.append(
                {
                    "Tree Depth": depth,
                    metric_name: np.mean(
                        fold_scores
                    )
                }
            )


        results_df = pd.DataFrame(
            results
        )


        # ====================================================
        # SELECT OPTIMAL DEPTH
        # ====================================================

        metrics_to_minimize = [
            "Misclassification Error",
            "False Positive Rate",
            "False Negative Rate"
        ]


        if metric_name in metrics_to_minimize:

            best_index = (
                results_df[
                    metric_name
                ].idxmin()
            )

            selection_word = "lowest"

        else:

            best_index = (
                results_df[
                    metric_name
                ].idxmax()
            )

            selection_word = "highest"


        best_depth = int(
            results_df.loc[
                best_index,
                "Tree Depth"
            ]
        )

        best_score = (
            results_df.loc[
                best_index,
                metric_name
            ]
        )


        # ====================================================
        # SHOW CROSS-VALIDATION RESULTS
        # ====================================================

        st.subheader(
            "Cross-Validation Results"
        )


        display_results = (
            results_df.copy()
        )

        display_results[
            metric_name
        ] = (
            display_results[
                metric_name
            ].round(4)
        )


        st.dataframe(
            display_results,
            use_container_width=True
        )


        st.subheader(
            f"{metric_name} vs. Tree Depth"
        )


        chart_df = (
            results_df
            .set_index("Tree Depth")
        )


        st.line_chart(
            chart_df
        )


        st.success(
            f"""
            Optimal tree depth = **{best_depth}**

            It was selected because tree depth
            **{best_depth}** produced the **{selection_word}
            mean cross-validated {metric_name}**
            across the candidate tree depths.
            """
        )


        st.metric(
            f"Cross-Validated {metric_name}",
            f"{best_score:.3f}"
        )


        # ====================================================
        # FIT FINAL MODEL
        # ====================================================

        final_model = DecisionTreeClassifier(
            max_depth=best_depth,
            random_state=42
        )


        final_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    final_model
                )
            ]
        )


        final_pipeline.fit(
            X_train,
            y_train
        )


        # ====================================================
        # SHOW DECISION TREE
        # ====================================================

        st.subheader(
            "Optimal Decision Tree"
        )


        fitted_preprocessor = (
            final_pipeline
            .named_steps[
                "preprocessor"
            ]
        )


        transformed_feature_names = (
            fitted_preprocessor
            .get_feature_names_out()
        )


        tree_model = (
            final_pipeline
            .named_steps["model"]
        )


        fig, ax = plt.subplots(
            figsize=(20, 10)
        )


        plot_tree(
            tree_model,
            feature_names=transformed_feature_names,
            class_names=["0", "1"],
            filled=True,
            rounded=True,
            proportion=False,
            ax=ax
        )


        st.pyplot(fig)


        # ====================================================
        # VARIABLE IMPORTANCE
        # ====================================================

        st.subheader(
            "Variable Importance"
        )


        transformed_importance = (
            tree_model.feature_importances_
        )


        importance_by_variable = {
            predictor: 0.0
            for predictor in predictors
        }


        for feature_name, importance in zip(
            transformed_feature_names,
            transformed_importance
        ):

            clean_name = (
                feature_name
                .replace(
                    "numeric__",
                    ""
                )
                .replace(
                    "categorical__",
                    ""
                )
            )


            for predictor in predictors:

                if (
                    clean_name == predictor
                    or clean_name.startswith(
                        predictor + "_"
                    )
                ):

                    importance_by_variable[
                        predictor
                    ] += importance

                    break


        importance_df = pd.DataFrame(
            {
                "Variable": list(
                    importance_by_variable.keys()
                ),
                "Importance": list(
                    importance_by_variable.values()
                )
            }
        )


        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )


        importance_df[
            "Importance"
        ] = (
            importance_df[
                "Importance"
            ].round(4)
        )


        st.dataframe(
            importance_df,
            use_container_width=True
        )


        st.bar_chart(
            importance_df.set_index(
                "Variable"
            )
        )


        # ====================================================
        # CUTOFF
        # ====================================================

        st.subheader(
            "Classification Cutoff"
        )


        cutoff = st.number_input(
            "Enter classification cutoff:",
            min_value=0.00,
            max_value=1.00,
            value=0.50,
            step=0.01,
            format="%.2f"
        )


        st.write(
            f"""
            An observation is classified as **1**
            when its predicted probability is
            greater than or equal to **{cutoff:.2f}**.

            Otherwise, it is classified as **0**.
            """
        )


        # ====================================================
        # TRAINING PERFORMANCE
        # ====================================================

        train_probabilities = (
            final_pipeline
            .predict_proba(
                X_train
            )[:, 1]
        )


        train_metrics, train_predictions = (
            calculate_metrics(
                y_train,
                train_probabilities,
                cutoff
            )
        )


        train_cm = confusion_matrix(
            y_train,
            train_predictions,
            labels=[0, 1]
        )


        # ====================================================
        # TEST PERFORMANCE
        # ====================================================

        test_probabilities = (
            final_pipeline
            .predict_proba(
                X_test
            )[:, 1]
        )


        test_metrics, test_predictions = (
            calculate_metrics(
                y_test,
                test_probabilities,
                cutoff
            )
        )


        test_cm = confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1]
        )


        # ====================================================
        # CONFUSION MATRICES
        # ====================================================

        st.subheader(
            "Confusion Matrices"
        )


        cm_col1, cm_col2 = st.columns(2)


        with cm_col1:

            st.write(
                "**Training Set**"
            )

            train_cm_df = pd.DataFrame(
                train_cm,
                index=[
                    "Actual 0",
                    "Actual 1"
                ],
                columns=[
                    "Predicted 0",
                    "Predicted 1"
                ]
            )

            st.dataframe(
                train_cm_df,
                use_container_width=True
            )


        with cm_col2:

            st.write(
                "**Testing Set**"
            )

            test_cm_df = pd.DataFrame(
                test_cm,
                index=[
                    "Actual 0",
                    "Actual 1"
                ],
                columns=[
                    "Predicted 0",
                    "Predicted 1"
                ]
            )

            st.dataframe(
                test_cm_df,
                use_container_width=True
            )


        # ====================================================
        # PERFORMANCE METRICS
        # ====================================================

        st.subheader(
            "Model Performance"
        )


        performance_df = pd.DataFrame(
            {
                "Metric": list(
                    train_metrics.keys()
                ),
                "Training Set": list(
                    train_metrics.values()
                ),
                "Testing Set": [
                    test_metrics[key]
                    for key in train_metrics.keys()
                ]
            }
        )


        performance_df[
            "Training Set"
        ] = (
            performance_df[
                "Training Set"
            ].round(4)
        )


        performance_df[
            "Testing Set"
        ] = (
            performance_df[
                "Testing Set"
            ].round(4)
        )


        st.dataframe(
            performance_df,
            use_container_width=True
        )


        # ====================================================
        # EXPORT DATA
        # ====================================================

        st.subheader(
            "Download Predictions"
        )


        train_output = model_df.loc[
            X_train.index
        ].copy()


        train_output[
            "Data_Set"
        ] = "Training"


        train_output[
            "Predicted_Probability"
        ] = train_probabilities


        train_output[
            "Predicted_Class"
        ] = train_predictions


        test_output = model_df.loc[
            X_test.index
        ].copy()


        test_output[
            "Data_Set"
        ] = "Testing"


        test_output[
            "Predicted_Probability"
        ] = test_probabilities


        test_output[
            "Predicted_Class"
        ] = test_predictions


        output_df = pd.concat(
            [
                train_output,
                test_output
            ]
        ).sort_index()


        st.write(
            f"""
            Predicted classifications in the downloaded
            file use a cutoff of **{cutoff:.2f}**.
            """
        )


        st.dataframe(
            output_df.head(20),
            use_container_width=True
        )


        csv = (
            output_df
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="decision_tree_predictions.csv",
            mime="text/csv"
        )
