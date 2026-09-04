import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="Decision Tree Model Builder",
    layout="wide"
)

st.title("Decision Tree Model Builder")

st.write(
    """
    Upload a dataset, select the outcome variable and predictor variables,
    and build a classification decision tree using cross-validation.
    """
)


# --------------------------------------------------
# Upload data
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    # Read the uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")


    # --------------------------------------------------
    # Select outcome variable
    # --------------------------------------------------

    st.subheader("1. Select Outcome Variable")

    target = st.selectbox(
        "Which column are you predicting?",
        options=df.columns
    )


    # --------------------------------------------------
    # Select predictor variables
    # --------------------------------------------------

    st.subheader("2. Select Predictor Variables")

    available_predictors = [
        col for col in df.columns
        if col != target
    ]

    predictors = st.multiselect(
        "Select the columns to use as predictors:",
        options=available_predictors
    )


    # --------------------------------------------------
    # Tree-depth range
    # --------------------------------------------------

    st.subheader("3. Select Tree Depth Range")

    col1, col2 = st.columns(2)

    with col1:

        min_depth = st.number_input(
            "Minimum depth",
            min_value=1,
            max_value=50,
            value=1
        )

    with col2:

        max_depth = st.number_input(
            "Maximum depth",
            min_value=1,
            max_value=50,
            value=10
        )


    # --------------------------------------------------
    # Cross-validation
    # --------------------------------------------------

    st.subheader("4. Cross-Validation")

    cv_folds = st.selectbox(
        "Number of folds:",
        options=[3, 5, 10],
        index=1
    )

    st.write("Model-selection metric: **F1 Score**")


    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    if st.button("Build Decision Tree"):

        if len(predictors) == 0:

            st.error(
                "Please select at least one predictor variable."
            )

        elif min_depth > max_depth:

            st.error(
                "Minimum depth cannot be larger than maximum depth."
            )

        else:

            X = df[predictors].copy()
            y = df[target].copy()

            # Remove observations with missing target values
            valid_rows = y.notna()

            X = X.loc[valid_rows]
            y = y.loc[valid_rows]


            # --------------------------------------------------
            # Check that outcome is binary
            # --------------------------------------------------

            if y.nunique() != 2:

                st.error(
                    """
                    This version currently supports binary
                    classification only.

                    The outcome variable must contain exactly
                    two classes.
                    """
                )

            else:

                # ----------------------------------------------
                # Identify numeric and categorical predictors
                # ----------------------------------------------

                numeric_features = (
                    X.select_dtypes(include=np.number)
                    .columns
                    .tolist()
                )

                categorical_features = [
                    col
                    for col in X.columns
                    if col not in numeric_features
                ]


                # ----------------------------------------------
                # Numeric preprocessing
                # ----------------------------------------------

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


                # ----------------------------------------------
                # Categorical preprocessing
                # ----------------------------------------------

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


                # ----------------------------------------------
                # Combine preprocessing
                # ----------------------------------------------

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


                # ----------------------------------------------
                # Cross-validation setup
                # ----------------------------------------------

                cv = StratifiedKFold(
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=42
                )

                results = []


                # ----------------------------------------------
                # Test every tree depth
                # ----------------------------------------------

                for depth in range(
                    int(min_depth),
                    int(max_depth) + 1
                ):

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

                    scores = cross_val_score(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        scoring="f1"
                    )

                    results.append(
                        {
                            "Max Depth": depth,
                            "Mean CV F1": scores.mean(),
                            "Std. Dev.": scores.std()
                        }
                    )


                # ----------------------------------------------
                # Store CV results
                # ----------------------------------------------

                results_df = pd.DataFrame(results)

                best_row = results_df.loc[
                    results_df[
                        "Mean CV F1"
                    ].idxmax()
                ]

                best_depth = int(
                    best_row["Max Depth"]
                )

                best_f1 = (
                    best_row["Mean CV F1"]
                )


                # ----------------------------------------------
                # Display cross-validation results
                # ----------------------------------------------

                st.success(
                    "Model selection completed."
                )

                st.subheader(
                    "Cross-Validation Results"
                )

                display_results = (
                    results_df.copy()
                )

                display_results[
                    "Mean CV F1"
                ] = (
                    display_results[
                        "Mean CV F1"
                    ].round(4)
                )

                display_results[
                    "Std. Dev."
                ] = (
                    display_results[
                        "Std. Dev."
                    ].round(4)
                )

                st.dataframe(
                    display_results,
                    use_container_width=True
                )

                st.metric(
                    "Selected Tree Depth",
                    best_depth
                )

                st.metric(
                    "Cross-Validated F1 Score",
                    f"{best_f1:.3f}"
                )

                st.write(
                    f"""
                    The model with **max_depth = {best_depth}**
                    had the highest mean cross-validated F1 score.
                    """
                )


                # ----------------------------------------------
                # Fit final model
                # ----------------------------------------------

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
                    X,
                    y
                )


                # ----------------------------------------------
                # Generate predictions
                # ----------------------------------------------

                predictions = (
                    final_pipeline.predict(X)
                )

                probabilities = (
                    final_pipeline.predict_proba(X)
                )

                classes = (
                    final_pipeline
                    .named_steps["model"]
                    .classes_
                )

                positive_class = classes[1]

                positive_probabilities = (
                    probabilities[:, 1]
                )


                # ----------------------------------------------
                # Create downloadable dataset
                # ----------------------------------------------

                output_df = (
                    df.loc[valid_rows].copy()
                )

                output_df[
                    "Predicted_Class"
                ] = predictions

                output_df[
                    f"Predicted_Probability_{positive_class}"
                ] = positive_probabilities


                # ----------------------------------------------
                # Show predictions
                # ----------------------------------------------

                st.subheader(
                    "Predictions"
                )

                st.dataframe(
                    output_df.head(20),
                    use_container_width=True
                )


                # ----------------------------------------------
                # Download predictions
                # ----------------------------------------------

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
