import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
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
    Build and evaluate a binary classification decision tree.
    The outcome variable must be coded as 0 and 1.
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "model_built" not in st.session_state:
    st.session_state.model_built = False

if "use_global_optimum" not in st.session_state:
    st.session_state.use_global_optimum = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, probabilities, cutoff):

    predictions = (
        probabilities >= cutoff
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
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

    try:
        pr_auc = average_precision_score(
            y_true,
            probabilities
        )
    except ValueError:
        pr_auc = np.nan

    try:
        probability_log_loss = log_loss(
            y_true,
            probabilities,
            labels=[0, 1]
        )
    except ValueError:
        probability_log_loss = np.nan

    metrics = {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Misclassification Error": misclassification,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
        "Log Loss": probability_log_loss,
        "False Positive Rate": false_positive_rate,
        "False Negative Rate": false_negative_rate,
        "Recall": recall,
        "Precision": precision
    }

    return metrics, predictions


def metric_value(
    y_true,
    probabilities,
    cutoff,
    metric_name,
    business_values=None
):

    if metric_name == "Business Value":

        predictions = (
            probabilities >= cutoff
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        return (
            tp * business_values["TP"]
            + fp * business_values["FP"]
            + fn * business_values["FN"]
            + tn * business_values["TN"]
        )

    metrics, _ = calculate_metrics(
        y_true,
        probabilities,
        cutoff
    )

    return metrics[metric_name]


def metric_is_percentage(metric_name):

    return metric_name not in {
        "Business Value",
        "Log Loss"
    }


def format_metric_value(value, metric_name):

    if pd.isna(value):
        return "N/A"

    if metric_is_percentage(metric_name):
        return f"{value:.2%}"

    if metric_name == "Business Value":
        return f"{value:,.2f}"

    return f"{value:,.4f}"


def make_preprocessor(
    numeric_features,
    categorical_features
):

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

    return ColumnTransformer(
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


def make_pipeline(
    depth,
    min_samples_leaf,
    min_samples_split,
    numeric_features,
    categorical_features
):

    preprocessor = make_preprocessor(
        numeric_features,
        categorical_features
    )

    tree = DecisionTreeClassifier(
        max_depth=int(depth),
        min_samples_leaf=int(
            min_samples_leaf
        ),
        min_samples_split=int(
            min_samples_split
        ),
        random_state=42
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                tree
            )
        ]
    )


def confusion_table(
    y_true,
    predictions
):

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    table = pd.DataFrame(
        cm,
        index=[
            "Actual 0",
            "Actual 1"
        ],
        columns=[
            "Predicted 0",
            "Predicted 1"
        ]
    )

    table["Total"] = table.sum(
        axis=1
    )

    total_row = pd.DataFrame(
        [
            [
                table[
                    "Predicted 0"
                ].sum(),
                table[
                    "Predicted 1"
                ].sum(),
                table[
                    "Total"
                ].sum()
            ]
        ],
        index=["Total"],
        columns=[
            "Predicted 0",
            "Predicted 1",
            "Total"
        ]
    )

    return pd.concat(
        [
            table,
            total_row
        ]
    )


def get_variable_importance(
    pipeline,
    numeric_features,
    categorical_features
):

    preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    tree = (
        pipeline.named_steps[
            "model"
        ]
    )

    importance = (
        tree.feature_importances_
    )

    importance_dict = {}

    position = 0

    # Numeric variables each create one transformed column
    for variable in numeric_features:

        importance_dict[
            variable
        ] = float(
            importance[position]
        )

        position += 1

    # Categorical variables can create several dummy columns
    if len(categorical_features) > 0:

        encoder = (
            preprocessor
            .named_transformers_[
                "categorical"
            ]
            .named_steps[
                "onehot"
            ]
        )

        for variable, categories in zip(
            categorical_features,
            encoder.categories_
        ):

            number_of_columns = len(
                categories
            )

            importance_dict[
                variable
            ] = float(
                importance[
                    position:
                    position
                    + number_of_columns
                ].sum()
            )

            position += (
                number_of_columns
            )

    result = pd.DataFrame(
        {
            "Variable":
                list(
                    importance_dict.keys()
                ),
            "Importance":
                list(
                    importance_dict.values()
                )
        }
    )

    return (
        result
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# INTERACTIVE DECISION TREE
# ============================================================

def create_tree_figure(
    pipeline,
    feature_names
):

    tree_model = (
        pipeline.named_steps[
            "model"
        ]
    )

    tree = tree_model.tree_

    children_left = (
        tree.children_left
    )

    children_right = (
        tree.children_right
    )

    node_depth = {}

    x_position = {}

    leaf_counter = [0]


    def assign_positions(
        node_id,
        depth=0
    ):

        node_depth[
            node_id
        ] = depth

        left = children_left[
            node_id
        ]

        right = children_right[
            node_id
        ]

        if left == right:

            x_position[
                node_id
            ] = (
                leaf_counter[0]
            )

            leaf_counter[0] += 1

        else:

            assign_positions(
                left,
                depth + 1
            )

            assign_positions(
                right,
                depth + 1
            )

            x_position[
                node_id
            ] = (
                x_position[left]
                + x_position[right]
            ) / 2


    assign_positions(0)


    edge_x = []
    edge_y = []


    for node_id in range(
        tree.node_count
    ):

        left = children_left[
            node_id
        ]

        right = children_right[
            node_id
        ]

        for child in [
            left,
            right
        ]:

            if child != -1:

                edge_x.extend(
                    [
                        x_position[
                            node_id
                        ],
                        x_position[
                            child
                        ],
                        None
                    ]
                )

                edge_y.extend(
                    [
                        -node_depth[
                            node_id
                        ],
                        -node_depth[
                            child
                        ],
                        None
                    ]
                )


    node_x = []
    node_y = []

    node_text = []
    hover_text = []


    for node_id in range(
        tree.node_count
    ):

        node_x.append(
            x_position[
                node_id
            ]
        )

        node_y.append(
            -node_depth[
                node_id
            ]
        )

        samples = int(
            tree.n_node_samples[
                node_id
            ]
        )

        values = (
            tree.value[
                node_id
            ][0]
        )

        total = values.sum()

        probability_1 = (
            values[1] / total
            if total > 0
            and len(values) > 1
            else 0
        )

        predicted_class = (
            1
            if probability_1 >= 0.50
            else 0
        )

        left = children_left[
            node_id
        ]

        right = children_right[
            node_id
        ]


        if left != right:

            feature_index = (
                tree.feature[
                    node_id
                ]
            )

            feature_name = (
                feature_names[
                    feature_index
                ]
            )

            threshold = (
                tree.threshold[
                    node_id
                ]
            )

            node_text.append(
                f"{feature_name}<br>"
                f"≤ {threshold:.2f}"
            )

            hover_text.append(
                f"<b>Split:</b> "
                f"{feature_name} "
                f"≤ {threshold:.2f}"
                f"<br><b>Samples:</b> "
                f"{samples:,}"
                f"<br><b>P(Class 1):</b> "
                f"{probability_1:.2%}"
                f"<br><b>Majority class:</b> "
                f"{predicted_class}"
            )

        else:

            node_text.append(
                f"Leaf<br>"
                f"P(1) = "
                f"{probability_1:.1%}"
            )

            hover_text.append(
                f"<b>Terminal Leaf</b>"
                f"<br><b>Samples:</b> "
                f"{samples:,}"
                f"<br><b>P(Class 1):</b> "
                f"{probability_1:.2%}"
                f"<br><b>Majority class:</b> "
                f"{predicted_class}"
            )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            hoverinfo="skip",
            line=dict(
                width=1.5
            ),
            showlegend=False
        )
    )


    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            hovertext=hover_text,
            hoverinfo="text",
            marker=dict(
                size=70,
                line=dict(
                    width=2
                )
            ),
            textfont=dict(
                size=10
            ),
            showlegend=False
        )
    )


    fig.update_layout(
        height=max(
            500,
            130 * (
                max(
                    node_depth.values()
                ) + 1
            )
        ),
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        xaxis=dict(
            visible=False
        ),
        yaxis=dict(
            visible=False
        ),
        hovermode="closest"
    )

    return fig


# ============================================================
# UPLOAD DATA
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=[
        "csv",
        "xlsx"
    ]
)


if uploaded_file is not None:

    if uploaded_file.name.lower().endswith(
        ".csv"
    ):

        df = pd.read_csv(
            uploaded_file
        )

    else:

        df = pd.read_excel(
            uploaded_file
        )


    st.subheader(
        "Data Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write(
        f"Rows: **{df.shape[0]:,}** | "
        f"Columns: **{df.shape[1]}**"
    )


    # ========================================================
    # 1. OUTCOME
    # ========================================================

    st.subheader(
        "1. Select Outcome Variable"
    )

    outcome_control, _ = st.columns(
        [1, 1]
    )

    with outcome_control:

        target = st.selectbox(
            "Which column are you predicting?",
            options=df.columns
        )


    # ========================================================
    # 2. PREDICTORS
    # ========================================================

    st.subheader(
        "2. Select Predictor Variables"
    )

    predictors = st.multiselect(
        "Select the columns to use as predictors:",
        options=[
            column
            for column in df.columns
            if column != target
        ]
    )


    # ========================================================
    # 3. TRAIN / TEST / CROSS-VALIDATION
    # ========================================================

    st.subheader(
        "3. Configure Training, Testing, and Cross-Validation"
    )

    split_control, _ = st.columns(
        [1, 1]
    )

    with split_control:

        test_percent = st.selectbox(
            "Percentage of observations used for testing:",
            options=[
                20,
                30,
                40
            ],
            index=1
        )


        cv_folds = st.selectbox(
            "Number of cross-validation folds:",
            options=[
                3,
                5,
                10
            ],
            index=1
        )

    st.caption(
        "Cross-validation is performed only within the training data. "
        "The testing data is reserved for the final model evaluation."
    )


    # ========================================================
    # 4. DEPTH RANGE
    # ========================================================

    st.subheader(
        "4. Select Tree Depth Range"
    )

    depth_control, _ = st.columns(
        [2, 1]
    )

    with depth_control:

        depth_col1, depth_col2 = (
            st.columns(2)
        )

        with depth_col1:

            min_depth = st.number_input(
                "Minimum tree depth",
                min_value=1,
                max_value=30,
                value=1,
                step=1
            )

        with depth_col2:

            max_depth = st.number_input(
                "Maximum tree depth",
                min_value=1,
                max_value=30,
                value=10,
                step=1
            )


    # ========================================================
    # 5. CONSTRAINTS
    # ========================================================

    st.subheader(
        "5. Tree Constraints"
    )

    constraint_control, _ = st.columns(
        [2, 1]
    )

    with constraint_control:

        constraint_col1, constraint_col2 = (
            st.columns(2)
        )

        with constraint_col1:

            min_samples_leaf = st.number_input(
                "Minimum observations in a terminal leaf",
                min_value=1,
                value=10,
                step=1
            )

        with constraint_col2:

            min_samples_split = st.number_input(
                "Minimum observations required to split a node",
                min_value=2,
                value=20,
                step=1
            )


    # ========================================================
    # 6. MODEL PERFORMANCE MEASURE
    # ========================================================

    st.subheader(
        "6. Select Model Performance Measure"
    )

    metric_options = [
        "Business Value",
        "Accuracy",
        "Balanced Accuracy",
        "Misclassification Error",
        "F1 Score",
        "ROC AUC",
        "PR AUC",
        "Log Loss",
        "False Positive Rate",
        "False Negative Rate",
        "Recall",
        "Precision"
    ]

    metric_labels = {
        "Business Value": (
            "Business Value: Total Financial Impact"
        ),
        "Accuracy": (
            "Accuracy: Percentage of all observations classified correctly"
        ),
        "Balanced Accuracy": (
            "Balanced Accuracy: Average accuracy across classes 0 and 1"
        ),
        "Misclassification Error": (
            "Misclassification Error: Percentage of all observations "
            "classified incorrectly"
        ),
        "F1 Score": (
            "F1 Score: Balance between precision and recall"
        ),
        "ROC AUC": (
            "ROC AUC: Ability to rank class 1 above class 0 across all cutoffs"
        ),
        "PR AUC": (
            "PR AUC: Precision-recall performance across all cutoffs"
        ),
        "Log Loss": (
            "Log Loss: Accuracy and confidence of predicted probabilities"
        ),
        "False Positive Rate": (
            "False Positive Rate: Percentage of actual 0s incorrectly "
            "classified as 1"
        ),
        "False Negative Rate": (
            "False Negative Rate: Percentage of actual 1s incorrectly "
            "classified as 0"
        ),
        "Recall": (
            "Recall: Percentage of actual 1s correctly classified as 1"
        ),
        "Precision": (
            "Precision: Percentage of predicted 1s that are actually 1"
        )
    }

    metric_control, _ = st.columns(
        [3, 2]
    )

    with metric_control:

        metric_name = st.selectbox(
            "Measure used to compare alternative trees:",
            options=metric_options,
            format_func=lambda metric: metric_labels[metric]
        )


    if metric_name == "Business Value":

        st.markdown(
            "#### Define the Cost-Benefit Matrix"
        )

        st.caption(
            "Enter the expected financial impact associated with each "
            "actual–predicted outcome."
        )

        matrix_control, _ = st.columns(
            [3, 2]
        )

        with matrix_control:

            value_header_1, value_header_2, value_header_3 = st.columns(
                [1.1, 1, 1]
            )

            with value_header_2:
                st.markdown(
                    "**Predicted 0 (Negative class)**"
                )

            with value_header_3:
                st.markdown(
                    "**Predicted 1 (Positive class)**"
                )

            actual_0_label, tn_column, fp_column = st.columns(
                [1.1, 1, 1]
            )

            with actual_0_label:
                st.markdown(
                    "**Actual 0 (Negative class)**"
                )

            with tn_column:
                tn_value = st.number_input(
                    "True Negative (TN)",
                    value=1.00,
                    step=0.50,
                    format="%.2f"
                )

            with fp_column:
                fp_value = st.number_input(
                    "False Positive (FP)",
                    value=0.00,
                    step=0.50,
                    format="%.2f"
                )

            actual_1_label, fn_column, tp_column = st.columns(
                [1.1, 1, 1]
            )

            with actual_1_label:
                st.markdown(
                    "**Actual 1 (Positive class)**"
                )

            with fn_column:
                fn_value = st.number_input(
                    "False Negative (FN)",
                    value=0.00,
                    step=0.50,
                    format="%.2f"
                )

            with tp_column:
                tp_value = st.number_input(
                    "True Positive (TP)",
                    value=1.00,
                    step=0.50,
                    format="%.2f"
                )

            st.markdown(
                """
                <div style="text-align: center; font-size: 1.1rem;
                            font-weight: 600; margin-top: 0.75rem;">
                    Select Business Objective:
                </div>
                """,
                unsafe_allow_html=True
            )

            objective_left, objective_center, objective_right = st.columns(
                [1, 4, 1]
            )

            with objective_center:

                optimization_direction = st.radio(
                    "Select Business Objective:",
                    options=[
                        "Maximize payoff or profit",
                        "Minimize cost or loss"
                    ],
                    horizontal=True,
                    label_visibility="collapsed"
                )

        business_values = {
            "TP": float(tp_value),
            "FP": float(fp_value),
            "FN": float(fn_value),
            "TN": float(tn_value)
        }

    else:

        business_values = {
            "TP": 0.0,
            "FP": 0.0,
            "FN": 0.0,
            "TN": 0.0
        }

        optimization_direction = ""


    # ========================================================
    # 7. CUTOFF BEFORE BUILDING
    # ========================================================

    st.subheader(
        "7. Classification Cutoff"
    )

    cutoff_control, _ = st.columns(
        [1, 1]
    )

    with cutoff_control:

        chosen_cutoff = st.number_input(
            "Enter classification cutoff:",
            min_value=0.00,
            max_value=1.00,
            value=0.50,
            step=0.01,
            format="%.2f"
        )

    st.caption(
        "An observation is classified as 1 "
        "when its predicted probability is "
        "greater than or equal to the cutoff."
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    if st.button(
        "Build Decision Tree",
        type="primary"
    ):

        if len(
            predictors
        ) == 0:

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


        model_df = (
            df.loc[
                df[target].notna()
            ]
            .copy()
        )

        X = model_df[
            predictors
        ].copy()

        y = model_df[
            target
        ].copy()


        if set(
            y.unique()
        ) != {
            0,
            1
        }:

            st.error(
                "The outcome variable must contain "
                "exactly two classes coded 0 and 1."
            )

            st.stop()


        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=(
                    test_percent
                    / 100
                ),
                stratify=y,
                random_state=42
            )
        )


        smallest_class = (
            y_train
            .value_counts()
            .min()
        )


        if smallest_class < cv_folds:

            st.error(
                "There are not enough observations "
                "in the smaller class for the "
                "selected number of CV folds."
            )

            st.stop()


        numeric_features = (
            X_train
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        categorical_features = [
            column
            for column
            in predictors
            if column
            not in numeric_features
        ]


        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42
        )


        depths = list(
            range(
                int(min_depth),
                int(max_depth) + 1
            )
        )

        cutoffs = np.round(
            np.arange(
                0.00,
                1.001,
                0.01
            ),
            2
        )


        # ----------------------------------------------------
        # CROSS-VALIDATED OUT-OF-FOLD PROBABILITIES
        #
        # Each tree depth is fitted once per fold.
        # We then evaluate every cutoff from 0.00 to 1.00
        # using the same out-of-fold probabilities.
        # ----------------------------------------------------

        oof_probabilities = {}


        for depth in depths:

            probabilities = pd.Series(
                index=X_train.index,
                dtype=float
            )


            for train_index, validation_index in cv.split(
                X_train,
                y_train
            ):

                X_cv_train = (
                    X_train.iloc[
                        train_index
                    ]
                )

                X_cv_validation = (
                    X_train.iloc[
                        validation_index
                    ]
                )

                y_cv_train = (
                    y_train.iloc[
                        train_index
                    ]
                )


                pipeline = make_pipeline(
                    depth,
                    min_samples_leaf,
                    min_samples_split,
                    numeric_features,
                    categorical_features
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


                probabilities.loc[
                    X_cv_validation.index
                ] = (
                    validation_probabilities
                )


            oof_probabilities[
                depth
            ] = probabilities


        # ----------------------------------------------------
        # DEPTH PERFORMANCE AT USER-SELECTED CUTOFF
        # ----------------------------------------------------

        depth_results = []


        for depth in depths:

            value = metric_value(
                y_train,
                oof_probabilities[
                    depth
                ].loc[
                    y_train.index
                ].values,
                chosen_cutoff,
                metric_name,
                business_values
            )


            depth_results.append(
                {
                    "Tree Depth":
                        depth,
                    metric_name:
                        value
                }
            )


        depth_results_df = pd.DataFrame(
            depth_results
        )


        metrics_to_minimize = {
            "Misclassification Error",
            "False Positive Rate",
            "False Negative Rate",
            "Log Loss"
        }


        if metric_name == "Business Value":

            minimize_metric = (
                optimization_direction
                == "Minimize cost or loss"
            )

        else:

            minimize_metric = (
                metric_name
                in metrics_to_minimize
            )


        if minimize_metric:

            best_depth_index = (
                depth_results_df[
                    metric_name
                ].idxmin()
            )

        else:

            best_depth_index = (
                depth_results_df[
                    metric_name
                ].idxmax()
            )


        selected_depth = int(
            depth_results_df.loc[
                best_depth_index,
                "Tree Depth"
            ]
        )

        selected_cv_score = float(
            depth_results_df.loc[
                best_depth_index,
                metric_name
            ]
        )


        # ----------------------------------------------------
        # FULL CUTOFF × DEPTH PERFORMANCE MATRIX
        # ----------------------------------------------------

        matrix = pd.DataFrame(
            index=cutoffs,
            columns=depths,
            dtype=float
        )


        for depth in depths:

            probabilities = (
                oof_probabilities[
                    depth
                ].loc[
                    y_train.index
                ].values
            )


            for cutoff in cutoffs:

                matrix.loc[
                    cutoff,
                    depth
                ] = metric_value(
                    y_train,
                    probabilities,
                    float(cutoff),
                    metric_name,
                    business_values
                )


        # ----------------------------------------------------
        # GLOBAL OPTIMAL DEPTH + CUTOFF
        # ----------------------------------------------------

        cutoff_independent_metrics = {
            "ROC AUC",
            "PR AUC",
            "Log Loss"
        }


        if metric_name in cutoff_independent_metrics:

            # These metrics do not depend on cutoff.
            # Optimize depth only and retain user's chosen cutoff.

            depth_scores = (
                matrix.iloc[0]
            )

            if minimize_metric:

                global_depth = int(
                    depth_scores.idxmin()
                )

            else:

                global_depth = int(
                    depth_scores.idxmax()
                )

            global_cutoff = float(
                chosen_cutoff
            )

            global_score = float(
                depth_scores.loc[
                    global_depth
                ]
            )

        else:

            matrix_values = (
                matrix.values.astype(
                    float
                )
            )


            if minimize_metric:

                flat_position = (
                    np.nanargmin(
                        matrix_values
                    )
                )

            else:

                flat_position = (
                    np.nanargmax(
                        matrix_values
                    )
                )


            row_index, column_index = (
                np.unravel_index(
                    flat_position,
                    matrix_values.shape
                )
            )

            global_cutoff = float(
                matrix.index[
                    row_index
                ]
            )

            global_depth = int(
                matrix.columns[
                    column_index
                ]
            )

            global_score = float(
                matrix.iloc[
                    row_index,
                    column_index
                ]
            )


        # ----------------------------------------------------
        # FIT MODEL SELECTED AT USER CUTOFF
        # ----------------------------------------------------

        selected_pipeline = (
            make_pipeline(
                selected_depth,
                min_samples_leaf,
                min_samples_split,
                numeric_features,
                categorical_features
            )
        )


        selected_pipeline.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        st.session_state.model_built = True

        st.session_state.use_global_optimum = False

        st.session_state.model_df = model_df

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test

        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        st.session_state.numeric_features = (
            numeric_features
        )

        st.session_state.categorical_features = (
            categorical_features
        )

        st.session_state.predictors = (
            predictors
        )

        st.session_state.min_samples_leaf_saved = (
            int(
                min_samples_leaf
            )
        )

        st.session_state.min_samples_split_saved = (
            int(
                min_samples_split
            )
        )

        st.session_state.metric_name_saved = (
            metric_name
        )

        st.session_state.business_values_saved = (
            business_values
        )

        st.session_state.optimization_direction_saved = (
            optimization_direction
        )

        st.session_state.chosen_cutoff_saved = (
            float(
                chosen_cutoff
            )
        )

        st.session_state.selected_depth = (
            selected_depth
        )

        st.session_state.selected_cv_score = (
            selected_cv_score
        )

        st.session_state.depth_results_df = (
            depth_results_df
        )

        st.session_state.cutoff_depth_matrix = (
            matrix
        )

        st.session_state.global_depth = (
            global_depth
        )

        st.session_state.global_cutoff = (
            global_cutoff
        )

        st.session_state.global_score = (
            global_score
        )

        st.session_state.selected_pipeline = (
            selected_pipeline
        )

        st.session_state.minimize_metric = (
            minimize_metric
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.model_built:

    metric_name = (
        st.session_state.metric_name_saved
    )

    business_values = (
        st.session_state.business_values_saved
    )

    optimization_direction = (
        st.session_state.optimization_direction_saved
    )

    chosen_cutoff_saved = (
        st.session_state.chosen_cutoff_saved
    )

    selected_depth = (
        st.session_state.selected_depth
    )

    selected_cv_score = (
        st.session_state.selected_cv_score
    )

    depth_results_df = (
        st.session_state.depth_results_df
    )

    matrix = (
        st.session_state.cutoff_depth_matrix
    )

    global_depth = (
        st.session_state.global_depth
    )

    global_cutoff = (
        st.session_state.global_cutoff
    )

    global_score = (
        st.session_state.global_score
    )

    minimize_metric = (
        st.session_state.minimize_metric
    )

    X_train = (
        st.session_state.X_train
    )

    X_test = (
        st.session_state.X_test
    )

    y_train = (
        st.session_state.y_train
    )

    y_test = (
        st.session_state.y_test
    )

    numeric_features = (
        st.session_state.numeric_features
    )

    categorical_features = (
        st.session_state.categorical_features
    )

    min_samples_leaf_saved = (
        st.session_state.min_samples_leaf_saved
    )

    min_samples_split_saved = (
        st.session_state.min_samples_split_saved
    )


    # ========================================================
    # SAMPLE SIZES
    # ========================================================

    st.divider()

    st.subheader(
        "Training and Testing Samples"
    )

    sample_col1, sample_col2 = (
        st.columns(2)
    )

    with sample_col1:

        st.metric(
            "Training Observations",
            f"{len(X_train):,}"
        )

    with sample_col2:

        st.metric(
            "Testing Observations",
            f"{len(X_test):,}"
        )


    # ========================================================
    # CV LINE GRAPH
    # ========================================================

    st.subheader(
        "Cross-Validation Results"
    )


    if metric_is_percentage(metric_name):

        graph_y = (
            depth_results_df[
                metric_name
            ] * 100
        )

        graph_suffix = "%"

        graph_hover_format = ".2f"

        graph_tick_format = ".2f"

    else:

        graph_y = depth_results_df[
            metric_name
        ]

        graph_suffix = ""

        graph_hover_format = (
            ",.2f"
            if metric_name == "Business Value"
            else ",.4f"
        )

        graph_tick_format = graph_hover_format


    cv_fig = go.Figure()


    cv_fig.add_trace(
        go.Scatter(
            x=depth_results_df[
                "Tree Depth"
            ],
            y=graph_y,
            mode="lines+markers",
            name=metric_name,
            hovertemplate=(
                "Tree depth: %{x}"
                "<br>"
                + metric_name
                + ": %{y:"
                + graph_hover_format
                + "}"
                + graph_suffix
                + "<extra></extra>"
            )
        )
    )


    cv_fig.add_vline(
        x=selected_depth,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=(
            f"Optimal depth = "
            f"{selected_depth}"
        ),
        annotation_position="top"
    )


    cv_fig.update_layout(
        xaxis_title="Tree Depth",
        yaxis=dict(
            title=(
                f"{metric_name}"
                + (
                    " (%)"
                    if metric_is_percentage(metric_name)
                    else ""
                )
            ),
            tickformat=graph_tick_format
        ),
        height=450,
        margin=dict(
            l=50,
            r=30,
            t=40,
            b=50
        )
    )


    st.plotly_chart(
        cv_fig,
        use_container_width=True
    )


    if minimize_metric:

        selection_description = (
            "lowest"
        )

    else:

        selection_description = (
            "highest"
        )


    st.success(
        f"""
At the selected cutoff of **{chosen_cutoff_saved:.2f}**,
tree depth **{selected_depth}** was selected because it produced
the **{selection_description} cross-validated {metric_name}**.

Cross-validated {metric_name}: **{format_metric_value(selected_cv_score, metric_name)}**
        """
    )


    # ========================================================
    # CURRENT MODEL
    # ========================================================

    if (
        st.session_state.use_global_optimum
    ):

        active_depth = (
            global_depth
        )

        active_cutoff = (
            global_cutoff
        )

        active_pipeline = (
            make_pipeline(
                active_depth,
                min_samples_leaf_saved,
                min_samples_split_saved,
                numeric_features,
                categorical_features
            )
        )

        active_pipeline.fit(
            X_train,
            y_train
        )

        active_label = (
            "Globally Optimal "
            "Depth / Cutoff Combination"
        )

    else:

        active_depth = (
            selected_depth
        )

        active_cutoff = (
            chosen_cutoff_saved
        )

        active_pipeline = (
            st.session_state.selected_pipeline
        )

        active_label = (
            "Model Selected at "
            "Your Classification Cutoff"
        )


    st.subheader(
        active_label
    )

    model_col1, model_col2 = (
        st.columns(2)
    )

    with model_col1:

        st.metric(
            "Tree Depth",
            active_depth
        )

    with model_col2:

        st.metric(
            "Classification Cutoff",
            f"{active_cutoff:.2f}"
        )


    # ========================================================
    # INTERACTIVE TREE
    # ========================================================

    st.subheader(
        "Decision Tree"
    )

    fitted_preprocessor = (
        active_pipeline
        .named_steps[
            "preprocessor"
        ]
    )

    transformed_feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    readable_feature_names = [
        name
        .replace(
            "numeric__",
            ""
        )
        .replace(
            "categorical__",
            ""
        )
        for name
        in transformed_feature_names
    ]


    tree_fig = create_tree_figure(
        active_pipeline,
        readable_feature_names
    )


    st.plotly_chart(
        tree_fig,
        use_container_width=True
    )

    st.caption(
        "Hover over a node for details. "
        "You can zoom and pan the tree."
    )


    # ========================================================
    # VARIABLE IMPORTANCE
    # ========================================================

    st.subheader(
        "Variable Importance"
    )

    st.write(
        "Each percentage represents that variable's share of the tree's "
        "total improvement in separating class 0 from class 1 across all "
        "splits. The percentages sum to 100%; a larger percentage means "
        "the tree relied more heavily on that variable. Importance does "
        "not show whether the effect is positive or negative and does not "
        "establish causation."
    )


    importance_df = (
        get_variable_importance(
            active_pipeline,
            numeric_features,
            categorical_features
        )
    )


    importance_fig = go.Figure()


    importance_fig.add_trace(
        go.Bar(
            x=importance_df[
                "Variable"
            ],
            y=(
                importance_df[
                    "Importance"
                ] * 100
            ),
            text=[
                f"{value:.2%}"
                for value
                in importance_df[
                    "Importance"
                ]
            ],
            textposition="outside",
            hovertemplate=(
                "%{x}<br>"
                "Importance: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )


    importance_fig.update_layout(
        yaxis_title="Importance (%)",
        xaxis_title="",
        height=450,
        margin=dict(
            l=50,
            r=20,
            t=40,
            b=70
        )
    )


    st.plotly_chart(
        importance_fig,
        use_container_width=True
    )


    # ========================================================
    # TRAINING / TEST PROBABILITIES
    # ========================================================

    train_probabilities = (
        active_pipeline
        .predict_proba(
            X_train
        )[:, 1]
    )

    test_probabilities = (
        active_pipeline
        .predict_proba(
            X_test
        )[:, 1]
    )


    train_metrics, train_predictions = (
        calculate_metrics(
            y_train,
            train_probabilities,
            active_cutoff
        )
    )

    test_metrics, test_predictions = (
        calculate_metrics(
            y_test,
            test_probabilities,
            active_cutoff
        )
    )


    # ========================================================
    # CONFUSION MATRICES WITH TOTALS
    # ========================================================

    st.subheader(
        "Confusion Matrices"
    )


    cm_left_margin, cm_center, cm_right_margin = st.columns(
        [1, 4, 1]
    )


    with cm_center:

        cm_col1, cm_col2 = st.columns(2)


        with cm_col1:

            st.markdown(
                "<div style='text-align: center;'><strong>Training Set"
                "</strong></div>",
                unsafe_allow_html=True
            )

            st.table(
                confusion_table(
                    y_train,
                    train_predictions
                )
                .style
                .set_properties(
                    **{"text-align": "center"}
                )
                .set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": [
                                ("text-align", "center")
                            ]
                        }
                    ]
                )
            )


        with cm_col2:

            st.markdown(
                "<div style='text-align: center;'><strong>Testing Set"
                "</strong></div>",
                unsafe_allow_html=True
            )

            st.table(
                confusion_table(
                    y_test,
                    test_predictions
                )
                .style
                .set_properties(
                    **{"text-align": "center"}
                )
                .set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": [
                                ("text-align", "center")
                            ]
                        }
                    ]
                )
            )


    if metric_name == "Business Value":

        train_business_value = metric_value(
            y_train,
            train_probabilities,
            active_cutoff,
            "Business Value",
            business_values
        )

        test_business_value = metric_value(
            y_test,
            test_probabilities,
            active_cutoff,
            "Business Value",
            business_values
        )

        st.subheader(
            "Business Performance"
        )

        business_col1, business_col2 = st.columns(2)

        with business_col1:

            st.metric(
                "Training Total Business Value",
                f"{train_business_value:,.2f}"
            )

            st.caption(
                "Average business value per prediction: "
                f"{train_business_value / len(y_train):,.2f}"
            )

        with business_col2:

            st.metric(
                "Testing Total Business Value",
                f"{test_business_value:,.2f}"
            )

            st.caption(
                "Average business value per prediction: "
                f"{test_business_value / len(y_test):,.2f}"
            )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader(
        "Model Performance"
    )


    practical_descriptions = {
        "Accuracy": (
            "Overall percentage classified correctly; most useful when "
            "the classes and error consequences are reasonably balanced."
        ),
        "Balanced Accuracy": (
            "Measures how well the model identifies both classes while "
            "giving class 0 and class 1 equal importance."
        ),
        "Misclassification Error": (
            "Overall percentage classified incorrectly. Lower values are "
            "better."
        ),
        "F1 Score": (
            "Summarizes the balance between finding actual class 1 cases "
            "and avoiding too many incorrect class 1 predictions."
        ),
        "ROC AUC": (
            "Shows how well the model ranks class 1 above class 0 across "
            "all possible cutoffs."
        ),
        "PR AUC": (
            "Summarizes precision and recall across cutoffs; especially "
            "useful when class 1 is uncommon."
        ),
        "Log Loss": (
            "Evaluates the quality and confidence of predicted "
            "probabilities. Lower values are better."
        ),
        "False Positive Rate": (
            "Among actual class 0 cases, the percentage incorrectly flagged "
            "as class 1."
        ),
        "False Negative Rate": (
            "Among actual class 1 cases, the percentage the model misses."
        ),
        "Recall": (
            "Among actual class 1 cases, the percentage the model "
            "successfully identifies."
        ),
        "Precision": (
            "Among cases predicted as class 1, the percentage that truly "
            "belong to class 1."
        )
    }


    performance_rows = []


    for metric in train_metrics:

        performance_rows.append(
            {
                "Metric":
                    metric,
                "Training":
                    format_metric_value(
                        train_metrics[metric],
                        metric
                    ),
                "Testing":
                    format_metric_value(
                        test_metrics[metric],
                        metric
                    ),
                "Practical Interpretation":
                    practical_descriptions[metric]
            }
        )


    performance_df = pd.DataFrame(
        performance_rows
    )


    st.dataframe(
        performance_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Metric":
                st.column_config.TextColumn(
                    width="medium"
                ),
            "Training":
                st.column_config.TextColumn(
                    width="small"
                ),
            "Testing":
                st.column_config.TextColumn(
                    width="small"
                ),
            "Practical Interpretation":
                st.column_config.TextColumn(
                    width="large"
                )
        }
    )


    # ========================================================
    # CUTOFF × DEPTH ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "Cutoff × Tree Depth Analysis"
    )


    st.write(
        f"""
The table and 3D graph below evaluate **{metric_name}**
using cross-validation on the training data for every
classification cutoff from **0.00 to 1.00** and every
tree depth you allowed.
        """
    )


    cutoff_independent_metrics = {
        "ROC AUC",
        "PR AUC",
        "Log Loss"
    }


    if metric_name in cutoff_independent_metrics:

        st.info(
            f"{metric_name} does not depend on a classification cutoff. "
            f"Therefore, {metric_name} is identical across all cutoffs "
            "for a given tree depth. The optimal depth is selected using "
            f"{metric_name}, while your entered cutoff is retained for "
            "classification."
        )


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    if metric_is_percentage(metric_name):

        display_matrix = matrix.copy() * 100

    else:

        display_matrix = matrix.copy()


    display_matrix.index = [
        f"{cutoff:.2f}"
        for cutoff
        in display_matrix.index
    ]


    display_matrix.columns = [
        f"Depth {depth}"
        for depth
        in display_matrix.columns
    ]


    if metric_is_percentage(metric_name):

        formatted_matrix = display_matrix.map(
            lambda value: f"{value:.2f}%"
        )

    elif metric_name == "Business Value":

        formatted_matrix = display_matrix.map(
            lambda value: f"{value:,.2f}"
        )

    else:

        formatted_matrix = display_matrix.map(
            lambda value: f"{value:,.4f}"
        )


    formatted_matrix.index.name = (
        "Cutoff"
    )


    st.subheader(
        f"{metric_name} by Cutoff and Tree Depth"
    )


    st.dataframe(
        formatted_matrix,
        height=500,
        use_container_width=True
    )


    # ========================================================
    # 3D SURFACE
    # ========================================================

    st.subheader(
        "Interactive 3D Performance Surface"
    )


    surface_z = matrix.values.astype(float)

    if metric_is_percentage(metric_name):
        surface_z = surface_z * 100


    if metric_is_percentage(metric_name):

        surface_hover_value = "%{z:.2f}%"

        surface_z_title = f"{metric_name} (%)"

        optimal_surface_z = global_score * 100

        surface_tick_format = ".2f"

    elif metric_name == "Business Value":

        surface_hover_value = "%{z:,.2f}"

        surface_z_title = metric_name

        optimal_surface_z = global_score

        surface_tick_format = ",.2f"

    else:

        surface_hover_value = "%{z:,.4f}"

        surface_z_title = metric_name

        optimal_surface_z = global_score

        surface_tick_format = ",.4f"


    surface_fig = go.Figure()


    surface_fig.add_trace(
        go.Surface(
            x=np.array(
                matrix.columns,
                dtype=float
            ),
            y=np.array(
                matrix.index,
                dtype=float
            ),
            z=surface_z,
            hovertemplate=(
                "Tree depth: %{x:.0f}"
                "<br>Cutoff: %{y:.2f}"
                "<br>"
                + metric_name
                + ": "
                + surface_hover_value
                + "<extra></extra>"
            ),
            colorbar=dict(
                tickformat=surface_tick_format
            ),
            showscale=True
        )
    )


    surface_fig.add_trace(
        go.Scatter3d(
            x=[
                global_depth
            ],
            y=[
                global_cutoff
            ],
            z=[
                optimal_surface_z
            ],
            mode="markers+text",
            marker=dict(
                size=9,
                color="red"
            ),
            text=[
                f"<b>Optimal</b>"
                f"<br>Depth: {global_depth}"
                f"<br>Cutoff: {global_cutoff:.2f}"
                f"<br>{metric_name}: "
                f"{format_metric_value(global_score, metric_name)}"
            ],
            textposition="top center",
            textfont=dict(
                size=15,
                color="darkred"
            ),
            hovertemplate=(
                f"Optimal depth: "
                f"{global_depth}"
                f"<br>Optimal cutoff: "
                f"{global_cutoff:.2f}"
                f"<br>{metric_name}: "
                f"{format_metric_value(global_score, metric_name)}"
                "<extra></extra>"
            )
        )
    )


    surface_fig.update_layout(
        height=650,
        scene=dict(
            xaxis_title=(
                "Tree Depth"
            ),
            yaxis_title=(
                "Classification Cutoff"
            ),
            zaxis=dict(
                title=surface_z_title,
                tickformat=surface_tick_format
            )
        ),
        margin=dict(
            l=0,
            r=0,
            t=20,
            b=0
        ),
        showlegend=False
    )


    st.plotly_chart(
        surface_fig,
        use_container_width=True
    )


    # ========================================================
    # OPTIMAL COMBINATION
    # ========================================================

    if metric_name in cutoff_independent_metrics:

        st.success(
            f"""
**Optimal tree depth:** {global_depth}

**{metric_name}:** {format_metric_value(global_score, metric_name)}

{metric_name} does not identify an optimal classification cutoff,
so the classification cutoff remains **{global_cutoff:.2f}**.
            """
        )

    else:

        optimal_word = (
            "minimum"
            if minimize_metric
            else "maximum"
        )

        st.success(
            f"""
**Globally optimal combination based on cross-validation**

Tree depth: **{global_depth}**

Classification cutoff: **{global_cutoff:.2f}**

{metric_name}: **{format_metric_value(global_score, metric_name)}**

This combination produced the **{optimal_word}**
cross-validated {metric_name} among all depth/cutoff
combinations evaluated.
            """
        )


    # ========================================================
    # BUTTON TO USE GLOBAL OPTIMUM
    # ========================================================

    if not (
        st.session_state.use_global_optimum
    ):

        if st.button(
            "Use Optimal Depth and Cutoff for Predictions",
            type="primary"
        ):

            st.session_state.use_global_optimum = True

            st.rerun()

    else:

        if st.button(
            "Return to My Selected Cutoff Model"
        ):

            st.session_state.use_global_optimum = False

            st.rerun()


    # ========================================================
    # DOWNLOAD PREDICTIONS
    # ========================================================

    st.divider()

    st.subheader(
        "Download Predictions"
    )


    model_df = (
        st.session_state.model_df
    )


    train_output = (
        model_df.loc[
            X_train.index
        ]
        .copy()
    )

    train_output[
        "Data_Set"
    ] = "Training"

    train_output[
        "Predicted_Probability"
    ] = train_probabilities

    train_output[
        "Predicted_Class"
    ] = train_predictions


    test_output = (
        model_df.loc[
            X_test.index
        ]
        .copy()
    )

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
The downloaded file uses:

**Tree depth:** {active_depth}

**Classification cutoff:** {active_cutoff:.2f}
        """
    )


    st.dataframe(
        output_df.head(20),
        use_container_width=True
    )


    csv = (
        output_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(
        "Download Predictions",
        data=csv,
        file_name=(
            "decision_tree_predictions.csv"
        ),
        mime="text/csv"
    )
