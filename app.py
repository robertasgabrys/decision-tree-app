import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from html import escape

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
    confusion_matrix,
)


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Decision Tree Model Builder",
    layout="wide"
)

st.markdown(
    """
    <style>
    div[data-testid="stWidgetLabel"] p {
        font-size: 1rem;
        color: #262730;
    }
    div[role="radiogroup"] label p {
        font-size: 1rem;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 1rem;
        color: #262730;
        font-weight: 600;
    }
    .important-instruction {
        color: #262730;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    .business-objective {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0.25rem 0 0.4rem 0;
    }
    .cm-wrapper {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        gap: 96px;
        flex-wrap: wrap;
        margin: 0.5rem auto 1.5rem auto;
    }
    .cm-block h4 {
        text-align: center;
        font-size: 1.15rem;
        margin: 0 0 0.5rem 0;
    }
    .cm-table {
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 1.05rem;
    }
    .cm-table th, .cm-table td {
        border: 1px solid #d9d9d9;
        min-width: 92px;
        height: 46px;
        padding: 8px 12px;
        text-align: center !important;
        vertical-align: middle !important;
    }
    .cm-table th { font-weight: 700; }
    .cm-table tr:last-child th,
    .cm-table tr:last-child td,
    .cm-table th:last-child,
    .cm-table td:last-child { font-weight: 700; }
    .performance-table, .business-table {
        border-collapse: collapse;
        table-layout: fixed;
        margin: 0 auto 1rem auto;
        color: #262730;
    }
    .performance-table { width: 100%; }
    .business-table { width: 50%; }
    .performance-table th, .performance-table td,
    .business-table th, .business-table td {
        border: 1px solid #d9d9d9;
        padding: 9px 10px;
        vertical-align: top;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .performance-table th, .business-table th {
        background: #f4f6f8;
        font-weight: 700;
        text-align: center;
    }
    .performance-table th:nth-child(1),
    .performance-table td:nth-child(1) { width: 17%; }
    .performance-table th:nth-child(2),
    .performance-table td:nth-child(2),
    .performance-table th:nth-child(3),
    .performance-table td:nth-child(3) {
        width: 10%;
        text-align: center;
    }
    .performance-table th:nth-child(4),
    .performance-table td:nth-child(4) { width: 63%; }
    .business-table td { text-align: center; vertical-align: middle; }
    </style>
    """,
    unsafe_allow_html=True
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

    metrics = {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Misclassification Error": misclassification,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
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
        "Business Value"
    }


def business_value_per_prediction(
    y_true,
    predictions,
    business_values
):

    y_array = np.asarray(y_true)
    prediction_array = np.asarray(predictions)

    return np.select(
        [
            (y_array == 1) & (prediction_array == 1),
            (y_array == 0) & (prediction_array == 1),
            (y_array == 1) & (prediction_array == 0),
            (y_array == 0) & (prediction_array == 0)
        ],
        [
            business_values["TP"],
            business_values["FP"],
            business_values["FN"],
            business_values["TN"]
        ],
        default=np.nan
    )


def format_metric_value(value, metric_name):

    if pd.isna(value):
        return "N/A"

    if metric_is_percentage(metric_name):
        return f"{value:.2%}"

    if metric_name == "Business Value":
        return f"{value:,.2f}"

    return f"{value:,.2f}"


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
    categorical_features,
    random_seed
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
        random_state=int(random_seed)
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


def confusion_matrix_html(title, y_true, predictions):

    table = confusion_table(
        y_true,
        predictions
    )

    rows = []

    for row_label in [
        "Actual 0",
        "Actual 1",
        "Total"
    ]:

        cells = "".join(
            f"<td>{int(table.loc[row_label, column])}</td>"
            for column in [
                "Predicted 0",
                "Predicted 1",
                "Total"
            ]
        )

        rows.append(
            f"<tr><th>{escape(row_label)}</th>{cells}</tr>"
        )

    return (
        "<div class='cm-block'>"
        f"<h4>{escape(title)}</h4>"
        "<table class='cm-table'>"
        "<thead><tr><th></th><th>Predicted 0</th>"
        "<th>Predicted 1</th><th>Total</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
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
    feature_names,
    classification_cutoff
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
            if probability_1 >= classification_cutoff
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
                f"<br><b>Predicted class at cutoff "
                f"{classification_cutoff:.2f}:</b> "
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
                f"<br><b>Predicted class at cutoff "
                f"{classification_cutoff:.2f}:</b> "
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
                size=82,
                line=dict(
                    width=2
                )
            ),
            textfont=dict(
                size=12
            ),
            showlegend=False
        )
    )


    fig.update_layout(
        height=max(
            500,
            150 * (
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
        hoverlabel=dict(
            font_size=14
        ),
        hovermode="closest"
    )

    return fig


# ============================================================
# UPLOAD DATA
# ============================================================

upload_control, _ = st.columns(
    [2, 3]
)

with upload_control:

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
        [1, 4]
    )

    with outcome_control:

        target = st.selectbox(
            "Which column are you predicting?",
            options=df.columns,
            index=None,
            placeholder="Select outcome"
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
        [1, 4]
    )

    with split_control:

        test_percent_input = st.text_input(
            "Percentage of observations used for testing:",
            value="",
            placeholder="e.g., 30"
        )


        cv_folds_input = st.text_input(
            "Number of cross-validation folds:",
            value="",
            placeholder="e.g., 5"
        )


        random_seed_input = st.text_input(
            "Random seed for reproducibility:",
            value="",
            placeholder="e.g., 42"
        )

    st.write(
        "Cross-validation is performed only within the training data. "
        "The testing data is reserved for the final model evaluation. "
        "The random seed fixes the training/testing split and "
        "cross-validation folds so the same data and settings reproduce "
        "the same results."
    )


    # ========================================================
    # 4. DEPTH RANGE
    # ========================================================

    st.subheader(
        "4. Select Tree Depth Range"
    )

    depth_control, _ = st.columns(
        [1, 4]
    )

    with depth_control:

        depth_col1, depth_col2 = (
            st.columns(2)
        )

        with depth_col1:

            min_depth_input = st.text_input(
                "Minimum tree depth",
                value="",
                placeholder="e.g., 1"
            )

        with depth_col2:

            max_depth_input = st.text_input(
                "Maximum tree depth",
                value="",
                placeholder="e.g., 10"
            )


    # ========================================================
    # 5. CONSTRAINTS
    # ========================================================

    st.subheader(
        "5. Tree Constraints"
    )

    constraint_control, _ = st.columns(
        [1, 4]
    )

    with constraint_control:

        constraint_col1, constraint_col2 = (
            st.columns(2)
        )

        with constraint_col1:

            min_samples_leaf_input = st.text_input(
                "Minimum observations in a terminal leaf",
                value="",
                placeholder="e.g., 10"
            )

        with constraint_col2:

            min_samples_split_input = st.text_input(
                "Minimum observations required to split a node",
                value="",
                placeholder="e.g., 20"
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
        "False Positive Rate",
        "False Negative Rate",
        "Recall",
        "Precision"
    ]

    metric_labels = {
        "Business Value": (
            "Business Value: Total Net Financial Value or Cost"
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
        [3, 7]
    )

    with metric_control:

        metric_name = st.selectbox(
            "Measure used to compare alternative trees:",
            options=metric_options,
            format_func=lambda metric: metric_labels[metric]
        )


    if metric_name == "Business Value":

        objective_control, _ = st.columns(
            [3, 2]
        )

        with objective_control:

            objective_label, objective_choices = st.columns(
                [1.1, 3]
            )

            with objective_label:
                st.markdown(
                    "<div class='business-objective'>Select Business "
                    "Objective:</div>",
                    unsafe_allow_html=True
                )

            with objective_choices:
                optimization_direction = st.radio(
                    "Select Business Objective:",
                    options=[
                        "Maximize payoff or profit",
                        "Minimize cost or loss"
                    ],
                    horizontal=True,
                    label_visibility="collapsed"
                )

        st.markdown(
            "### Baseline/Benchmark Financial Value (Without the Model)"
        )

        st.markdown(
            "<div class='important-instruction'>Enter the financial value "
            "for each actual outcome under the current or no-model policy. "
            "These values provide the benchmark used to calculate Business "
            "Value Added or cost savings.</div>",
            unsafe_allow_html=True
        )

        baseline_control, _ = st.columns(
            [2, 5]
        )

        with baseline_control:

            baseline_0_label, baseline_0_value = st.columns(
                [1, 1.4]
            )

            with baseline_0_label:
                st.markdown(
                    "<div style='padding-top:0.65rem; font-weight:700;'>"
                    "Actual 0</div>",
                    unsafe_allow_html=True
                )

            with baseline_0_value:
                baseline_actual_0_input = st.text_input(
                    "Baseline value for Actual 0",
                    value="",
                    label_visibility="collapsed",
                    placeholder="Actual 0 value"
                )
            baseline_1_label, baseline_1_value = st.columns(
                [1, 1.4]
            )

            with baseline_1_label:
                st.markdown(
                    "<div style='padding-top:0.65rem; font-weight:700;'>"
                    "Actual 1</div>",
                    unsafe_allow_html=True
                )

            with baseline_1_value:
                baseline_actual_1_input = st.text_input(
                    "Baseline value for Actual 1",
                    value="",
                    label_visibility="collapsed",
                    placeholder="Actual 1 value"
                )

        st.markdown(
            "### Define the Cost-Benefit Matrix"
        )

        st.markdown(
            "<div class='important-instruction'>Enter the expected financial "
            "impact associated with each actual–predicted outcome. When "
            "maximizing, enter gains as positive values and losses as "
            "negative values. When minimizing, enter costs or losses as "
            "positive values.</div>",
            unsafe_allow_html=True
        )

        matrix_control, _ = st.columns(
            [2, 3]
        )

        with matrix_control:

            value_header_1, value_header_2, value_header_3 = st.columns(
                [0.8, 1, 1]
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
                [0.8, 1, 1]
            )

            with actual_0_label:
                st.markdown(
                    "<div style='padding-top: 2rem; font-size: 1rem; "
                    "font-weight: 700;'>Actual 0 (Negative class)</div>",
                    unsafe_allow_html=True
                )

            with tn_column:
                tn_value_input = st.text_input(
                    "True Negative (TN)",
                    value="",
                    placeholder="Enter value"
                )

            with fp_column:
                fp_value_input = st.text_input(
                    "False Positive (FP)",
                    value="",
                    placeholder="Enter value"
                )

            actual_1_label, fn_column, tp_column = st.columns(
                [0.8, 1, 1]
            )

            with actual_1_label:
                st.markdown(
                    "<div style='padding-top: 2rem; font-size: 1rem; "
                    "font-weight: 700;'>Actual 1 (Positive class)</div>",
                    unsafe_allow_html=True
                )

            with fn_column:
                fn_value_input = st.text_input(
                    "False Negative (FN)",
                    value="",
                    placeholder="Enter value"
                )

            with tp_column:
                tp_value_input = st.text_input(
                    "True Positive (TP)",
                    value="",
                    placeholder="Enter value"
                )

    else:

        optimization_direction = ""


    # ========================================================
    # 7. CUTOFF BEFORE BUILDING
    # ========================================================

    st.subheader(
        "7. Classification Cutoff"
    )

    cutoff_control, _ = st.columns(
        [1, 11]
    )

    with cutoff_control:

        chosen_cutoff_input = st.text_input(
            "Enter classification cutoff:",
            value="",
            placeholder="0.XX"
        )

    st.write(
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

        validation_errors = []


        def required_integer(raw_value, section, field_name, minimum, maximum):

            if not str(raw_value).strip():
                validation_errors.append(
                    f"Section {section}: enter {field_name}."
                )
                return None

            try:
                numeric_value = float(str(raw_value).strip())
            except ValueError:
                validation_errors.append(
                    f"Section {section}: {field_name} must be a number."
                )
                return None

            if not numeric_value.is_integer():
                validation_errors.append(
                    f"Section {section}: {field_name} must be a whole number."
                )
                return None

            integer_value = int(numeric_value)

            if integer_value < minimum or integer_value > maximum:
                validation_errors.append(
                    f"Section {section}: {field_name} must be between "
                    f"{minimum} and {maximum}."
                )
                return None

            return integer_value


        def required_number(raw_value, section, field_name):

            if not str(raw_value).strip():
                validation_errors.append(
                    f"Section {section}: enter {field_name}."
                )
                return None

            try:
                return float(str(raw_value).strip().replace(",", ""))
            except ValueError:
                validation_errors.append(
                    f"Section {section}: {field_name} must be numeric."
                )
                return None


        if target is None:
            validation_errors.append(
                "Section 1: select an outcome variable."
            )

        if len(predictors) == 0:
            validation_errors.append(
                "Section 2: select at least one predictor variable."
            )

        test_percent = required_integer(
            test_percent_input,
            3,
            "the testing percentage",
            1,
            50
        )

        cv_folds = required_integer(
            cv_folds_input,
            3,
            "the number of cross-validation folds",
            2,
            20
        )

        random_seed = required_integer(
            random_seed_input,
            3,
            "the random seed",
            0,
            999999
        )

        min_depth = required_integer(
            min_depth_input,
            4,
            "the minimum tree depth",
            1,
            30
        )

        max_depth = required_integer(
            max_depth_input,
            4,
            "the maximum tree depth",
            1,
            30
        )

        min_samples_leaf = required_integer(
            min_samples_leaf_input,
            5,
            "the minimum observations in a terminal leaf",
            1,
            999999
        )

        min_samples_split = required_integer(
            min_samples_split_input,
            5,
            "the minimum observations required to split a node",
            2,
            999999
        )

        chosen_cutoff = required_number(
            chosen_cutoff_input,
            7,
            "the classification cutoff"
        )

        if chosen_cutoff is not None and not 0 <= chosen_cutoff <= 1:
            validation_errors.append(
                "Section 7: the classification cutoff must be between 0 and 1."
            )

        if (
            min_depth is not None
            and max_depth is not None
            and min_depth > max_depth
        ):
            validation_errors.append(
                "Section 4: minimum tree depth cannot be larger than "
                "maximum tree depth."
            )

        if metric_name == "Business Value":

            baseline_actual_0 = required_number(
                baseline_actual_0_input,
                6,
                "the baseline value for Actual 0"
            )
            baseline_actual_1 = required_number(
                baseline_actual_1_input,
                6,
                "the baseline value for Actual 1"
            )
            tn_value = required_number(
                tn_value_input,
                6,
                "the True Negative value"
            )
            fp_value = required_number(
                fp_value_input,
                6,
                "the False Positive value"
            )
            fn_value = required_number(
                fn_value_input,
                6,
                "the False Negative value"
            )
            tp_value = required_number(
                tp_value_input,
                6,
                "the True Positive value"
            )

            if not validation_errors:
                baseline_values = {
                    0: baseline_actual_0,
                    1: baseline_actual_1
                }
                business_values = {
                    "TP": tp_value,
                    "FP": fp_value,
                    "FN": fn_value,
                    "TN": tn_value
                }

        else:
            baseline_values = None
            business_values = {
                "TP": 0.0,
                "FP": 0.0,
                "FN": 0.0,
                "TN": 0.0
            }

        if validation_errors:
            st.error(
                "The model was not built. Please complete or correct the "
                "following inputs:\n\n- "
                + "\n- ".join(validation_errors)
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
                random_state=int(random_seed)
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
            random_state=int(random_seed)
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
                    categorical_features,
                    random_seed
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
            "False Negative Rate"
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
            "PR AUC"
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
                categorical_features,
                random_seed
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

        st.session_state.baseline_values_saved = (
            baseline_values
        )

        st.session_state.cv_folds_saved = int(cv_folds)

        st.session_state.random_seed_saved = int(random_seed)

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

    baseline_values = (
        st.session_state.baseline_values_saved
    )

    cv_folds_saved = (
        st.session_state.cv_folds_saved
    )

    random_seed_saved = (
        st.session_state.random_seed_saved
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


    output_section_number = [1]


    def output_section(title):

        st.header(
            f"{output_section_number[0]}. {title}"
        )

        output_section_number[0] += 1


    # ========================================================
    # SAMPLE SIZES
    # ========================================================

    st.divider()

    st.markdown(
        "<h1 style='text-align: center;'>Decision Tree Model Results</h1>",
        unsafe_allow_html=True
    )

    output_section(
        "Data Summary"
    )

    sample_col1, sample_col2, sample_col3, sample_col4 = (
        st.columns(4)
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

    with sample_col3:

        st.metric(
            "Cross-Validation Folds",
            cv_folds_saved
        )

    with sample_col4:

        st.metric(
            "Random Seed",
            random_seed_saved
        )


    # ========================================================
    # CV LINE GRAPH
    # ========================================================

    output_section(
        "Cross-Validation and Model Selection "
        f"({len(X_train):,} Training Observations, "
        f"{cv_folds_saved} Folds)"
    )

    average_validation_size = (
        len(X_train) / cv_folds_saved
    )

    average_fold_training_size = (
        len(X_train) - average_validation_size
    )

    st.write(
        f"Cross-validation used only the **{len(X_train):,}-observation "
        f"training set**. In each of the {cv_folds_saved} rounds, "
        f"approximately **{average_fold_training_size:,.0f} observations** "
        "fitted the tree and **"
        f"{average_validation_size:,.0f} observations** evaluated it. "
        "Each training observation was evaluated once while excluded from "
        "model fitting. The testing set was not used to select the tree "
        "depth or cutoff."
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
            marker=dict(
                size=8
            ),
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
        annotation_position="top",
        annotation_font_size=15
    )


    cv_fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Tree Depth",
                font=dict(size=17)
            ),
            tickfont=dict(size=14),
            dtick=1
        ),
        yaxis=dict(
            title=dict(
                text=(
                    f"{metric_name}"
                    + (
                        " (%)"
                        if metric_is_percentage(metric_name)
                        else ""
                    )
                ),
                font=dict(size=17)
            ),
            tickformat=graph_tick_format,
            tickfont=dict(size=14)
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


    if metric_name == "Business Value":

        if optimization_direction == "Minimize cost or loss":

            cv_result_label = (
                "Pooled Out-of-Fold Total Cost/Loss (Training Data)"
            )

            cv_average_label = (
                "Average Cost/Loss per Cross-Validated Prediction"
            )

        else:

            cv_result_label = (
                "Pooled Out-of-Fold Total Business Value (Training Data)"
            )

            cv_average_label = (
                "Average Business Value per Cross-Validated Prediction"
            )

        cv_average_line = (
            f"\n\n{cv_average_label}: "
            f"**{selected_cv_score / len(y_train):,.2f}**"
        )

    else:

        cv_result_label = (
            f"Out-of-Fold Cross-Validated {metric_name}"
        )

        cv_average_line = ""


    st.success(
        f"""
At the selected cutoff of **{chosen_cutoff_saved:.2f}**,
tree depth **{selected_depth}** was selected because it produced
the **{selection_description} pooled out-of-fold {metric_name} on the training data**.

{cv_result_label}: **{format_metric_value(selected_cv_score, metric_name)}**
{cv_average_line}
        """
    )

    st.write(
        f"This total is the **sum across all {len(y_train):,} pooled "
        "out-of-fold validation predictions**. In "
        f"{cv_folds_saved}-fold cross-validation, the validation partitions "
        "do not overlap and together cover the entire training set, so each "
        "training observation contributes once while excluded from model "
        "fitting. It is **not** the final testing-set result and it is "
        "**not** the average of the fold totals. Final testing performance "
        "appears later under Business Performance and Model Performance "
        "Metrics."
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
                categorical_features,
                random_seed_saved
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


    output_section(
        "Selected Model Configuration"
    )

    st.write(
        f"Currently displaying: **{active_label}**."
    )

    model_col1, model_col2 = (
        st.columns(2)
    )

    with model_col1:

        st.markdown(
            f"""
            <div style="text-align:center; border:1px solid #d9d9d9;
                        border-radius:8px; padding:14px;">
                <div style="font-size:1.15rem; font-weight:700;">
                    Tree Depth
                </div>
                <div style="font-size:2rem; font-weight:700;">
                    {active_depth}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with model_col2:

        st.markdown(
            f"""
            <div style="text-align:center; border:1px solid #d9d9d9;
                        border-radius:8px; padding:14px;">
                <div style="font-size:1.15rem; font-weight:700;">
                    Classification Cutoff
                </div>
                <div style="font-size:2rem; font-weight:700;">
                    {active_cutoff:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # INTERACTIVE TREE
    # ========================================================

    output_section(
        "Decision Tree"
    )

    st.write(
        "Start at the top of the tree and follow the splits downward. "
        "If the displayed condition is true, move left; if it is false, "
        "move right. The final terminal node shows the model's predicted "
        "probability of class 1. The classification cutoff converts that "
        "probability into a predicted class. Hover over any node for "
        "additional details; zoom and pan to inspect the tree."
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
        readable_feature_names,
        active_cutoff
    )


    st.plotly_chart(
        tree_fig,
        use_container_width=True
    )

    # ========================================================
    # VARIABLE IMPORTANCE
    # ========================================================

    output_section(
        "Variable Importance"
    )

    st.write(
        "Variable Importance is based on the tree's reductions in Gini "
        "impurity when it creates splits; it is not based on the selected "
        "performance measure or the Cost-Benefit Matrix. The percentages "
        "sum to 100%. For example, 20% means the variable accounted for "
        "about 20% of the fitted tree's total improvement in separating "
        "class 0 from class 1. It does not mean that profit, Business Value, "
        "or accuracy increased by 20%. A value of 0% means the final tree "
        "did not use that variable in a split. This does not prove that the "
        "variable has no predictive value, because another correlated "
        "variable may have been selected instead. Importance also does not "
        "show whether an effect is positive or negative and does not "
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

    output_section(
        "Confusion Matrices"
    )


    st.markdown(
        "<div class='cm-wrapper'>"
        + confusion_matrix_html(
            "Training Set",
            y_train,
            train_predictions
        )
        + confusion_matrix_html(
            "Testing Set",
            y_test,
            test_predictions
        )
        + "</div>",
        unsafe_allow_html=True
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

        output_section(
            "Business Performance"
        )

        if baseline_values is not None:

            minimizing_business_cost = (
                optimization_direction == "Minimize cost or loss"
            )

            train_baseline_value = (
                int((y_train == 0).sum()) * baseline_values[0]
                + int((y_train == 1).sum()) * baseline_values[1]
            )

            test_baseline_value = (
                int((y_test == 0).sum()) * baseline_values[0]
                + int((y_test == 1).sum()) * baseline_values[1]
            )

            if minimizing_business_cost:

                train_bva = (
                    train_baseline_value - train_business_value
                )

                test_bva = (
                    test_baseline_value - test_business_value
                )

                baseline_column_label = "Baseline Cost/Loss"

                model_column_label = "Model Cost/Loss"

                improvement_column_label = "Cost Savings"

                improvement_percent_label = "Savings %"

            else:

                train_bva = (
                    train_business_value - train_baseline_value
                )

                test_bva = (
                    test_business_value - test_baseline_value
                )

                baseline_column_label = "Baseline Business Value"

                model_column_label = "Model Business Value"

                improvement_column_label = "Business Value Added"

                improvement_percent_label = "BVA %"

            train_bva_percent = (
                train_bva / train_baseline_value * 100
                if train_baseline_value != 0
                else np.nan
            )

            test_bva_percent = (
                test_bva / test_baseline_value * 100
                if test_baseline_value != 0
                else np.nan
            )

            business_performance_df = pd.DataFrame(
                [
                    {
                        "Data Set": "Training",
                        baseline_column_label: (
                            f"{train_baseline_value:,.2f}"
                        ),
                        model_column_label: (
                            f"{train_business_value:,.2f}"
                        ),
                        improvement_column_label: (
                            f"{train_bva:,.2f}"
                        ),
                        improvement_percent_label: (
                            f"{train_bva_percent:.2f}%"
                            if not pd.isna(train_bva_percent)
                            else "N/A"
                        )
                    },
                    {
                        "Data Set": "Testing",
                        baseline_column_label: (
                            f"{test_baseline_value:,.2f}"
                        ),
                        model_column_label: (
                            f"{test_business_value:,.2f}"
                        ),
                        improvement_column_label: (
                            f"{test_bva:,.2f}"
                        ),
                        improvement_percent_label: (
                            f"{test_bva_percent:.2f}%"
                            if not pd.isna(test_bva_percent)
                            else "N/A"
                        )
                    }
                ]
            )

            if minimizing_business_cost:

                st.write(
                    "Baseline Cost/Loss represents the expected amount "
                    "without using the model. Cost Savings equals Baseline "
                    "Cost/Loss minus Model Cost/Loss."
                )

            else:

                st.write(
                    "Baseline Business Value represents the expected value "
                    "without using the model. Business Value Added equals "
                    "Model Business Value minus Baseline Business Value."
                )

            st.markdown(
                business_performance_df.to_html(
                    index=False,
                    escape=True,
                    classes="business-table"
                ),
                unsafe_allow_html=True
            )

            if (
                train_baseline_value == 0
                or test_baseline_value == 0
            ):

                st.write(
                    "The percentage improvement is unavailable when the "
                    "corresponding baseline amount equals zero."
                )

        else:

            business_col1, business_col2 = st.columns(2)

            business_total_label = (
                "Total Cost/Loss"
                if optimization_direction == "Minimize cost or loss"
                else "Total Business Value"
            )

            business_average_label = (
                "Average cost/loss per prediction"
                if optimization_direction == "Minimize cost or loss"
                else "Average business value per prediction"
            )

            with business_col1:

                st.metric(
                    f"Training {business_total_label}",
                    f"{train_business_value:,.2f}"
                )

                st.write(
                    f"{business_average_label}: "
                    f"**{train_business_value / len(y_train):,.2f}**"
                )

            with business_col2:

                st.metric(
                    f"Testing {business_total_label}",
                    f"{test_business_value:,.2f}"
                )

                st.write(
                    f"{business_average_label}: "
                    f"**{test_business_value / len(y_test):,.2f}**"
                )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    output_section(
        "Model Performance Metrics"
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
            "Combines precision and recall into one score at the selected "
            "cutoff. Use it when finding class 1 matters and both missed "
            "cases and false alarms are important."
        ),
        "ROC AUC": (
            "Probability that a randomly selected class 1 case receives a "
            "higher predicted score than a randomly selected class 0 case. "
            "Use it to compare ranking ability before selecting a cutoff."
        ),
        "PR AUC": (
            "Summarizes how successfully the model finds class 1 cases "
            "while limiting incorrect class 1 predictions across cutoffs. "
            "Most useful when class 1 is uncommon."
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


    st.markdown(
        performance_df.to_html(
            index=False,
            escape=True,
            classes="performance-table"
        ),
        unsafe_allow_html=True
    )


    # ========================================================
    # CUTOFF × DEPTH ANALYSIS
    # ========================================================

    output_section(
        "Cross-Validated Cutoff × Tree Depth Analysis"
    )


    st.write(
        f"""
The table and 3D graph below evaluate **{metric_name}**
using cross-validation on the training data for every
classification cutoff from **0.00 to 1.00** and every
tree depth you allowed. These values are used for model
selection and are **not testing-set performance**.
        """
    )


    cutoff_independent_metrics = {
        "ROC AUC",
        "PR AUC"
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
            xaxis=dict(
                title=dict(
                    text="Tree Depth",
                    font=dict(size=16)
                ),
                tickfont=dict(size=13),
                dtick=1
            ),
            yaxis=dict(
                title=dict(
                    text="Classification Cutoff",
                    font=dict(size=16)
                ),
                tickfont=dict(size=13)
            ),
            zaxis=dict(
                title=dict(
                    text=surface_z_title,
                    font=dict(size=16)
                ),
                tickformat=surface_tick_format,
                tickfont=dict(size=13)
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

    output_section(
        "Download Predictions"
    )


    model_df = (
        st.session_state.model_df
    )


    selected_pipeline_for_download = (
        st.session_state.selected_pipeline
    )

    global_pipeline_for_download = make_pipeline(
        global_depth,
        min_samples_leaf_saved,
        min_samples_split_saved,
        numeric_features,
        categorical_features,
        random_seed_saved
    )

    global_pipeline_for_download.fit(
        X_train,
        y_train
    )

    selected_train_probabilities = (
        selected_pipeline_for_download.predict_proba(
            X_train
        )[:, 1]
    )

    selected_test_probabilities = (
        selected_pipeline_for_download.predict_proba(
            X_test
        )[:, 1]
    )

    global_train_probabilities = (
        global_pipeline_for_download.predict_proba(
            X_train
        )[:, 1]
    )

    global_test_probabilities = (
        global_pipeline_for_download.predict_proba(
            X_test
        )[:, 1]
    )

    selected_train_predictions = (
        selected_train_probabilities
        >= chosen_cutoff_saved
    ).astype(int)

    selected_test_predictions = (
        selected_test_probabilities
        >= chosen_cutoff_saved
    ).astype(int)

    global_train_predictions = (
        global_train_probabilities
        >= global_cutoff
    ).astype(int)

    global_test_predictions = (
        global_test_probabilities
        >= global_cutoff
    ).astype(int)


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
        "Predicted_Probability_of_1_Selected_Model"
    ] = selected_train_probabilities

    selected_class_column = (
        "Predicted_Class_Selected_Cutoff_"
        f"{chosen_cutoff_saved:.2f}"
    )

    optimal_class_column = (
        "Predicted_Class_Optimal_Cutoff_"
        f"{global_cutoff:.2f}"
    )

    train_output[
        selected_class_column
    ] = selected_train_predictions

    train_output[
        "Predicted_Probability_of_1_Optimal_Model"
    ] = global_train_probabilities

    train_output[
        optimal_class_column
    ] = global_train_predictions


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
        "Predicted_Probability_of_1_Selected_Model"
    ] = selected_test_probabilities

    test_output[
        selected_class_column
    ] = selected_test_predictions

    test_output[
        "Predicted_Probability_of_1_Optimal_Model"
    ] = global_test_probabilities

    test_output[
        optimal_class_column
    ] = global_test_predictions


    if metric_name == "Business Value":

        train_output[
            "Business_Value_Selected_Model"
        ] = business_value_per_prediction(
            y_train.values,
            selected_train_predictions,
            business_values
        )

        test_output[
            "Business_Value_Selected_Model"
        ] = business_value_per_prediction(
            y_test.values,
            selected_test_predictions,
            business_values
        )

        train_output[
            "Business_Value_Optimal_Model"
        ] = business_value_per_prediction(
            y_train.values,
            global_train_predictions,
            business_values
        )

        test_output[
            "Business_Value_Optimal_Model"
        ] = business_value_per_prediction(
            y_test.values,
            global_test_predictions,
            business_values
        )


    output_df = pd.concat(
        [
            train_output,
            test_output
        ]
    ).sort_index()


    st.write(
        f"""
The downloaded file includes results from both model-selection policies:

**Selected model:** depth {selected_depth}, cutoff {chosen_cutoff_saved:.2f}

**Cross-validated optimal model:** depth {global_depth}, cutoff {global_cutoff:.2f}

`Predicted_Probability_of_1` is the model's predicted probability that the
outcome equals 1. When Business Value is selected, the file also contains the
Cost-Benefit Matrix value assigned to each prediction outcome.
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
