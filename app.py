import io
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"
MAX_CONTEXT_ROWS = 8
MAX_CONTEXT_COLUMNS = 6
MAX_GROUP_SUMMARIES = 2
MAX_FUZZY_SEARCH_ROWS = 12000
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "data",
    "dataset",
    "do",
    "does",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "show",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "us",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}
CHART_COLORS = ["#0B6E4F", "#117864", "#2E86C1", "#D35400", "#AF601A"]


def normalize_text(text):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", normalized).strip()


def simplify_token(token):
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text):
    return [
        simplify_token(token)
        for token in normalize_text(text).split()
        if len(token) >= 3 and token not in STOPWORDS
    ]


def unique_in_order(items):
    unique_items = []
    seen = set()
    for item in items:
        if item and item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


def truncate_text(value, max_length=60):
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def token_overlap_score(question_tokens, text):
    text_tokens = set(tokenize(text))
    if not question_tokens or not text_tokens:
        return 0.0

    score = 0.0
    for question_token in question_tokens:
        if question_token in text_tokens:
            score += 2.0
            continue

        for text_token in text_tokens:
            if question_token in text_token or text_token in question_token:
                score += 1.0
                break

    return score


def collect_sample_values(series, limit=6):
    values = []
    for value in series.dropna().astype(str).head(400):
        cleaned = value.strip()
        if cleaned and cleaned not in values:
            values.append(truncate_text(cleaned, 40))
        if len(values) >= limit:
            break
    return values


def compute_numeric_stats(series):
    numeric_values = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_values.empty:
        return None

    return {
        "min": round(float(numeric_values.min()), 2),
        "median": round(float(numeric_values.median()), 2),
        "mean": round(float(numeric_values.mean()), 2),
        "max": round(float(numeric_values.max()), 2),
    }


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    if "Rank" in df.columns:
        df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def build_column_profiles(df):
    profiles = []
    row_count = len(df)

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        non_null_count = int(non_null.shape[0])
        unique_count = int(non_null.nunique()) if non_null_count else 0
        unique_ratio = unique_count / max(non_null_count, 1)
        is_numeric = pd.api.types.is_numeric_dtype(series)

        profile = {
            "column": column,
            "dtype": str(series.dtype),
            "row_count": row_count,
            "non_null_count": non_null_count,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "is_numeric": is_numeric,
            "sample_values": collect_sample_values(series),
        }

        if is_numeric:
            profile["stats"] = compute_numeric_stats(series)
        else:
            profile["stats"] = None

        profiles.append(profile)

    return profiles


@st.cache_data(show_spinner=False)
def detect_date_columns(df):
    date_columns = []

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(column)
            continue

        if pd.api.types.is_numeric_dtype(series):
            continue

        sample = series.dropna().astype(str).head(500)
        if sample.empty:
            continue

        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.85:
            date_columns.append(column)

    return date_columns


def get_text_columns(df, exclude_columns=None):
    exclude_columns = set(exclude_columns or [])
    return [
        column
        for column in df.columns
        if column not in exclude_columns and not pd.api.types.is_numeric_dtype(df[column])
    ]


def score_column_profile(profile, question, question_tokens, selected_column):
    name_score = token_overlap_score(question_tokens, profile["column"]) * 3
    sample_score = token_overlap_score(question_tokens, " ".join(profile["sample_values"]))
    normalized_question = normalize_text(question)
    normalized_column = normalize_text(profile["column"])
    exact_name_bonus = 3.0 if normalized_column and normalized_column in normalized_question else 0.0
    selection_bonus = 1.0 if profile["column"] == selected_column else 0.0
    return name_score + sample_score + exact_name_bonus + selection_bonus


def rank_column_profiles(df, question, selected_column):
    question_tokens = tokenize(question)
    profiles = build_column_profiles(df)

    for profile in profiles:
        profile["relevance_score"] = score_column_profile(
            profile,
            question,
            question_tokens,
            selected_column,
        )

    return sorted(
        profiles,
        key=lambda profile: (
            profile["relevance_score"],
            profile["non_null_count"],
            -profile["unique_ratio"],
        ),
        reverse=True,
    )


def choose_metric_column(ranked_profiles, selected_column):
    numeric_profiles = [profile for profile in ranked_profiles if profile["is_numeric"]]
    if not numeric_profiles:
        return None

    if selected_column:
        for profile in numeric_profiles:
            if profile["column"] == selected_column and profile["relevance_score"] >= 0:
                return selected_column

    return numeric_profiles[0]["column"]


def choose_group_columns(ranked_profiles, metric_column, excluded_columns=None, limit=MAX_GROUP_SUMMARIES):
    excluded_columns = set(excluded_columns or [])
    group_profiles = []

    for profile in ranked_profiles:
        if profile["column"] == metric_column or profile["column"] in excluded_columns or profile["is_numeric"]:
            continue
        if profile["non_null_count"] < 2:
            continue
        if profile["unique_count"] <= 1:
            continue
        group_profiles.append(profile)

    return [profile["column"] for profile in group_profiles[:limit]]


def choose_preview_columns(df, selected_column):
    date_columns = detect_date_columns(df)
    text_columns = get_text_columns(df, exclude_columns=date_columns)
    preview_columns = []

    if date_columns:
        preview_columns.append(date_columns[0])

    preview_columns.extend(text_columns[:2])
    preview_columns.append(selected_column)

    return unique_in_order([column for column in preview_columns if column in df.columns])[:MAX_CONTEXT_COLUMNS]


def build_record_label(df, label_columns):
    if not label_columns:
        return df.index.astype(str)

    label = df[label_columns[0]].astype(str).map(lambda value: truncate_text(value, 32))
    if len(label_columns) > 1:
        secondary = df[label_columns[1]].astype(str).map(lambda value: truncate_text(value, 28))
        label = label + " | " + secondary

    return label


def format_column_profile(profile):
    if profile["is_numeric"] and profile["stats"]:
        stats = profile["stats"]
        return (
            f"- {profile['column']} ({profile['dtype']}, {profile['non_null_count']} non-null): "
            f"min={stats['min']}, median={stats['median']}, mean={stats['mean']}, max={stats['max']}"
        )

    sample_values = ", ".join(profile["sample_values"][:3]) or "no sample values"
    return (
        f"- {profile['column']} ({profile['dtype']}, {profile['non_null_count']} non-null, "
        f"{profile['unique_count']} unique): examples={sample_values}"
    )


def format_table(df_subset):
    if df_subset is None or df_subset.empty:
        return "No rows available"

    display_df = df_subset.copy()
    for column in display_df.columns:
        if not pd.api.types.is_numeric_dtype(display_df[column]):
            display_df[column] = display_df[column].astype(str).map(lambda value: truncate_text(value, 60))

    return display_df.to_string(index=False)


def extract_candidate_phrases(question):
    words = [word for word in normalize_text(question).split() if len(word) >= 3 and word not in STOPWORDS]
    phrases = []

    for size in range(min(4, len(words)), 1, -1):
        for index in range(len(words) - size + 1):
            phrase = " ".join(words[index : index + size])
            if phrase not in phrases:
                phrases.append(phrase)

    return phrases


def find_exact_text_matches(df, question):
    text_columns = get_text_columns(df)
    if not text_columns:
        return None, [], None

    candidate_phrases = extract_candidate_phrases(question)
    if not candidate_phrases:
        return None, [], None

    for phrase in candidate_phrases:
        combined_mask = pd.Series(False, index=df.index)
        matched_columns = []

        for column in text_columns:
            normalized_series = df[column].fillna("").astype(str).map(normalize_text)
            mask = normalized_series.str.contains(re.escape(phrase), regex=True)
            if mask.any():
                combined_mask |= mask
                matched_columns.append(column)

        if combined_mask.any():
            matched_rows = df.loc[combined_mask].copy()
            return matched_rows, matched_columns, phrase

    return None, [], None


def select_textually_related_rows(df, question_tokens, candidate_columns):
    if not question_tokens:
        return None

    search_columns = candidate_columns or df.columns.tolist()
    search_df = df[search_columns].head(MAX_FUZZY_SEARCH_ROWS).copy()
    if search_df.empty:
        return None

    scores = search_df.fillna("").astype(str).apply(
        lambda row: token_overlap_score(question_tokens, " ".join(row.values.tolist())),
        axis=1,
    )
    top_indices = scores[scores > 0].sort_values(ascending=False).head(MAX_CONTEXT_ROWS).index

    if top_indices.empty:
        return None

    return df.loc[top_indices, search_columns]


def build_group_summary(df, group_column, metric_column):
    working_df = df[[group_column, metric_column]].copy()
    working_df[metric_column] = pd.to_numeric(working_df[metric_column], errors="coerce")
    working_df = working_df.dropna(subset=[group_column, metric_column])

    if working_df.empty:
        return None

    global_mean = working_df[metric_column].mean()
    prior_count = max(3, int(working_df[group_column].value_counts().median()))

    grouped = (
        working_df.groupby(group_column, dropna=True)[metric_column]
        .agg(average="mean", maximum="max", minimum="min", count="count")
        .reset_index()
    )

    if grouped.empty:
        return None

    for column in ["average", "maximum", "minimum"]:
        grouped[column] = grouped[column].map(lambda value: round(float(value), 2))

    grouped["stability_score"] = (
        (grouped["average"] * grouped["count"]) + (global_mean * prior_count)
    ) / (grouped["count"] + prior_count)
    grouped = grouped.sort_values(
        ["stability_score", "count", "maximum"],
        ascending=False,
    ).head(MAX_CONTEXT_ROWS)
    grouped[group_column] = grouped[group_column].astype(str).map(lambda value: truncate_text(value, 60))
    grouped["stability_score"] = grouped["stability_score"].round(2)

    return grouped


def build_exact_match_summary(match_df, metric_column, date_columns):
    if match_df is None or match_df.empty or not metric_column or metric_column not in match_df.columns:
        return None

    numeric_values = pd.to_numeric(match_df[metric_column], errors="coerce").dropna()
    if numeric_values.empty:
        return None

    summary_parts = [
        f"matched rows={len(match_df)}",
        f"min={numeric_values.min():.2f}",
        f"mean={numeric_values.mean():.2f}",
        f"max={numeric_values.max():.2f}",
    ]

    for date_column in date_columns:
        parsed_dates = pd.to_datetime(match_df[date_column], errors="coerce")
        if parsed_dates.notna().any():
            latest_index = parsed_dates.idxmax()
            latest_value = numeric_values.reindex(match_df.index).reindex([latest_index]).dropna()
            if not latest_value.empty:
                summary_parts.append(
                    f"latest {date_column}={match_df.loc[latest_index, date_column]}"
                )
                summary_parts.append(f"latest {metric_column}={latest_value.iloc[0]:.2f}")
            break

    return ", ".join(summary_parts)


def build_insights_context(df, selected_column):
    ranked_profiles = rank_column_profiles(df, selected_column, selected_column)
    focus_profiles = ranked_profiles[: min(5, len(ranked_profiles))]
    preview_columns = choose_preview_columns(df, selected_column)

    context_lines = [
        f"Dataset row count: {len(df)}",
        f"Dataset columns: {', '.join(map(str, df.columns.tolist()))}",
        f"Current focus metric: {selected_column}",
        "Column profiles:",
        "\n".join(format_column_profile(profile) for profile in focus_profiles),
        f"Preview rows:\n{format_table(df[preview_columns].head(MAX_CONTEXT_ROWS))}",
    ]

    if selected_column in df.columns:
        ranked_df = df.copy()
        ranked_df[selected_column] = pd.to_numeric(ranked_df[selected_column], errors="coerce")
        ranked_df = ranked_df.dropna(subset=[selected_column]).sort_values(selected_column, ascending=False)

        if not ranked_df.empty:
            context_lines.append(
                f"Top rows by {selected_column}:\n"
                f"{format_table(ranked_df[preview_columns].head(MAX_CONTEXT_ROWS))}"
            )

    return "\n\n".join(context_lines)


def build_chat_context(df, question, selected_column):
    ranked_profiles = rank_column_profiles(df, question, selected_column)
    question_tokens = tokenize(question)
    date_columns = detect_date_columns(df)
    metric_column = choose_metric_column(ranked_profiles, selected_column)
    exact_match_rows, exact_match_columns, exact_phrase = find_exact_text_matches(df, question)
    group_columns = choose_group_columns(ranked_profiles, metric_column, excluded_columns=date_columns)

    candidate_columns = unique_in_order(
        [metric_column]
        + exact_match_columns
        + date_columns[:1]
        + group_columns
        + [profile["column"] for profile in ranked_profiles[:MAX_CONTEXT_COLUMNS]]
    )
    candidate_columns = [column for column in candidate_columns if column in df.columns][:MAX_CONTEXT_COLUMNS]

    if metric_column and metric_column not in candidate_columns:
        candidate_columns.append(metric_column)

    related_rows = select_textually_related_rows(df, question_tokens, candidate_columns)
    relevant_profiles = [profile for profile in ranked_profiles if profile["column"] in candidate_columns]
    exact_match_summary = build_exact_match_summary(exact_match_rows, metric_column, date_columns)

    context_lines = [
        f"Dataset row count: {len(df)}",
        f"Dataset columns: {', '.join(map(str, df.columns.tolist()))}",
        f"User question: {question}",
        f"Likely metric column: {metric_column or 'none found'}",
        f"Context columns chosen from schema and content: {', '.join(candidate_columns)}",
        "Relevant column profiles:",
        "\n".join(format_column_profile(profile) for profile in relevant_profiles),
    ]

    if exact_match_rows is not None and not exact_match_rows.empty:
        context_lines.append(
            f"Exact text matches found for phrase '{exact_phrase}' in columns: {', '.join(exact_match_columns)}"
        )
        if exact_match_summary:
            context_lines.append(f"Exact match metric summary: {exact_match_summary}")
        context_lines.append(
            f"Exact match rows:\n{format_table(exact_match_rows[candidate_columns].head(MAX_CONTEXT_ROWS))}"
        )

    if metric_column and metric_column in df.columns:
        ranked_df = df[candidate_columns].copy()
        ranked_df[metric_column] = pd.to_numeric(df[metric_column], errors="coerce")
        ranked_df = ranked_df.dropna(subset=[metric_column])

        if not ranked_df.empty:
            top_rows = ranked_df.sort_values(metric_column, ascending=False).head(MAX_CONTEXT_ROWS)
            bottom_rows = ranked_df.sort_values(metric_column, ascending=True).head(MAX_CONTEXT_ROWS)

            context_lines.append(f"Top rows by {metric_column}:\n{format_table(top_rows)}")
            context_lines.append(f"Bottom rows by {metric_column}:\n{format_table(bottom_rows)}")

            for group_column in group_columns:
                grouped = build_group_summary(df, group_column, metric_column)
                if grouped is not None:
                    context_lines.append(
                        f"Grouped summary by {group_column} using {metric_column}:\n"
                        f"{format_table(grouped)}"
                    )

    if related_rows is not None and not related_rows.empty:
        context_lines.append(
            f"Rows most textually related to the question:\n{format_table(related_rows.head(MAX_CONTEXT_ROWS))}"
        )

    context_lines.append(
        f"Sample rows:\n{format_table(df[candidate_columns].head(MAX_CONTEXT_ROWS))}"
    )

    return "\n\n".join(context_lines)


def render_metric_cards(df, selected_column):
    metric_values = pd.to_numeric(df[selected_column], errors="coerce").dropna()
    if metric_values.empty:
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Max", f"{metric_values.max():.2f}")
    metric_columns[1].metric("Median", f"{metric_values.median():.2f}")
    metric_columns[2].metric("Average", f"{metric_values.mean():.2f}")
    metric_columns[3].metric("Rows", f"{len(metric_values):,}")


def render_distribution_chart(df, selected_column):
    metric_df = df[[selected_column]].copy()
    metric_df[selected_column] = pd.to_numeric(metric_df[selected_column], errors="coerce")
    metric_df = metric_df.dropna(subset=[selected_column])

    histogram = px.histogram(
        metric_df,
        x=selected_column,
        nbins=min(40, max(12, len(metric_df) // 250)),
        marginal="box",
        color_discrete_sequence=[CHART_COLORS[0]],
        title=f"Distribution of {selected_column}",
    )
    histogram.update_layout(
        template="plotly_white",
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    histogram.update_traces(marker_line_width=1, marker_line_color="#ffffff")
    st.plotly_chart(histogram, use_container_width=True)


def render_leaderboard_chart(df, selected_column):
    date_columns = detect_date_columns(df)
    label_columns = get_text_columns(df, exclude_columns=date_columns)[:2]

    leaders_df = df.copy()
    leaders_df[selected_column] = pd.to_numeric(leaders_df[selected_column], errors="coerce")
    leaders_df = leaders_df.dropna(subset=[selected_column]).sort_values(selected_column, ascending=False).head(12)

    if leaders_df.empty:
        st.info("Not enough data to build a leaderboard chart.")
        return

    leaders_df = leaders_df.copy()
    leaders_df["record_label"] = build_record_label(leaders_df, label_columns)

    leaderboard = px.bar(
        leaders_df.iloc[::-1],
        x=selected_column,
        y="record_label",
        orientation="h",
        color=selected_column,
        color_continuous_scale=["#D5F5E3", "#0B6E4F"],
        title=f"Top records by {selected_column}",
        text=selected_column,
    )
    leaderboard.update_layout(
        template="plotly_white",
        yaxis_title="",
        xaxis_title=selected_column,
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    leaderboard.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(leaderboard, use_container_width=True)


def render_trend_chart(df, selected_column):
    date_columns = detect_date_columns(df)

    if date_columns:
        date_column = date_columns[0]
        trend_df = df[[date_column, selected_column]].copy()
        trend_df[date_column] = pd.to_datetime(trend_df[date_column], errors="coerce")
        trend_df[selected_column] = pd.to_numeric(trend_df[selected_column], errors="coerce")
        trend_df = trend_df.dropna(subset=[date_column, selected_column])

        if trend_df.empty:
            st.info("Not enough valid date values to build a trend chart.")
            return

        aggregated = (
            trend_df.groupby(date_column)[selected_column]
            .agg(mean="mean", median="median", maximum="max")
            .reset_index()
            .sort_values(date_column)
        )

        trend_chart = go.Figure()
        trend_chart.add_trace(
            go.Scatter(
                x=aggregated[date_column],
                y=aggregated["mean"],
                mode="lines",
                name="Average",
                line=dict(color=CHART_COLORS[0], width=3),
            )
        )
        trend_chart.add_trace(
            go.Scatter(
                x=aggregated[date_column],
                y=aggregated["median"],
                mode="lines",
                name="Median",
                line=dict(color=CHART_COLORS[2], width=2, dash="dot"),
            )
        )
        trend_chart.add_trace(
            go.Scatter(
                x=aggregated[date_column],
                y=aggregated["maximum"],
                mode="lines",
                name="Daily max",
                line=dict(color=CHART_COLORS[3], width=2, dash="dash"),
            )
        )
        trend_chart.update_layout(
            title=f"{selected_column} trend over time",
            template="plotly_white",
            xaxis_title=date_column,
            yaxis_title=selected_column,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(trend_chart, use_container_width=True)
        return

    trend_df = df[[selected_column]].copy()
    trend_df[selected_column] = pd.to_numeric(trend_df[selected_column], errors="coerce")
    trend_df = trend_df.dropna(subset=[selected_column]).reset_index()

    if trend_df.empty:
        st.info("Not enough numeric values to build a trend chart.")
        return

    trend_df = trend_df.rename(columns={"index": "row_index"})
    window = max(25, min(250, max(25, len(trend_df) // 40)))
    trend_df["rolling_average"] = trend_df[selected_column].rolling(
        window=window,
        min_periods=max(5, window // 5),
    ).mean()

    fallback_chart = go.Figure()
    fallback_chart.add_trace(
        go.Scattergl(
            x=trend_df["row_index"],
            y=trend_df[selected_column],
            mode="markers",
            name="Rows",
            marker=dict(color=CHART_COLORS[2], size=5, opacity=0.22),
        )
    )
    fallback_chart.add_trace(
        go.Scatter(
            x=trend_df["row_index"],
            y=trend_df["rolling_average"],
            mode="lines",
            name=f"Rolling average ({window})",
            line=dict(color=CHART_COLORS[0], width=3),
        )
    )
    fallback_chart.update_layout(
        title=f"{selected_column} pattern across rows",
        template="plotly_white",
        xaxis_title="Row order",
        yaxis_title=selected_column,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fallback_chart, use_container_width=True)


def render_visualization_preview(df, selected_column):
    preview_columns = choose_preview_columns(df, selected_column)
    preview_df = df.copy()
    preview_df[selected_column] = pd.to_numeric(preview_df[selected_column], errors="coerce")
    preview_df = preview_df.dropna(subset=[selected_column]).sort_values(selected_column, ascending=False)

    st.write("Most relevant preview:")
    st.dataframe(preview_df[preview_columns].head(10), use_container_width=True)


st.set_page_config(page_title="Vyntaq AI", layout="wide")
st.title("Vyntaq AI")
st.caption("Grounded data analysis with cleaner visuals and evidence-backed answers.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_dataset(uploaded_file.getvalue())

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Basic Statistics")
    st.dataframe(df.describe(include="all").transpose().head(20), use_container_width=True)

    st.subheader("Interactive Visualization")

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        st.warning("No numeric columns available")
    else:
        selected_column = st.selectbox("Select metric to analyze", numeric_columns)

        render_metric_cards(df, selected_column)
        render_visualization_preview(df, selected_column)

        overview_tab, leaders_tab, trend_tab = st.tabs(["Distribution", "Leaders", "Trend"])

        with overview_tab:
            render_distribution_chart(df, selected_column)

        with leaders_tab:
            render_leaderboard_chart(df, selected_column)

        with trend_tab:
            render_trend_chart(df, selected_column)

        st.subheader("AI Insights")
        if st.button("Generate AI Insights"):
            summary = build_insights_context(df, selected_column)

            try:
                response = requests.post(
                    f"{BACKEND_URL}/insights",
                    json={"summary": summary},
                    timeout=60,
                )
                result = response.json()

                if "insight" in result:
                    st.success("AI Insights:")
                    st.write(result["insight"])
                else:
                    st.error("API Error")
                    st.write(result)
            except Exception as error:
                st.error("Backend connection failed")
                st.write(str(error))

        st.subheader("Chat with your Data")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Ask a question about your data")
        if st.button("Ask") and user_input:
            context = build_chat_context(df, user_input, selected_column)

            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": user_input, "context": context},
                    timeout=60,
                )
                result = response.json()

                if "response" in result:
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("AI", result["response"]))
                else:
                    st.error("API Error")
                    st.write(result)
            except Exception as error:
                st.error("Backend connection failed")
                st.write(str(error))

        for role, message in st.session_state.chat_history:
            if role == "You":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**Vyntaq:** {message}")
