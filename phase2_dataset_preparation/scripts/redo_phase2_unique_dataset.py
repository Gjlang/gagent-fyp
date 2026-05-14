import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("GAGENT_DATASET_ROOT", r"D:\FYP\Dataset"))
BEHAVIOURAL_ROOT = DATASET_ROOT / "Behavioral interaction datasets"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "processed_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CHUNK_SIZE = 200_000
TARGET_INTERACTIONS_ROWS = 90_000
MAX_ROWS_PER_INTERACTIONS_CHUNK = 4_000

FINAL_COLUMNS = [
    "source_dataset",
    "flow_type",
    "completion_time",
    "click_count",
    "scroll_count",
    "keyboard_count",
    "retry_count",
    "error_count",
    "failed_clicks",
    "feedback_delay",
    "task_completed",
    "screenshot_count",
    "error_message_clarity",
    "friction_level",
]

NUMERIC_COLUMNS = [
    "completion_time",
    "click_count",
    "scroll_count",
    "keyboard_count",
    "retry_count",
    "error_count",
    "failed_clicks",
    "feedback_delay",
    "task_completed",
    "screenshot_count",
    "error_message_clarity",
]

LABEL_ORDER = ["Low", "Medium", "High"]


def safe_numeric(series, default=-1):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def file_exists_or_skip(path: Path) -> bool:
    if not path.exists():
        print(f"[SKIP] File not found: {path}")
        return False
    return True


def clean_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in FINAL_COLUMNS:
        if col not in df.columns:
            if col == "friction_level":
                df[col] = "Unlabeled"
            elif col in ["source_dataset", "flow_type"]:
                df[col] = "unknown"
            else:
                df[col] = -1

    df["source_dataset"] = df["source_dataset"].fillna("unknown").astype(str)
    df["flow_type"] = df["flow_type"].fillna("unknown").astype(str)

    for col in NUMERIC_COLUMNS:
        df[col] = safe_numeric(df[col], default=-1)

    # Keep -1 as the official unknown sentinel.
    # Anything below -1 is invalid and should be clipped to -1.
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].clip(lower=-1)

    df["friction_level"] = df["friction_level"].astype(str)
    return df[FINAL_COLUMNS]


def parse_timestamp_to_seconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() > 0.8:
        median_value = numeric.dropna().median()

        # Most raw timestamps in interactions.csv look like epoch milliseconds.
        if median_value > 10_000_000_000:
            return numeric / 1000.0

        # Epoch seconds.
        if median_value > 1_000_000_000:
            return numeric

        return numeric

    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.astype("int64") / 1_000_000_000


def assign_friction_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_final_dataset(df)

    metric_cols = [
        "completion_time",
        "click_count",
        "scroll_count",
        "keyboard_count",
        "retry_count",
        "error_count",
        "failed_clicks",
        "feedback_delay",
    ]

    thresholds = {}
    for col in metric_cols:
        valid = df.loc[df[col] >= 0, col]
        if len(valid) == 0:
            thresholds[col] = {"p75": np.inf, "p90": np.inf}
        else:
            thresholds[col] = {
                "p75": valid.quantile(0.75),
                "p90": valid.quantile(0.90),
            }

    score = pd.Series(0, index=df.index, dtype="float64")

    score += np.where(df["task_completed"] == 0, 4, 0)

    for col in ["completion_time", "click_count", "scroll_count", "keyboard_count"]:
        score += np.where(df[col] > thresholds[col]["p90"], 2, 0)
        score += np.where(
            (df[col] > thresholds[col]["p75"]) & (df[col] <= thresholds[col]["p90"]),
            1,
            0,
        )

    score += np.where(df["retry_count"] >= 3, 2, np.where(df["retry_count"] >= 1, 1, 0))
    score += np.where(df["error_count"] >= 2, 3, np.where(df["error_count"] >= 1, 2, 0))
    score += np.where(df["failed_clicks"] >= 2, 3, np.where(df["failed_clicks"] >= 1, 2, 0))
    score += np.where(df["feedback_delay"] >= 6, 2, np.where(df["feedback_delay"] >= 3, 1, 0))
    score += np.where(df["error_message_clarity"] == 0, 2, np.where(df["error_message_clarity"] == 1, 1, 0))

    positive_scores = score[score > 0]
    if len(positive_scores) > 0:
        medium_dynamic = max(3, positive_scores.quantile(0.45))
        high_dynamic = max(7, positive_scores.quantile(0.80))
    else:
        medium_dynamic = 3
        high_dynamic = 7

    labels = np.where(
        score >= high_dynamic,
        "High",
        np.where(score >= medium_dynamic, "Medium", "Low"),
    )

    df["friction_level"] = labels
    return df[FINAL_COLUMNS]


def print_dedup_report(name: str, before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    original_rows = len(before_df)
    duplicate_rows = int(before_df.duplicated().sum())
    unique_rows = len(after_df)
    duplicate_percentage = (duplicate_rows / original_rows * 100) if original_rows else 0

    print(f"\n===== {name} =====")
    print(f"Original rows: {original_rows}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Unique rows: {unique_rows}")
    print(f"Duplicate percentage: {duplicate_percentage:.2f}%")

    print("\nClass distribution before deduplication:")
    print(before_df["friction_level"].value_counts(dropna=False))

    print("\nClass distribution after deduplication:")
    print(after_df["friction_level"].value_counts(dropna=False))

    print("\nSource distribution before deduplication:")
    print(before_df["source_dataset"].value_counts(dropna=False))

    print("\nSource distribution after deduplication:")
    print(after_df["source_dataset"].value_counts(dropna=False))


def process_interaction_telemetry() -> pd.DataFrame:
    path = BEHAVIOURAL_ROOT / "interaction_telemetry_large.csv"
    if not file_exists_or_skip(path):
        return pd.DataFrame(columns=FINAL_COLUMNS)

    print(f"\nProcessing: {path.name}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    for col in ["Session_ID", "Event_Type", "Event_Target", "Error_Code", "Session_Length_s", "Dwell_Time_ms", "Conversion"]:
        if col not in df.columns:
            df[col] = np.nan

    df["Session_ID"] = df["Session_ID"].fillna("unknown_session").astype(str)
    df["Event_Type"] = df["Event_Type"].fillna("").astype(str).str.lower()
    df["Event_Target"] = df["Event_Target"].fillna("general_interaction").astype(str)
    df["Error_Code"] = df["Error_Code"].fillna("").astype(str)

    df["Session_Length_s"] = safe_numeric(df["Session_Length_s"], default=0)
    df["Dwell_Time_ms"] = safe_numeric(df["Dwell_Time_ms"], default=0)
    df["Conversion"] = safe_numeric(df["Conversion"], default=0)

    df["has_error"] = (~df["Error_Code"].str.strip().str.lower().isin(["", "nan", "none", "0"])).astype(int)
    df["is_click"] = df["Event_Type"].str.contains("click", case=False, na=False).astype(int)
    df["is_scroll"] = df["Event_Type"].str.contains("scroll", case=False, na=False).astype(int)
    df["is_keyboard"] = df["Event_Type"].str.contains("input|key|type|keyboard", case=False, na=False).astype(int)

    grouped = df.groupby("Session_ID").agg(
        source_dataset=("Session_ID", lambda x: "interaction_telemetry_large.csv"),
        flow_type=("Event_Target", lambda x: x.mode().iloc[0] if not x.mode().empty else "general_interaction"),
        completion_time=("Session_Length_s", "max"),
        click_count=("is_click", "sum"),
        scroll_count=("is_scroll", "sum"),
        keyboard_count=("is_keyboard", "sum"),
        error_count=("has_error", "sum"),
        feedback_delay=("Dwell_Time_ms", lambda x: x.mean() / 1000),
        task_completed=("Conversion", "max"),
    ).reset_index()

    retry_data = df.groupby(["Session_ID", "Event_Target"]).size().reset_index(name="target_count")
    retry_data["retry_extra"] = retry_data["target_count"].apply(lambda x: max(x - 1, 0))
    retry_summary = retry_data.groupby("Session_ID")["retry_extra"].sum().reset_index(name="retry_count")

    grouped = grouped.merge(retry_summary, on="Session_ID", how="left")
    grouped["retry_count"] = grouped["retry_count"].fillna(0)
    grouped["failed_clicks"] = np.where(
        (grouped["click_count"] > 0) & (grouped["error_count"] > 0),
        grouped["error_count"],
        0,
    )
    grouped["screenshot_count"] = 0
    grouped["error_message_clarity"] = -1

    return clean_final_dataset(grouped.drop(columns=["Session_ID"], errors="ignore"))


def process_eshop() -> pd.DataFrame:
    path = BEHAVIOURAL_ROOT / "e-shop clothing 2008.csv"
    if not file_exists_or_skip(path):
        return pd.DataFrame(columns=FINAL_COLUMNS)

    print(f"\nProcessing: {path.name}")
    df = pd.read_csv(path, sep=";", low_memory=False)
    df.columns = df.columns.str.strip()

    if "session ID" not in df.columns:
        print("[SKIP] e-shop dataset has no session ID column.")
        return pd.DataFrame(columns=FINAL_COLUMNS)

    df["session ID"] = df["session ID"].fillna("unknown_session").astype(str)

    category_col = "page 1 (main category)" if "page 1 (main category)" in df.columns else None
    page_col = "page" if "page" in df.columns else None

    if category_col and page_col:
        df["flow_type_proxy"] = (
            "category_" + df[category_col].astype(str) +
            "_page_" + df[page_col].astype(str)
        )
    elif category_col:
        df["flow_type_proxy"] = "category_" + df[category_col].astype(str)
    elif page_col:
        df["flow_type_proxy"] = "page_" + df[page_col].astype(str)
    else:
        df["flow_type_proxy"] = "ecommerce_browsing"

    if "order" in df.columns:
        df["order_numeric"] = safe_numeric(df["order"], default=0)
    else:
        df["order_numeric"] = df.groupby("session ID").cumcount() + 1

    grouped = df.groupby("session ID").agg(
        source_dataset=("session ID", lambda x: "e-shop clothing 2008.csv"),
        flow_type=("flow_type_proxy", lambda x: x.mode().iloc[0] if not x.mode().empty else "ecommerce_browsing"),
        click_count=("session ID", "count"),
        min_order=("order_numeric", "min"),
        max_order=("order_numeric", "max"),
    ).reset_index()

    grouped["completion_time"] = (grouped["max_order"] - grouped["min_order"] + 1).clip(lower=1)

    if "page 2 (clothing model)" in df.columns:
        retry_data = df.groupby(["session ID", "page 2 (clothing model)"]).size().reset_index(name="count")
        retry_data["retry_extra"] = retry_data["count"].apply(lambda x: max(x - 1, 0))
        retry_summary = retry_data.groupby("session ID")["retry_extra"].sum().reset_index(name="retry_count")
        grouped = grouped.merge(retry_summary, on="session ID", how="left")
        grouped["retry_count"] = grouped["retry_count"].fillna(0)
    else:
        grouped["retry_count"] = 0

    grouped["scroll_count"] = -1
    grouped["keyboard_count"] = -1
    grouped["error_count"] = 0
    grouped["failed_clicks"] = 0
    grouped["feedback_delay"] = -1
    grouped["task_completed"] = -1
    grouped["screenshot_count"] = 0
    grouped["error_message_clarity"] = -1

    grouped = grouped.drop(columns=["session ID", "min_order", "max_order"], errors="ignore")
    return clean_final_dataset(grouped)


def process_interactions_large() -> pd.DataFrame:
    path = BEHAVIOURAL_ROOT / "interactions.csv"
    if not file_exists_or_skip(path):
        return pd.DataFrame(columns=FINAL_COLUMNS)

    print(f"\nProcessing large file in chunks: {path.name}")
    collected = []
    collected_rows = 0

    for chunk_number, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False), start=1):
        print(f"Processing chunk {chunk_number}")
        chunk.columns = chunk.columns.str.strip()

        for col in ["user_id", "timestamp", "url", "mouse.clicks", "keyboard"]:
            if col not in chunk.columns:
                chunk[col] = np.nan

        chunk["user_id"] = chunk["user_id"].fillna("unknown_user").astype(str)
        chunk["url"] = chunk["url"].fillna("unknown_flow").astype(str)

        chunk = chunk.reset_index(drop=True)
        chunk["session_window"] = chunk.index // 50
        chunk["synthetic_session_id"] = (
            chunk["user_id"].astype(str)
            + "_chunk"
            + str(chunk_number)
            + "_window"
            + chunk["session_window"].astype(str)
        )

        chunk["timestamp_seconds"] = parse_timestamp_to_seconds(chunk["timestamp"])
        chunk["timestamp_seconds"] = chunk["timestamp_seconds"].replace([np.inf, -np.inf], np.nan)

        click_candidates = ["mouse.clicks", "mouse.clicks.left", "mouse.clicks.right", "mouse.clicks.middle", "mouse.clicks.others"]
        for col in click_candidates:
            if col not in chunk.columns:
                chunk[col] = False

        click_flags = []
        for col in click_candidates:
            click_flags.append(chunk[col].astype(str).str.lower().isin(["true", "1", "yes"]))
        chunk["is_click"] = np.logical_or.reduce(click_flags).astype(int)

        keyboard_candidates = ["keyboard", "keyboard.alpha", "keyboard.numeric", "keyboard.function", "keyboard.symbol"]
        for col in keyboard_candidates:
            if col not in chunk.columns:
                chunk[col] = False

        keyboard_flags = []
        for col in keyboard_candidates:
            keyboard_flags.append(chunk[col].astype(str).str.lower().isin(["true", "1", "yes"]))
        chunk["is_keyboard"] = np.logical_or.reduce(keyboard_flags).astype(int)

        scroll_cols = ["scroll.absolute.x", "scroll.absolute.y", "scroll.relative.x", "scroll.relative.y"]
        for col in scroll_cols:
            if col not in chunk.columns:
                chunk[col] = 0
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

        chunk["is_scroll"] = (
            (chunk["scroll.absolute.x"] != 0)
            | (chunk["scroll.absolute.y"] != 0)
            | (chunk["scroll.relative.x"] != 0)
            | (chunk["scroll.relative.y"] != 0)
        ).astype(int)

        grouped = chunk.groupby("synthetic_session_id").agg(
            source_dataset=("synthetic_session_id", lambda x: "interactions.csv"),
            flow_type=("url", lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown_flow"),
            min_time=("timestamp_seconds", "min"),
            max_time=("timestamp_seconds", "max"),
            event_count=("synthetic_session_id", "count"),
            click_count=("is_click", "sum"),
            scroll_count=("is_scroll", "sum"),
            keyboard_count=("is_keyboard", "sum"),
        ).reset_index(drop=True)

        grouped["completion_time"] = (grouped["max_time"] - grouped["min_time"]).replace([np.inf, -np.inf], np.nan).fillna(0)
        grouped["completion_time"] = grouped["completion_time"].clip(lower=0)

        grouped["feedback_delay"] = grouped["completion_time"] / grouped["event_count"].replace(0, 1)
        grouped["retry_count"] = np.where(grouped["click_count"] > 5, ((grouped["click_count"] - 5) / 5).astype(int), 0)
        grouped["error_count"] = 0
        grouped["failed_clicks"] = np.where(
            (grouped["click_count"] > 10) & (grouped["scroll_count"] <= 1),
            grouped["click_count"] - 10,
            0,
        )
        grouped["task_completed"] = -1
        grouped["screenshot_count"] = 0
        grouped["error_message_clarity"] = -1

        grouped = grouped.drop(columns=["min_time", "max_time", "event_count"], errors="ignore")
        grouped = clean_final_dataset(grouped)

        if len(grouped) > MAX_ROWS_PER_INTERACTIONS_CHUNK:
            grouped = grouped.sample(n=MAX_ROWS_PER_INTERACTIONS_CHUNK, random_state=RANDOM_STATE + chunk_number)

        collected.append(grouped)
        collected_rows += len(grouped)

        if collected_rows >= TARGET_INTERACTIONS_ROWS:
            print(f"Reached target interactions rows: {collected_rows}")
            break

    if not collected:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    return clean_final_dataset(pd.concat(collected, ignore_index=True))


def create_balanced_sample(unique_df: pd.DataFrame) -> pd.DataFrame:
    unique_df = unique_df.copy()
    counts = unique_df["friction_level"].value_counts()
    print("\nAvailable unique rows by class:")
    print(counts)

    existing_classes = [label for label in LABEL_ORDER if label in counts.index and counts[label] > 0]

    if len(existing_classes) < 2:
        print("\nWarning: Less than two classes exist. Balance is not meaningful.")
        return unique_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    min_count = min(counts[label] for label in existing_classes)

    if all(counts.get(label, 0) >= 16_000 for label in LABEL_ORDER):
        target_per_class = 17_000
    else:
        target_per_class = max(min_count * 2, 5_000)

    sampled_parts = []

    for label in LABEL_ORDER:
        class_df = unique_df[unique_df["friction_level"] == label]

        if class_df.empty:
            continue

        # If High is limited, keep all High rows.
        if label == "High" and len(class_df) < target_per_class:
            sampled = class_df
        else:
            n = min(len(class_df), target_per_class)
            sampled = class_df.sample(n=n, random_state=RANDOM_STATE)

        sampled_parts.append(sampled)

    balanced_df = pd.concat(sampled_parts, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nBalanced/sampled class distribution:")
    print(balanced_df["friction_level"].value_counts())

    return clean_final_dataset(balanced_df)


def main():
    print("Starting Phase 2 V2 processing")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Behavioral root: {BEHAVIOURAL_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")

    datasets = [
        process_interaction_telemetry(),
        process_eshop(),
        process_interactions_large(),
    ]

    datasets = [df for df in datasets if len(df) > 0]

    if not datasets:
        raise RuntimeError("No usable datasets were processed. Check dataset paths.")

    combined = pd.concat(datasets, ignore_index=True)
    combined = clean_final_dataset(combined)
    combined = assign_friction_labels(combined)

    before_dedup = combined.copy()
    unique_df = combined.drop_duplicates().sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print_dedup_report("PHASE 2 V2 DEDUP REPORT", before_dedup, unique_df)

    unique_path = OUTPUT_DIR / "phase2_unique_ux_dataset.csv"
    unique_df.to_csv(unique_path, index=False)

    balanced_df = create_balanced_sample(unique_df)
    balanced_path = OUTPUT_DIR / "phase2_unique_balanced_ux_dataset.csv"
    balanced_df.to_csv(balanced_path, index=False)

    print("\nSaved outputs:")
    print(unique_path)
    print(balanced_path)

    print("\nFinal Phase 2 V2 summary:")
    print(f"Unique rows: {len(unique_df)}")
    print(f"Balanced/sampled rows: {len(balanced_df)}")
    print("\nBalanced/sampled source distribution:")
    print(balanced_df["source_dataset"].value_counts())


if __name__ == "__main__":
    main()