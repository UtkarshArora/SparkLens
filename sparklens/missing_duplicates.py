"""
sparklens.missing_duplicates

Functions for detecting and handling missing values and duplicates in
Spark DataFrames. Built by Debdeep Naha (dn2491).

Two views of duplicates are supported:

    Exact duplicates: every column value matches another row.
    Functional duplicates: same logical event, even if some fields differ.
        Example: two service requests at the same address for the same
        complaint within an hour are almost certainly the same incident.

Quick reference:

    Missing value detection:
        detect_nulls          dict of column -> null count
        null_percentages      dict of column -> percent
        null_summary          DataFrame summary, sorted by null pct
        missing_correlation   columns that go missing together

    Missing value fills (single column):
        fill_mean             numeric mean
        fill_median           numeric median (approximate)
        fill_mode             most frequent value
        fill_constant         user-supplied value
        fill_forward          last non-null in window
        fill_backward         next non-null in window
        fill_grouped          per-group mean or mode

    Bulk operations:
        apply_fills           apply a dict of column -> strategy
        smart_fill            auto-pick strategy based on column type

    Filtering:
        drop_high_null_cols   drop columns over threshold
        drop_high_null_rows   drop rows over threshold

    Diagnostics:
        add_missing_flag      add boolean is_null column for tracking

    Duplicate counting:
        count_exact_duplicates  total vs distinct row counts
        count_by_keys           dup count on a subset

    Duplicate removal:
        drop_exact_duplicates   remove full-row dupes
        drop_by_keys            dedup on a subset (keep first/last/max/min)

    Functional duplicates:
        find_functional_duplicates       group-by analysis with optional time bucket
        summarize_functional_duplicates  summary stats for reporting
        flag_duplicates                  add bool col instead of removing
        collapse_functional_duplicates   merge groups into one row each
        fuzzy_address_dups               normalize addresses before grouping
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .utils import (
    is_blank_or_null,
    compute_null_counts,
    get_numeric_columns,
    get_string_columns,
    logger,
)


# ======================================================================
# MISSING VALUE HANDLING
# ======================================================================

# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------

def detect_nulls(df: DataFrame) -> Dict[str, int]:
    """Count null and blank values per column. Single Spark job."""
    return compute_null_counts(df)


def null_percentages(df: DataFrame) -> Dict[str, float]:
    """Percent missing per column, rounded to 2 decimals."""
    total = df.count()
    if total == 0:
        return {c: 0.0 for c in df.columns}
    counts = detect_nulls(df)
    return {c: round(100 * n / total, 2) for c, n in counts.items()}


def null_summary(df: DataFrame) -> DataFrame:
    """Summary DataFrame with column, null_count, null_percent, sorted desc."""
    total = df.count()
    counts = detect_nulls(df)
    rows = [
        (c, counts[c], round(100 * counts[c] / total, 2) if total else 0.0)
        for c in df.columns
    ]
    return (df.sparkSession
              .createDataFrame(rows, ["column", "null_count", "null_percent"])
              .orderBy(F.desc("null_percent")))


def missing_correlation(df: DataFrame, top_n: int = 10) -> DataFrame:
    """
    Find pairs of columns that tend to be missing together.

    Returns a DataFrame with col_a, col_b, and a co_missing count
    (how many rows have both columns null at the same time).
    Useful for understanding whether sparseness has a structural cause.
    """
    cols = df.columns
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            count = df.filter(is_blank_or_null(a) & is_blank_or_null(b)).count()
            if count > 0:
                rows.append((a, b, count))

    return (df.sparkSession
              .createDataFrame(rows, ["col_a", "col_b", "co_missing"])
              .orderBy(F.desc("co_missing"))
              .limit(top_n))


# ----------------------------------------------------------------------
# Single-column fill strategies
# ----------------------------------------------------------------------

def fill_mean(df: DataFrame, col: str) -> DataFrame:
    """Fill nulls with the column mean. No-op if column missing or all-null."""
    if col not in df.columns:
        logger.warning("fill_mean: column %r not found, skipping", col)
        return df
    val = df.select(F.mean(F.col(col))).first()[0]
    return df.fillna({col: val}) if val is not None else df


def fill_median(df: DataFrame, col: str, accuracy: float = 0.01) -> DataFrame:
    """Fill nulls with the column median (approximate)."""
    if col not in df.columns:
        logger.warning("fill_median: column %r not found, skipping", col)
        return df
    quantile = df.approxQuantile(col, [0.5], accuracy)
    return df.fillna({col: quantile[0]}) if quantile else df


def fill_mode(df: DataFrame, col: str) -> DataFrame:
    """Fill nulls with the most frequent non-null value."""
    if col not in df.columns:
        logger.warning("fill_mode: column %r not found, skipping", col)
        return df
    mode_row = (df.filter(F.col(col).isNotNull())
                  .groupBy(col).count()
                  .orderBy(F.desc("count"))
                  .first())
    return df.fillna({col: mode_row[0]}) if mode_row else df


def fill_constant(df: DataFrame, col: str, value: Union[str, int, float]) -> DataFrame:
    """Fill nulls with a user-supplied constant."""
    if col not in df.columns:
        logger.warning("fill_constant: column %r not found, skipping", col)
        return df
    return df.fillna({col: value})


# ----------------------------------------------------------------------
# Window-based fills
# ----------------------------------------------------------------------

def fill_forward(df: DataFrame, col: str, order_by: str,
                 partition_by: Optional[List[str]] = None) -> DataFrame:
    """Forward-fill: replace null with the most recent non-null value."""
    if col not in df.columns:
        logger.warning("fill_forward: column %r not found, skipping", col)
        return df

    window = Window.orderBy(order_by).rowsBetween(Window.unboundedPreceding, 0)
    if partition_by:
        window = window.partitionBy(*partition_by)
    return df.withColumn(col, F.last(F.col(col), ignorenulls=True).over(window))


def fill_backward(df: DataFrame, col: str, order_by: str,
                  partition_by: Optional[List[str]] = None) -> DataFrame:
    """Backward-fill: replace null with the next non-null value."""
    if col not in df.columns:
        logger.warning("fill_backward: column %r not found, skipping", col)
        return df

    window = Window.orderBy(order_by).rowsBetween(0, Window.unboundedFollowing)
    if partition_by:
        window = window.partitionBy(*partition_by)
    return df.withColumn(col, F.first(F.col(col), ignorenulls=True).over(window))


# ----------------------------------------------------------------------
# Group-based fills
# ----------------------------------------------------------------------

def fill_grouped(df: DataFrame, col: str, group_by: List[str],
                 strategy: str = "mean") -> DataFrame:
    """
    Fill nulls using group-level statistics.

    Far more accurate than a global fill when the column has natural
    groupings. Example: fill missing fares grouped by complaint_type.

    strategy: 'mean' (numeric) or 'mode' (categorical)
    """
    if col not in df.columns:
        logger.warning("fill_grouped: column %r not found, skipping", col)
        return df

    if strategy == "mean":
        if col not in get_numeric_columns(df):
            raise ValueError(f"fill_grouped(mean) requires numeric column, got {col}")
        agg = df.groupBy(*group_by).agg(F.mean(col).alias("_fill"))
    elif strategy == "mode":
        w = Window.partitionBy(*group_by).orderBy(F.desc("_n"))
        ranked = (df.filter(F.col(col).isNotNull())
                    .groupBy(*group_by, col).count().withColumnRenamed("count", "_n")
                    .withColumn("_rank", F.row_number().over(w))
                    .filter(F.col("_rank") == 1)
                    .select(*group_by, F.col(col).alias("_fill")))
        agg = ranked
    else:
        raise ValueError(f"strategy must be 'mean' or 'mode', got {strategy!r}")

    joined = df.join(agg, on=group_by, how="left")
    return (joined
            .withColumn(col, F.coalesce(F.col(col), F.col("_fill")))
            .drop("_fill"))


# ----------------------------------------------------------------------
# Bulk operations
# ----------------------------------------------------------------------

def apply_fills(df: DataFrame, plan: Dict[str, Union[str, tuple]]) -> DataFrame:
    """
    Apply a dict of column -> fill strategy in one call.

    Strategy can be:
        'mean', 'median', 'mode'
        ('constant', value)
        ('grouped', group_by_list, 'mean' or 'mode')

    Example:
        plan = {
            "latitude": "mean",
            "longitude": "median",
            "borough": "mode",
            "resolution_description": ("constant", "NOT_PROVIDED"),
            "fare_amount": ("grouped", ["complaint_type"], "mean"),
        }
    """
    out = df
    for col, strategy in plan.items():
        if strategy == "mean":
            out = fill_mean(out, col)
        elif strategy == "median":
            out = fill_median(out, col)
        elif strategy == "mode":
            out = fill_mode(out, col)
        elif isinstance(strategy, tuple):
            kind = strategy[0]
            if kind == "constant":
                out = fill_constant(out, col, strategy[1])
            elif kind == "grouped":
                _, group_by, sub_strategy = strategy
                out = fill_grouped(out, col, group_by, sub_strategy)
            else:
                raise ValueError(f"Unknown tuple strategy {strategy!r}")
        else:
            raise ValueError(f"Unknown strategy {strategy!r} for column {col!r}.")
    return out


def smart_fill(df: DataFrame, exclude: Optional[List[str]] = None) -> DataFrame:
    """
    Pick a fill strategy automatically based on column type.

    Numeric columns get the median (robust to outliers).
    String columns get the mode.
    """
    exclude = set(exclude or [])
    out = df
    for col in get_numeric_columns(df):
        if col not in exclude:
            out = fill_median(out, col)
    for col in get_string_columns(df):
        if col not in exclude:
            out = fill_mode(out, col)
    return out


# ----------------------------------------------------------------------
# Threshold-based filters
# ----------------------------------------------------------------------

def drop_high_null_cols(df: DataFrame, threshold: float = 0.5) -> DataFrame:
    """Drop columns whose null fraction exceeds threshold (in [0, 1])."""
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    total = df.count()
    if total == 0:
        return df
    counts = compute_null_counts(df)
    keep = [c for c, n in counts.items() if (n / total) <= threshold]
    dropped = set(df.columns) - set(keep)
    if dropped:
        logger.info("Dropping %d sparse columns: %s", len(dropped), sorted(dropped))
    return df.select(*keep)


def drop_high_null_rows(df: DataFrame, threshold: float = 0.5) -> DataFrame:
    """Drop rows where the fraction of null fields exceeds threshold."""
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    cols = df.columns
    if not cols:
        return df
    null_count_expr = sum(F.when(F.col(c).isNull(), 1).otherwise(0) for c in cols)
    return (df.withColumn("_null_count", null_count_expr)
              .filter(F.col("_null_count") / F.lit(len(cols)) <= threshold)
              .drop("_null_count"))


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

def add_missing_flag(df: DataFrame, col: str, flag_name: Optional[str] = None) -> DataFrame:
    """
    Add a boolean column tracking whether `col` was missing BEFORE any fill.
    Lets downstream consumers distinguish real values from imputed ones.
    """
    if col not in df.columns:
        logger.warning("add_missing_flag: column %r not found, skipping", col)
        return df
    flag_name = flag_name or f"{col}_was_missing"
    return df.withColumn(flag_name, is_blank_or_null(col))


# ======================================================================
# DUPLICATE HANDLING
# ======================================================================

# ----------------------------------------------------------------------
# Exact duplicates
# ----------------------------------------------------------------------

def count_exact_duplicates(df: DataFrame) -> dict:
    """Count exact full-row duplicates."""
    total = df.count()
    distinct = df.distinct().count()
    return {
        "total_rows": total,
        "distinct_rows": distinct,
        "duplicates": total - distinct,
    }


def drop_exact_duplicates(df: DataFrame) -> DataFrame:
    """Remove rows that exactly match another row across all columns."""
    return df.distinct()


# ----------------------------------------------------------------------
# Key-based deduplication
# ----------------------------------------------------------------------

def drop_by_keys(df: DataFrame, keys: List[str], keep: str = "first") -> DataFrame:
    """
    Drop duplicates based on key columns, with control over which row to keep.

    keep:
        'first'      -- arbitrary first occurrence (Spark default)
        'last'       -- arbitrary last occurrence
        'max:col'    -- row with max value of col
        'min:col'    -- row with min value of col

    Example:
        drop_by_keys(df, ["user_id"], keep="max:updated_at")
    """
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    if keep == "first":
        return df.dropDuplicates(keys)

    if keep == "last":
        w = Window.partitionBy(*keys).orderBy(F.monotonically_increasing_id().desc())
        return (df.withColumn("_rn", F.row_number().over(w))
                  .filter(F.col("_rn") == 1)
                  .drop("_rn"))

    if ":" in keep:
        direction, target_col = keep.split(":", 1)
        if target_col not in df.columns:
            raise ValueError(f"keep='{keep}' references missing column {target_col!r}")
        order_col = F.col(target_col).desc() if direction == "max" else F.col(target_col).asc()
        w = Window.partitionBy(*keys).orderBy(order_col)
        return (df.withColumn("_rn", F.row_number().over(w))
                  .filter(F.col("_rn") == 1)
                  .drop("_rn"))

    raise ValueError(
        f"Unknown keep strategy {keep!r}. Use 'first', 'last', 'max:col', or 'min:col'."
    )


def count_by_keys(df: DataFrame, keys: List[str]) -> dict:
    """Count rows before and after dedup on a key set."""
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    total = df.count()
    distinct = df.select(*keys).distinct().count()
    return {
        "total_rows": total,
        "distinct_keys": distinct,
        "duplicates": total - distinct,
    }


# ----------------------------------------------------------------------
# Functional duplicate detection
# ----------------------------------------------------------------------

def find_functional_duplicates(
    df: DataFrame,
    group_by_cols: List[str],
    time_col: Optional[str] = None,
    time_bucket: str = "hour",
) -> DataFrame:
    """
    Surface groups of records that look like duplicates of the same event.

    Returns a DataFrame with one row per group of size > 1, sorted by
    descending count.
    """
    missing = [c for c in group_by_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    grouped = df
    bucket_col = None

    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"Time column not found: {time_col}")
        bucket_col = f"{time_col}_{time_bucket}"
        grouped = grouped.withColumn(bucket_col, F.date_trunc(time_bucket, F.col(time_col)))

    full_keys = group_by_cols + ([bucket_col] if bucket_col else [])
    return (grouped.groupBy(*full_keys)
                   .count()
                   .filter(F.col("count") > 1)
                   .orderBy(F.desc("count")))


def summarize_functional_duplicates(
    df: DataFrame,
    group_by_cols: List[str],
    time_col: Optional[str] = None,
    time_bucket: str = "hour",
) -> dict:
    """Summary stats for functional duplicates."""
    func_dups = find_functional_duplicates(df, group_by_cols, time_col, time_bucket)
    summary = func_dups.agg(
        F.count("*").alias("groups"),
        F.max("count").alias("max_size"),
        F.sum(F.col("count") - 1).alias("redundant"),
    ).first()
    return {
        "groups_with_duplicates": int(summary["groups"] or 0),
        "max_group_size": int(summary["max_size"] or 0),
        "total_redundant_records": int(summary["redundant"] or 0),
    }


def flag_duplicates(
    df: DataFrame,
    keys: List[str],
    flag_name: str = "is_duplicate",
) -> DataFrame:
    """Tag duplicates with a boolean column instead of removing them."""
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    w = Window.partitionBy(*keys)
    return df.withColumn(flag_name, F.count("*").over(w) > 1)


def collapse_functional_duplicates(
    df: DataFrame,
    group_by_cols: List[str],
    time_col: Optional[str] = None,
    time_bucket: str = "hour",
    aggregations: Optional[dict] = None,
) -> DataFrame:
    """
    Merge functional duplicate groups into one row each.

    Each group becomes one row with a `report_count` column. Customize
    other column merging via aggregations dict:

        aggregations = {
            "descriptor": "first",
            "latitude": "avg",
            "status": "max",
        }
    """
    missing = [c for c in group_by_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    grouped = df
    bucket_col = None
    if time_col is not None:
        bucket_col = f"{time_col}_{time_bucket}"
        grouped = grouped.withColumn(bucket_col, F.date_trunc(time_bucket, F.col(time_col)))

    keys = group_by_cols + ([bucket_col] if bucket_col else [])

    agg_exprs = [F.count("*").alias("report_count")]
    aggregations = aggregations or {}
    for col, how in aggregations.items():
        if col not in df.columns:
            continue
        if how == "first":
            agg_exprs.append(F.first(F.col(col), ignorenulls=True).alias(col))
        elif how == "avg":
            agg_exprs.append(F.avg(F.col(col)).alias(col))
        elif how == "max":
            agg_exprs.append(F.max(F.col(col)).alias(col))
        elif how == "min":
            agg_exprs.append(F.min(F.col(col)).alias(col))
        elif how == "collect":
            agg_exprs.append(F.collect_list(F.col(col)).alias(col))
        else:
            raise ValueError(f"Unknown aggregation {how!r} for column {col!r}")

    return grouped.groupBy(*keys).agg(*agg_exprs)


def fuzzy_address_dups(
    df: DataFrame,
    address_col: str,
    other_keys: List[str],
    time_col: Optional[str] = None,
    time_bucket: str = "hour",
) -> DataFrame:
    """
    Find functional duplicates with normalized addresses.

    "100 Main St", "100 main st.", "100  MAIN  STREET" all collapse to
    a canonical form before grouping.
    """
    if address_col not in df.columns:
        raise ValueError(f"Address column not found: {address_col}")

    normalized = df.withColumn(
        f"{address_col}_norm",
        F.trim(F.regexp_replace(F.upper(F.col(address_col)), r"\s+", " "))
    )

    return find_functional_duplicates(
        normalized,
        group_by_cols=other_keys + [f"{address_col}_norm"],
        time_col=time_col,
        time_bucket=time_bucket,
    )
