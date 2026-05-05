"""
Unit tests for sparklens.missing_duplicates.

Run with: PYTHONPATH=. pytest tests/test_missing_duplicates.py -v
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sparklens.missing_duplicates import (
    # Missing values
    detect_nulls, null_percentages, null_summary, missing_correlation,
    fill_mean, fill_median, fill_mode, fill_constant,
    fill_forward, fill_backward, fill_grouped,
    apply_fills, smart_fill,
    drop_high_null_cols, drop_high_null_rows,
    add_missing_flag,
    # Duplicates
    count_exact_duplicates, drop_exact_duplicates,
    drop_by_keys, count_by_keys,
    find_functional_duplicates, summarize_functional_duplicates,
    flag_duplicates, collapse_functional_duplicates,
    fuzzy_address_dups,
)


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder
         .master("local[2]")
         .appName("sparklens-tests")
         .getOrCreate())
    yield s
    s.stop()


# ----------------------------------------------------------------------
# MISSING VALUE TESTS
# ----------------------------------------------------------------------

@pytest.fixture
def messy_df(spark):
    data = [
        (1, "a", 10.0, "open"),
        (2, None, 20.0, "open"),
        (3, "b", None, ""),
        (4, "a", 40.0, None),
        (5, None, 50.0, "open"),
    ]
    return spark.createDataFrame(data, ["id", "label", "value", "status"])


def test_detect_nulls_counts_nulls_and_blanks(messy_df):
    counts = detect_nulls(messy_df)
    assert counts["id"] == 0
    assert counts["label"] == 2
    assert counts["value"] == 1
    assert counts["status"] == 2


def test_null_percentages_within_zero_to_hundred(messy_df):
    pct = null_percentages(messy_df)
    for p in pct.values():
        assert 0 <= p <= 100
    assert pct["label"] == 40.0


def test_null_summary_returns_sorted_dataframe(messy_df):
    summary = null_summary(messy_df).collect()
    assert summary[0]["null_percent"] >= summary[-1]["null_percent"]


def test_missing_correlation_finds_pairs(spark):
    data = [(1, None, None), (2, None, None), (3, "x", "y"), (4, None, "y")]
    df = spark.createDataFrame(data, ["id", "a", "b"])
    corr = missing_correlation(df).collect()
    pair = next(r for r in corr if {r["col_a"], r["col_b"]} == {"a", "b"})
    assert pair["co_missing"] == 2


def test_fill_mean(messy_df):
    out = fill_mean(messy_df, "value")
    assert out.filter(out["value"].isNull()).count() == 0


def test_fill_median(messy_df):
    out = fill_median(messy_df, "value")
    assert out.filter(out["value"].isNull()).count() == 0


def test_fill_mode_picks_most_frequent(messy_df):
    out = fill_mode(messy_df, "label")
    filled = out.filter(out["id"].isin([2, 5])).collect()
    for r in filled:
        assert r["label"] == "a"


def test_fill_constant(messy_df):
    out = fill_constant(messy_df, "status", "UNKNOWN")
    assert out.filter(out["status"].isNull()).count() == 0


def test_fill_handles_missing_columns_gracefully(messy_df):
    out = fill_mean(messy_df, "does_not_exist")
    assert out.count() == messy_df.count()


def test_fill_forward(spark):
    data = [(1, 10), (2, None), (3, None), (4, 40), (5, None)]
    df = spark.createDataFrame(data, ["t", "v"])
    out = fill_forward(df, "v", order_by="t").orderBy("t").collect()
    assert [r["v"] for r in out] == [10, 10, 10, 40, 40]


def test_fill_backward(spark):
    data = [(1, None), (2, 20), (3, None), (4, 40)]
    df = spark.createDataFrame(data, ["t", "v"])
    out = fill_backward(df, "v", order_by="t").orderBy("t").collect()
    assert [r["v"] for r in out] == [20, 20, 40, 40]


def test_fill_grouped_mean(spark):
    data = [
        ("A", 10.0), ("A", 20.0), ("A", None),
        ("B", 100.0), ("B", None), ("B", 200.0),
    ]
    df = spark.createDataFrame(data, ["g", "v"])
    out = fill_grouped(df, "v", group_by=["g"], strategy="mean")
    assert out.filter(out["v"].isNull()).count() == 0
    a_filled = out.filter("g = 'A' AND v = 15.0").count()
    b_filled = out.filter("g = 'B' AND v = 150.0").count()
    assert a_filled == 1 and b_filled == 1


def test_fill_grouped_mode(spark):
    data = [("A", "x"), ("A", "x"), ("A", None), ("B", "y"), ("B", None)]
    df = spark.createDataFrame(data, ["g", "v"])
    out = fill_grouped(df, "v", group_by=["g"], strategy="mode")
    assert out.filter(out["v"].isNull()).count() == 0


def test_apply_fills_handles_dict(messy_df):
    plan = {
        "value": "mean",
        "label": "mode",
        "status": ("constant", "UNKNOWN"),
    }
    out = apply_fills(messy_df, plan)
    assert out.filter(out["value"].isNull()).count() == 0
    assert out.filter(out["label"].isNull()).count() == 0
    assert out.filter(out["status"].isNull()).count() == 0


def test_apply_fills_with_grouped(spark):
    data = [("A", 10.0), ("A", None), ("B", 100.0), ("B", None)]
    df = spark.createDataFrame(data, ["g", "v"])
    plan = {"v": ("grouped", ["g"], "mean")}
    out = apply_fills(df, plan)
    assert out.filter(out["v"].isNull()).count() == 0


def test_apply_fills_rejects_unknown(messy_df):
    with pytest.raises(ValueError):
        apply_fills(messy_df, {"value": "invalid"})


def test_smart_fill(messy_df):
    out = smart_fill(messy_df, exclude=["id"])
    assert out.filter(out["value"].isNull()).count() == 0
    assert out.filter(out["label"].isNull()).count() == 0


def test_drop_high_null_cols(spark):
    data = [(1, None, "a"), (2, None, "b"), (3, None, "c"), (4, "x", "d")]
    df = spark.createDataFrame(data, ["id", "sparse", "dense"])
    out = drop_high_null_cols(df, threshold=0.5)
    assert "sparse" not in out.columns


def test_drop_high_null_rows(spark):
    data = [(1, "a", "x", "y"), (2, None, None, None), (3, "b", "x", "y")]
    df = spark.createDataFrame(data, ["id", "a", "b", "c"])
    out = drop_high_null_rows(df, threshold=0.5)
    ids = {r["id"] for r in out.collect()}
    assert 2 not in ids


def test_threshold_validates(messy_df):
    with pytest.raises(ValueError):
        drop_high_null_cols(messy_df, threshold=1.5)


def test_add_missing_flag(messy_df):
    flagged = add_missing_flag(messy_df, "value")
    assert "value_was_missing" in flagged.columns
    assert flagged.filter("value_was_missing = true").count() == 1


# ----------------------------------------------------------------------
# DUPLICATE TESTS
# ----------------------------------------------------------------------

@pytest.fixture
def df_with_dups(spark):
    data = [(1, "a", 100), (2, "b", 200), (3, "c", 300), (1, "a", 100)]
    return spark.createDataFrame(data, ["id", "label", "value"])


@pytest.fixture
def df_with_key_dups(spark):
    data = [(1, "first", 100), (2, "two", 200), (1, "second", 999)]
    return spark.createDataFrame(data, ["id", "label", "value"])


def test_count_exact_duplicates(df_with_dups):
    r = count_exact_duplicates(df_with_dups)
    assert r["total_rows"] == 4 and r["distinct_rows"] == 3 and r["duplicates"] == 1


def test_drop_exact_duplicates(df_with_dups):
    assert drop_exact_duplicates(df_with_dups).count() == 3


def test_drop_by_keys_first(df_with_key_dups):
    out = drop_by_keys(df_with_key_dups, ["id"], keep="first")
    assert out.count() == 2


def test_drop_by_keys_max(df_with_key_dups):
    out = drop_by_keys(df_with_key_dups, ["id"], keep="max:value")
    rows = {r["id"]: r["value"] for r in out.collect()}
    assert rows[1] == 999


def test_drop_by_keys_min(df_with_key_dups):
    out = drop_by_keys(df_with_key_dups, ["id"], keep="min:value")
    rows = {r["id"]: r["value"] for r in out.collect()}
    assert rows[1] == 100


def test_drop_by_keys_invalid_strategy(df_with_key_dups):
    with pytest.raises(ValueError):
        drop_by_keys(df_with_key_dups, ["id"], keep="weirdo")


def test_drop_by_keys_validates_columns(df_with_key_dups):
    with pytest.raises(ValueError):
        drop_by_keys(df_with_key_dups, ["nonexistent"])


def test_count_by_keys(df_with_key_dups):
    r = count_by_keys(df_with_key_dups, ["id"])
    assert r["total_rows"] == 3 and r["distinct_keys"] == 2


def test_find_functional_duplicates(spark):
    data = [
        ("noise", "100 main"),
        ("noise", "100 main"),
        ("noise", "200 oak"),
        ("noise", "100 main"),
    ]
    df = spark.createDataFrame(data, ["complaint", "address"])
    out = find_functional_duplicates(df, ["complaint", "address"]).collect()
    assert len(out) == 1 and out[0]["count"] == 3


def test_find_functional_duplicates_with_time_bucket(spark):
    data = [
        ("noise", "100 main", "2024-01-01 09:15:00"),
        ("noise", "100 main", "2024-01-01 09:45:00"),
        ("noise", "100 main", "2024-01-01 11:30:00"),
    ]
    df = (spark.createDataFrame(data, ["c", "a", "ts"])
                .withColumn("ts", F.to_timestamp("ts")))
    out = find_functional_duplicates(df, ["c", "a"], time_col="ts", time_bucket="hour")
    assert sorted(r["count"] for r in out.collect()) == [2]


def test_summarize_functional_duplicates(spark):
    data = [("a", "1"), ("a", "1"), ("a", "1"), ("b", "2"), ("b", "2")]
    df = spark.createDataFrame(data, ["c", "addr"])
    s = summarize_functional_duplicates(df, ["c", "addr"])
    assert s["groups_with_duplicates"] == 2
    assert s["max_group_size"] == 3
    assert s["total_redundant_records"] == 3


def test_flag_duplicates(df_with_key_dups):
    out = flag_duplicates(df_with_key_dups, ["id"], flag_name="is_dup")
    assert "is_dup" in out.columns
    assert out.filter("is_dup = true").count() == 2


def test_collapse_functional_duplicates_basic(spark):
    data = [
        ("noise", "100 main"), ("noise", "100 main"), ("noise", "100 main"),
        ("noise", "200 oak"),
    ]
    df = spark.createDataFrame(data, ["c", "addr"])
    out = collapse_functional_duplicates(df, ["c", "addr"])
    assert out.count() == 2
    rows = {r["addr"]: r["report_count"] for r in out.collect()}
    assert rows["100 main"] == 3 and rows["200 oak"] == 1


def test_collapse_with_aggregations(spark):
    data = [
        ("noise", "100 main", 10.0),
        ("noise", "100 main", 20.0),
        ("noise", "100 main", 30.0),
    ]
    df = spark.createDataFrame(data, ["c", "addr", "lat"])
    out = collapse_functional_duplicates(
        df, ["c", "addr"], aggregations={"lat": "avg"}
    ).collect()
    assert len(out) == 1
    assert out[0]["lat"] == 20.0


def test_fuzzy_address_dups(spark):
    data = [
        ("noise", "100 Main St"),
        ("noise", "100   Main   St"),
        ("noise", "100 main st"),
    ]
    df = spark.createDataFrame(data, ["c", "addr"])
    out = fuzzy_address_dups(df, "addr", other_keys=["c"])
    rows = out.collect()
    assert len(rows) == 1 and rows[0]["count"] == 3
