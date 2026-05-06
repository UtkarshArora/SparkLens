"""
Unit tests for sparklens.schema_profiling.

Run with: PYTHONPATH=. pytest tests/test_schema_profiling.py -v
"""

import json
import os

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

from sparklens.schema_profiling import (
    SchemaValidator,
    ColumnProfiler,
    QualityReport,
    load_schema_config,
)


@pytest.fixture(scope="module")
def spark():
    s = SparkSession.builder.master("local[2]").appName("sparklens-tests").getOrCreate()
    yield s
    s.stop()


@pytest.fixture
def sample_df(spark):
    schema = StructType(
        [
            StructField("id", IntegerType(), False),
            StructField("name", StringType(), True),
        ]
    )
    data = [(1, "alice"), (2, "bob"), (3, None), (4, "alice")]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def schema_config_path(tmp_path):
    config = {
        "columns": [
            {"name": "id", "type": "int", "nullable": False},
            {"name": "name", "type": "string", "nullable": True},
        ]
    }
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(config))
    return str(path)


# ----------------------------------------------------------------------
# SchemaValidator
# ----------------------------------------------------------------------


def test_schema_validator_passes_on_matching_schema(sample_df, schema_config_path):
    cfg = load_schema_config(schema_config_path)
    result = SchemaValidator(sample_df, cfg).validate()
    assert result["overall_status"] == "PASS"
    assert result["column_names"]["status"] == "PASS"
    assert result["data_type_mismatches"] == []


def test_schema_validator_flags_missing_column(sample_df):
    cfg = [
        {"name": "id", "type": "int", "nullable": False},
        {"name": "name", "type": "string", "nullable": True},
        {"name": "missing", "type": "string", "nullable": True},
    ]
    result = SchemaValidator(sample_df, cfg).validate()
    assert result["overall_status"] == "FAIL"
    assert "missing" in result["column_names"]["missing_columns"]


def test_schema_validator_flags_type_mismatch(sample_df):
    cfg = [
        {"name": "id", "type": "string", "nullable": False},  # actually int
        {"name": "name", "type": "string", "nullable": True},
    ]
    result = SchemaValidator(sample_df, cfg).validate()
    assert result["overall_status"] == "FAIL"
    assert any(m["column"] == "id" for m in result["data_type_mismatches"])


# ----------------------------------------------------------------------
# ColumnProfiler
# ----------------------------------------------------------------------


def test_column_profiler_basic_stats(sample_df):
    profiler = ColumnProfiler(sample_df)
    profiles = profiler.profile()
    profiler.unpersist()

    assert len(profiles) == 2
    by_name = {p["column"]: p for p in profiles}
    assert by_name["id"]["null_count"] == 0
    assert by_name["id"]["total_rows"] == 4
    assert by_name["name"]["null_count"] == 1
    assert by_name["name"]["distinct_count"] == 2  # "alice", "bob"


def test_column_profiler_defers_spark_actions(sample_df):
    # Constructing should not trigger a count() -- _total_rows stays None.
    profiler = ColumnProfiler(sample_df)
    assert profiler._total_rows is None
    assert profiler._is_cached is False


# ----------------------------------------------------------------------
# QualityReport
# ----------------------------------------------------------------------


def test_quality_report_runs_end_to_end(sample_df, schema_config_path):
    with QualityReport(sample_df, schema_config_path) as report:
        result = report.run()

    assert result["dataset_stats"]["total_rows"] == 4
    assert result["dataset_stats"]["total_columns"] == 2
    assert result["schema_validation"]["overall_status"] == "PASS"
    assert len(result["column_profiles"]) == 2


def test_quality_report_save_all_writes_files(sample_df, schema_config_path, tmp_path):
    with QualityReport(sample_df, schema_config_path) as report:
        report.run()
        out = str(tmp_path / "out")
        report.save_all(out)

    assert os.path.isfile(os.path.join(out, "quality_report.json"))
    assert os.path.isfile(os.path.join(out, "quality_report.html"))
    assert os.path.isdir(os.path.join(out, "column_profiles"))
