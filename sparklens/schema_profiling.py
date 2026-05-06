"""
sparklens.schema_profiling

Schema validation, column profiling, and quality reporting. Built by Utkarsh.

Three pieces:

    SchemaValidator   validate column names, types, and nullability against a JSON config
    ColumnProfiler    compute per-column statistics in a single Spark pass
    QualityReport     orchestrator that combines both and exports JSON / CSV / HTML

Quick reference:

    Validation:
        SchemaValidator(df, expected_schema).validate()

    Profiling:
        profiler = ColumnProfiler(df)
        profiler.profile()
        profiler.unpersist()

    Combined report:
        with QualityReport(df, schema_config_path) as report:
            report.run()
            report.save_all(output_dir)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from html import escape
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    NumericType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from .utils import logger, resolve_type

# ======================================================================
# CONFIG LOADING
# ======================================================================


def load_schema_config(path: str) -> List[Dict[str, Any]]:
    """
    Load an expected-schema JSON config. Format:

        {"columns": [{"name": "...", "type": "...", "nullable": true}, ...]}
    """
    with open(path, "r") as f:
        config = json.load(f)
    return config.get("columns", config)


# ======================================================================
# SCHEMA VALIDATION
# ======================================================================


class SchemaValidator:
    """Validate a DataFrame against an expected-schema spec."""

    def __init__(self, df: DataFrame, expected_schema: List[Dict[str, Any]]):
        self.df = df
        self.expected = expected_schema
        self._actual = {f.name: f for f in df.schema.fields}
        self._results: Optional[Dict[str, Any]] = None

    def validate_column_names(self) -> Dict[str, Any]:
        expected_names = {col["name"] for col in self.expected}
        actual_names = set(self._actual.keys())
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        status = "PASS" if (not missing and not extra) else "FAIL"
        logger.info(
            "schema_profiling: column-name check %s (missing=%d, extra=%d)",
            status,
            len(missing),
            len(extra),
        )
        return {
            "status": status,
            "expected_count": len(expected_names),
            "actual_count": len(actual_names),
            "missing_columns": missing,
            "extra_columns": extra,
        }

    def validate_data_types(self) -> List[Dict[str, str]]:
        mismatches = []
        for col_spec in self.expected:
            name = col_spec["name"]
            if name not in self._actual:
                continue  # already flagged by column-name check
            expected_cls = resolve_type(col_spec["type"])
            if expected_cls is None:
                logger.warning(
                    "schema_profiling: unknown type %r for column %r, skipping",
                    col_spec["type"],
                    name,
                )
                continue
            actual_type = self._actual[name].dataType
            if not isinstance(actual_type, expected_cls):
                mismatches.append(
                    {
                        "column": name,
                        "expected_type": col_spec["type"],
                        "actual_type": str(actual_type),
                    }
                )
        status = "PASS" if not mismatches else "FAIL"
        logger.info(
            "schema_profiling: data-type check %s (%d mismatches)",
            status,
            len(mismatches),
        )
        return mismatches

    def validate_nullable(
        self, total_rows: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        non_nullable_cols = [
            col_spec["name"]
            for col_spec in self.expected
            if not col_spec.get("nullable", True) and col_spec["name"] in self._actual
        ]
        if not non_nullable_cols:
            logger.info("schema_profiling: nullable check PASS (no constraints)")
            return []

        agg_exprs = [
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in non_nullable_cols
        ]
        null_counts = self.df.agg(*agg_exprs).collect()[0]
        if total_rows is None:
            total_rows = self.df.count()

        violations = []
        for col_name in non_nullable_cols:
            count = null_counts[col_name] or 0
            if count > 0:
                violations.append(
                    {
                        "column": col_name,
                        "null_count": int(count),
                        "null_pct": round(count / total_rows * 100, 2),
                    }
                )
        status = "PASS" if not violations else "FAIL"
        logger.info(
            "schema_profiling: nullable check %s (%d violations)",
            status,
            len(violations),
        )
        return violations

    def validate(self, total_rows: Optional[int] = None) -> Dict[str, Any]:
        start = time.time()
        self._results = {
            "column_names": self.validate_column_names(),
            "data_type_mismatches": self.validate_data_types(),
            "nullable_violations": self.validate_nullable(total_rows),
        }
        name_ok = self._results["column_names"]["status"] == "PASS"
        type_ok = len(self._results["data_type_mismatches"]) == 0
        null_ok = len(self._results["nullable_violations"]) == 0
        self._results["overall_status"] = (
            "PASS" if (name_ok and type_ok and null_ok) else "FAIL"
        )
        logger.info(
            "schema_profiling: validate done in %.2fs · status=%s",
            time.time() - start,
            self._results["overall_status"],
        )
        return self._results

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        return self._results


# ======================================================================
# COLUMN PROFILING
# ======================================================================


class ColumnProfiler:
    """Per-column statistics with deferred caching."""

    def __init__(
        self, df: DataFrame, *, approx_distinct: bool = True, cache: bool = True
    ):
        self.df = df
        self.approx_distinct = approx_distinct
        self._cache_enabled = cache
        self._is_cached = False
        self._profiles: Optional[List[Dict[str, Any]]] = None
        self._total_rows: Optional[int] = None

        # No Spark actions here -- defer cache + count to .profile() / .total_rows.
        self._numeric_fields = [
            f for f in df.schema.fields if isinstance(f.dataType, NumericType)
        ]
        self._string_fields = [
            f for f in df.schema.fields if isinstance(f.dataType, StringType)
        ]
        self._date_fields = [
            f
            for f in df.schema.fields
            if isinstance(f.dataType, (DateType, TimestampType))
        ]

        logger.info(
            "schema_profiling: %d cols (%d numeric, %d string, %d date/ts)",
            len(df.schema.fields),
            len(self._numeric_fields),
            len(self._string_fields),
            len(self._date_fields),
        )

    def _ensure_cached(self) -> None:
        if self._cache_enabled and not self._is_cached:
            logger.info("schema_profiling: caching DataFrame for profiling")
            self.df.cache()
            self._is_cached = True

    @property
    def total_rows(self) -> int:
        if self._total_rows is None:
            self._ensure_cached()
            self._total_rows = self.df.count()
            logger.info("schema_profiling: total rows = %d", self._total_rows)
        return self._total_rows

    def unpersist(self) -> None:
        """Release the cached DataFrame. Safe to call multiple times."""
        if self._is_cached:
            self.df.unpersist()
            self._is_cached = False
            logger.info("schema_profiling: DataFrame unpersisted")

    def _build_universal_agg_exprs(self) -> list:
        exprs = []
        distinct_fn = (
            F.approx_count_distinct if self.approx_distinct else F.countDistinct
        )
        for field in self.df.schema.fields:
            c = field.name
            exprs.append(
                F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"__null__{c}")
            )
            exprs.append(distinct_fn(F.col(c)).alias(f"__dist__{c}"))
        return exprs

    def _build_numeric_agg_exprs(self) -> list:
        exprs = []
        for field in self._numeric_fields:
            c = field.name
            exprs.extend(
                [
                    F.min(F.col(c)).alias(f"__nmin__{c}"),
                    F.max(F.col(c)).alias(f"__nmax__{c}"),
                    F.mean(F.col(c)).alias(f"__mean__{c}"),
                    F.stddev(F.col(c)).alias(f"__std__{c}"),
                ]
            )
        return exprs

    def _build_string_agg_exprs(self) -> list:
        exprs = []
        for field in self._string_fields:
            c = field.name
            length_col = F.length(F.col(c))
            exprs.extend(
                [
                    F.min(length_col).alias(f"__smin__{c}"),
                    F.max(length_col).alias(f"__smax__{c}"),
                    F.avg(length_col).alias(f"__savg__{c}"),
                ]
            )
        return exprs

    def _build_date_agg_exprs(self) -> list:
        exprs = []
        for field in self._date_fields:
            c = field.name
            exprs.extend(
                [
                    F.min(F.col(c)).alias(f"__dmin__{c}"),
                    F.max(F.col(c)).alias(f"__dmax__{c}"),
                ]
            )
        return exprs

    def _compute_percentiles(self) -> Dict[str, Dict[str, Optional[float]]]:
        if not self._numeric_fields:
            return {}
        exprs = [
            F.percentile_approx(F.col(field.name), [0.25, 0.5, 0.75]).alias(
                f"__pct__{field.name}"
            )
            for field in self._numeric_fields
        ]
        row = self.df.agg(*exprs).collect()[0]
        result = {}
        for field in self._numeric_fields:
            pcts = row[f"__pct__{field.name}"]
            if pcts and len(pcts) == 3:
                result[field.name] = {"p25": pcts[0], "median": pcts[1], "p75": pcts[2]}
            else:
                result[field.name] = {"p25": None, "median": None, "p75": None}
        return result

    def _compute_top_values(self, n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        result = {}
        for field in self._string_fields:
            c = field.name
            top = (
                self.df.filter(F.col(c).isNotNull())
                .groupBy(c)
                .count()
                .orderBy(F.desc("count"))
                .limit(n)
                .collect()
            )
            result[c] = [{"value": row[c], "count": row["count"]} for row in top]
        return result

    def profile(self, top_n: int = 5) -> List[Dict[str, Any]]:
        start = time.time()
        total_rows = self.total_rows  # triggers cache + count on first access

        all_exprs = (
            self._build_universal_agg_exprs()
            + self._build_numeric_agg_exprs()
            + self._build_string_agg_exprs()
            + self._build_date_agg_exprs()
        )
        logger.info(
            "schema_profiling: single-pass aggregation (%d exprs)", len(all_exprs)
        )
        agg_row = self.df.agg(*all_exprs).collect()[0]

        percentiles = self._compute_percentiles()
        top_values = self._compute_top_values(n=top_n)

        profiles = []
        for field in self.df.schema.fields:
            c = field.name
            null_count = int(agg_row[f"__null__{c}"] or 0)
            distinct_count = int(agg_row[f"__dist__{c}"] or 0)
            null_pct = (
                round(null_count / total_rows * 100, 2) if total_rows > 0 else 0.0
            )
            quality_score = round(max(0.0, 100.0 - null_pct), 2)

            profile = {
                "column": c,
                "data_type": str(field.dataType),
                "nullable": field.nullable,
                "total_rows": total_rows,
                "null_count": null_count,
                "null_pct": null_pct,
                "distinct_count": distinct_count,
                "quality_score": quality_score,
            }

            if isinstance(field.dataType, NumericType):
                profile["min"] = self._safe_round(
                    self._row_get(agg_row, f"__nmin__{c}")
                )
                profile["max"] = self._safe_round(
                    self._row_get(agg_row, f"__nmax__{c}")
                )
                profile["mean"] = self._safe_round(
                    self._row_get(agg_row, f"__mean__{c}")
                )
                profile["stddev"] = self._safe_round(
                    self._row_get(agg_row, f"__std__{c}")
                )
                pct = percentiles.get(c, {})
                profile["p25"] = self._safe_round(pct.get("p25"))
                profile["median"] = self._safe_round(pct.get("median"))
                profile["p75"] = self._safe_round(pct.get("p75"))
            elif isinstance(field.dataType, StringType):
                profile["min_length"] = self._row_get(agg_row, f"__smin__{c}")
                profile["max_length"] = self._row_get(agg_row, f"__smax__{c}")
                profile["avg_length"] = self._safe_round(
                    self._row_get(agg_row, f"__savg__{c}")
                )
                profile["top_values"] = top_values.get(c, [])
            elif isinstance(field.dataType, (DateType, TimestampType)):
                dmin = self._row_get(agg_row, f"__dmin__{c}")
                dmax = self._row_get(agg_row, f"__dmax__{c}")
                profile["min_date"] = str(dmin) if dmin else None
                profile["max_date"] = str(dmax) if dmax else None
                if dmin and dmax:
                    profile["range_days"] = (dmax - dmin).days

            profiles.append(profile)

        self._profiles = profiles
        logger.info("schema_profiling: profile done in %.2fs", time.time() - start)
        return profiles

    def to_spark_dataframe(self) -> DataFrame:
        if self._profiles is None:
            raise RuntimeError("Run .profile() first")

        schema = StructType(
            [
                StructField("column", StringType(), False),
                StructField("data_type", StringType(), False),
                StructField("nullable", BooleanType(), True),
                StructField("total_rows", IntegerType(), True),
                StructField("null_count", IntegerType(), True),
                StructField("null_pct", DoubleType(), True),
                StructField("distinct_count", IntegerType(), True),
                StructField("quality_score", DoubleType(), True),
                StructField("min", DoubleType(), True),
                StructField("max", DoubleType(), True),
                StructField("mean", DoubleType(), True),
                StructField("stddev", DoubleType(), True),
                StructField("p25", DoubleType(), True),
                StructField("median", DoubleType(), True),
                StructField("p75", DoubleType(), True),
                StructField("min_length", IntegerType(), True),
                StructField("max_length", IntegerType(), True),
                StructField("avg_length", DoubleType(), True),
                StructField("min_date", StringType(), True),
                StructField("max_date", StringType(), True),
                StructField("range_days", IntegerType(), True),
            ]
        )

        rows = [
            tuple(self._coerce(p.get(f.name), f.dataType) for f in schema.fields)
            for p in self._profiles
        ]
        return self.df.sparkSession.createDataFrame(rows, schema=schema)

    @property
    def profiles(self) -> Optional[List[Dict[str, Any]]]:
        return self._profiles

    @staticmethod
    def _coerce(value, target_type):
        """Coerce a profile value to match its target Spark schema type.
        Returns None on any failed conversion so createDataFrame can't blow up
        on edge cases like Decimals or datetimes landing in a Double slot."""
        if value is None:
            return None
        if isinstance(target_type, StringType):
            return value if isinstance(value, str) else str(value)
        if isinstance(target_type, BooleanType):
            return bool(value)
        if isinstance(target_type, IntegerType):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        if isinstance(target_type, DoubleType):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return value

    @staticmethod
    def _safe_round(val, decimals: int = 4):
        if val is None:
            return None
        try:
            return round(float(val), decimals)
        except (TypeError, ValueError):
            return val

    @staticmethod
    def _row_get(row, key, default=None):
        try:
            return row[key]
        except (KeyError, ValueError, IndexError):
            return default


# ======================================================================
# QUALITY REPORT
# ======================================================================


class QualityReport:
    """Orchestrates validation + profiling and exports reports."""

    def __init__(
        self, df: DataFrame, schema_config_path: str, *, approx_distinct: bool = True
    ):
        self.df = df
        self.expected_schema = load_schema_config(schema_config_path)
        self.validator = SchemaValidator(df, self.expected_schema)
        self.profiler = ColumnProfiler(df, approx_distinct=approx_distinct)
        self.validation_results: Optional[Dict[str, Any]] = None
        self.column_profiles: Optional[List[Dict[str, Any]]] = None
        self._report: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        start = time.time()
        # Profile first so total_rows is computed once and reused by the validator.
        self.column_profiles = self.profiler.profile()
        total_rows = self.profiler.total_rows
        self.validation_results = self.validator.validate(total_rows=total_rows)

        scores = [p["quality_score"] for p in self.column_profiles]
        overall_score = round(sum(scores) / len(scores), 2) if scores else 0.0

        self._report = {
            "generated_at": datetime.now().isoformat(),
            "dataset_stats": {
                "total_rows": total_rows,
                "total_columns": len(self.df.schema.fields),
                "overall_quality_score": overall_score,
            },
            "schema_validation": self.validation_results,
            "column_profiles": self.column_profiles,
        }
        logger.info("schema_profiling: full report done in %.2fs", time.time() - start)
        return self._report

    def close(self) -> None:
        """Release any cached Spark resources held by the underlying profiler."""
        self.profiler.unpersist()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def save_json(self, output_dir: str, filename: str = "quality_report.json"):
        if self._report is None:
            raise RuntimeError("Call .run() first")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            json.dump(self._report, f, indent=2, default=str)
        logger.info("schema_profiling: JSON report saved -> %s", path)

    def save_csv(self, output_dir: str, filename: str = "column_profiles"):
        if self.column_profiles is None:
            raise RuntimeError("Call .run() first")
        os.makedirs(output_dir, exist_ok=True)
        profile_df = self.profiler.to_spark_dataframe()
        path = os.path.join(output_dir, filename)
        profile_df.coalesce(1).write.csv(path, header=True, mode="overwrite")
        logger.info("schema_profiling: CSV profiles saved -> %s/", path)

    def save_html(self, output_dir: str, filename: str = "quality_report.html"):
        if self._report is None:
            raise RuntimeError("Call .run() first")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            f.write(self._generate_html())
        logger.info("schema_profiling: HTML report saved -> %s", path)

    def save_all(self, output_dir: str):
        self.save_json(output_dir)
        self.save_csv(output_dir)
        self.save_html(output_dir)

    def _generate_html(self) -> str:
        r = self._report
        ds = r["dataset_stats"]
        sv = r["schema_validation"]
        profiles = r["column_profiles"]

        overall_status = sv["overall_status"]
        status_color = "#22c55e" if overall_status == "PASS" else "#ef4444"

        profile_rows = ""
        for p in profiles:
            score = p["quality_score"]
            bar_color = (
                "#22c55e" if score >= 90 else "#f59e0b" if score >= 70 else "#ef4444"
            )
            profile_rows += f"""
            <tr>
                <td><code>{escape(str(p['column']))}</code></td>
                <td>{escape(str(p['data_type']))}</td>
                <td>{p['total_rows']:,}</td>
                <td>{p['null_count']:,}</td>
                <td>{p['null_pct']}%</td>
                <td>{p['distinct_count']:,}</td>
                <td>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="width:80px;height:12px;background:#e5e7eb;border-radius:6px;overflow:hidden;">
                            <div style="width:{score}%;height:100%;background:{bar_color};border-radius:6px;"></div>
                        </div>
                        <span>{score}</span>
                    </div>
                </td>
            </tr>"""

        issues_html = ""
        if sv["data_type_mismatches"]:
            issues_html += "<h3>Data Type Mismatches</h3><ul>"
            for m in sv["data_type_mismatches"]:
                issues_html += (
                    f"<li><code>{escape(str(m['column']))}</code>: "
                    f"expected <b>{escape(str(m['expected_type']))}</b>, "
                    f"got <b>{escape(str(m['actual_type']))}</b></li>"
                )
            issues_html += "</ul>"
        if sv["nullable_violations"]:
            issues_html += "<h3>Nullable Violations</h3><ul>"
            for v in sv["nullable_violations"]:
                issues_html += (
                    f"<li><code>{escape(str(v['column']))}</code>: "
                    f"{v['null_count']:,} nulls ({v['null_pct']}%)</li>"
                )
            issues_html += "</ul>"
        missing = sv["column_names"].get("missing_columns", [])
        extra = sv["column_names"].get("extra_columns", [])
        if missing:
            issues_html += (
                "<h3>Missing Columns</h3><p>"
                + ", ".join(f"<code>{escape(str(c))}</code>" for c in missing)
                + "</p>"
            )
        if extra:
            issues_html += (
                "<h3>Extra Columns</h3><p>"
                + ", ".join(f"<code>{escape(str(c))}</code>" for c in extra)
                + "</p>"
            )
        if not issues_html:
            issues_html = "<p style='color:#22c55e;'>All schema checks passed.</p>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SparkLens Quality Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f8fafc; color: #1e293b; padding: 32px; line-height: 1.6; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    h1 {{ font-size: 28px; margin-bottom: 4px; }}
    h2 {{ font-size: 20px; margin: 32px 0 16px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
    h3 {{ font-size: 16px; margin: 16px 0 8px; color: #475569; }}
    .subtitle {{ color: #64748b; margin-bottom: 24px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }}
    .card {{ background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .card .label {{ font-size: 13px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
    .card .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px;
             overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    th {{ background: #f1f5f9; text-align: left; padding: 12px 16px; font-size: 13px;
          text-transform: uppercase; letter-spacing: 0.5px; color: #475569; }}
    td {{ padding: 10px 16px; border-top: 1px solid #f1f5f9; font-size: 14px; }}
    tr:hover {{ background: #f8fafc; }}
    code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
    .issues {{ background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    ul {{ margin-left: 20px; }}
    li {{ margin: 4px 0; }}
</style>
</head>
<body>
<div class="container">
    <h1>SparkLens Quality Report</h1>
    <p class="subtitle">Generated on {escape(r['generated_at'])}</p>

    <div class="cards">
        <div class="card"><div class="label">Total Rows</div><div class="value">{ds['total_rows']:,}</div></div>
        <div class="card"><div class="label">Total Columns</div><div class="value">{ds['total_columns']}</div></div>
        <div class="card"><div class="label">Quality Score</div><div class="value">{ds['overall_quality_score']}</div></div>
        <div class="card"><div class="label">Schema Status</div><div class="value" style="color:{status_color};">{escape(overall_status)}</div></div>
    </div>

    <h2>Schema Validation</h2>
    <div class="issues">{issues_html}</div>

    <h2>Column Profiles</h2>
    <table>
        <thead>
            <tr><th>Column</th><th>Type</th><th>Rows</th><th>Nulls</th><th>Null %</th><th>Distinct</th><th>Quality</th></tr>
        </thead>
        <tbody>{profile_rows}</tbody>
    </table>
</div>
</body>
</html>"""
