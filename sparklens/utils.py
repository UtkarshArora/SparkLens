"""
sparklens.utils

Shared helper functions and audit infrastructure used across the library.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------

logger = logging.getLogger("sparklens")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[sparklens] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ----------------------------------------------------------------------
# Missing-value detection helper
# ----------------------------------------------------------------------

def is_blank_or_null(col_name: str):
    """
    Build a Column expression that evaluates to True when a value is null
    or blank. Treats nulls, empty strings, whitespace, and common placeholders
    like 'N/A' or 'NA' as missing because real-world datasets are messy.
    """
    col = F.col(col_name)
    cleaned = F.upper(F.trim(col.cast("string")))
    return col.isNull() | (cleaned == "") | cleaned.isin("N/A", "NA", "NULL", "NONE")


def get_numeric_columns(df: DataFrame) -> List[str]:
    """Return names of all columns whose Spark type is numeric."""
    numeric_types = {"int", "bigint", "double", "float", "decimal", "long", "short"}
    return [
        f.name for f in df.schema.fields
        if any(t in f.dataType.simpleString().lower() for t in numeric_types)
    ]


def get_string_columns(df: DataFrame) -> List[str]:
    """Return names of all columns whose Spark type is string."""
    return [f.name for f in df.schema.fields if f.dataType.simpleString() == "string"]


def row_count(df: DataFrame) -> int:
    """Tiny wrapper so call sites read more naturally."""
    return df.count()


def compute_null_counts(df: DataFrame) -> Dict[str, int]:
    """
    Count nulls/blanks for every column in a single Spark job.
    Way faster than looping over columns and calling count() each time.
    """
    if not df.columns:
        return {}
    agg_exprs = [
        F.sum(F.when(is_blank_or_null(c), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]
    row = df.select(*agg_exprs).collect()[0]
    return {c: int(row[c] or 0) for c in df.columns}


# ----------------------------------------------------------------------
# Audit trail for the cleaning pipeline
# ----------------------------------------------------------------------

@dataclass
class CleaningStep:
    """One entry in a cleaning audit trail."""
    operation: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def cols_removed(self) -> int:
        return self.cols_before - self.cols_after


@dataclass
class CleaningContext:
    """
    Accumulates an audit trail across multiple cleaning operations.

    Example:
        ctx = CleaningContext()
        df = ctx.record("fill_mean(latitude)", df, fill_mean(df, "latitude"))
        df = ctx.record("drop sparse cols", df, drop_high_null_cols(df, 0.6))
        ctx.summary()
    """
    steps: List[CleaningStep] = field(default_factory=list)

    def record(self, operation: str, before: DataFrame, after: DataFrame,
               details: Optional[Dict[str, Any]] = None) -> DataFrame:
        """Log a cleaning step and return the after-DataFrame."""
        start = time.time()
        rb, ca = before.count(), len(before.columns)
        ra, cb = after.count(), len(after.columns)
        step = CleaningStep(
            operation=operation,
            rows_before=rb, rows_after=ra,
            cols_before=ca, cols_after=cb,
            duration_seconds=round(time.time() - start, 2),
            details=details or {},
        )
        self.steps.append(step)
        logger.info(
            "%s · rows %d→%d (%+d) · cols %d→%d (%+d)",
            operation, rb, ra, ra - rb, ca, cb, cb - ca,
        )
        return after

    def summary(self) -> List[Dict[str, Any]]:
        """Return audit log as a list of dicts."""
        return [
            {
                "step": i + 1,
                "operation": s.operation,
                "rows_before": s.rows_before,
                "rows_after": s.rows_after,
                "rows_removed": s.rows_removed,
                "cols_before": s.cols_before,
                "cols_after": s.cols_after,
                "cols_removed": s.cols_removed,
            }
            for i, s in enumerate(self.steps)
        ]


def log_transform(fn: Callable) -> Callable:
    """
    Decorator that logs row-count and column-count deltas for any function
    that takes a DataFrame as its first argument and returns a DataFrame.
    """
    @functools.wraps(fn)
    def wrapper(df: DataFrame, *args, **kwargs):
        rb, cb = df.count(), len(df.columns)
        out = fn(df, *args, **kwargs)
        ra, ca = out.count(), len(out.columns)
        logger.debug(
            "%s · rows %d→%d (%+d) · cols %d→%d (%+d)",
            fn.__name__, rb, ra, ra - rb, cb, ca, ca - cb,
        )
        return out
    return wrapper
