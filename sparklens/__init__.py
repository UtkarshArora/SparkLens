"""
sparklens
=========

A reusable PySpark library for data quality assessment and preprocessing.

Modules:
    missing_duplicates  Missing value handling and duplicate detection (Debdeep)
    utils               Shared helpers, logger, and audit context (Debdeep)

Other modules will be added by teammates:
    outliers            Outlier detection and treatment (Harshita)
    pipeline            End-to-end SparkLens pipeline wrapper (team)
"""

from . import missing_duplicates
from . import schema_profiling
from . import utils

__version__ = "0.1.0"

__all__ = ["missing_duplicates", "schema_profiling", "utils"]
