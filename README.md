# SparkLens

SparkLens is a PySpark-based data quality and preprocessing library for large-scale datasets. It provides reusable modules for schema validation, column profiling, missing value handling, duplicate removal, and outlier detection. The goal of the project is to simplify data cleaning and validation workflows for messy real-world datasets using Apache Spark.

## Overview

Real-world datasets often contain missing values, duplicate records, schema inconsistencies, invalid data types, and outliers. While Spark provides the distributed processing engine and low-level APIs to handle large datasets, SparkLens builds a higher-level reusable framework for automated data inspection, validation, and preprocessing.

SparkLens is designed as a modular library in which each component of the cleaning pipeline can be developed, tested, and reused independently, while also being integrated into a single end-to-end pipeline.

## Features

- Schema validation

  - validate expected column names
  - validate data types
  - check nullable vs non-nullable columns

- Column profiling

  - null count
  - distinct count
  - min / max
  - mean / stddev for numeric columns

- Missing value handling

  - detect null values
  - compute null percentages
  - fill missing values using configurable strategies
  - drop rows / columns based on thresholds

- Duplicate handling

  - detect duplicate rows
  - remove duplicates
  - compare before / after duplicate counts

- Outlier detection and handling

  - detect outliers using IQR
  - optional z-score based detection
  - cap or remove extreme values
  - flag suspicious values

- Modular pipeline integration
- Scalable execution with Apache Spark

## Project Structure

```text
SparkLens/
│
├── sparklens/
│   ├── __init__.py
│   ├── missing_duplicates.py
│   ├── outliers.py
│   ├── schema_profiling.py
│   ├── pipeline.py
│   └── utils.py
│
├── tests/
├── demo/
├── data/
├── README.md
└── main.py
```

## Development setup

SparkLens requires Java 17. Newer JDKs (18+) trigger
`UnsupportedOperationException: getSubject` from Hadoop's filesystem layer
when Spark writes output files.

On macOS:

```bash
brew install openjdk@17
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.zshrc
source ~/.zshrc
java -version   # should print openjdk version 17.x.x
```

Then create the project venv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pyspark pytest
```

Run tests with:

```bash
PYTHONPATH=. pytest tests/ -v
```
