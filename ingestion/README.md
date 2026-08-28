# Data Ingestion

This module uploads the Olist dataset from the local machine to AWS S3.

## Flow

Local Dataset → Python → boto3 → AWS S3

## Script

`ingest_to_s3.py` finds all CSV files in the local Olist dataset folder and uploads them to:

`raw/olist/`