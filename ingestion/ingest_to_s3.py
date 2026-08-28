import boto3
from pathlib import Path

s3 = boto3.client("s3")

dataset_path = Path(r"C:\Users\chsum\Desktop\DE_PROJECT\Olist_Dataset")

bucket_name = "olist-data-engineering-142366489643"

for file in dataset_path.glob("*.csv"):
    s3_file = f"raw/olist/{file.name}"

    s3.upload_file(
        str(file),
        bucket_name,
        s3_file
    )

    print(f"{file.name} uploaded successfully")