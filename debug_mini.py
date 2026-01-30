import s3fs
import os
from dotenv import load_dotenv

load_dotenv()

fs = s3fs.S3FileSystem(
    key=os.getenv("MINIO_ROOT_USER", "admin"),
    secret=os.getenv("MINIO_ROOT_PASSWORD", "password123"),
    client_kwargs={"endpoint_url": "http://localhost:9000"}
)

print("--- Listing buckets ---")
print(fs.ls("/"))

print("\n--- Listing kkbox-raw bucket ---")
try:
    files = fs.ls("kkbox-raw")
    for f in files:
        print(f)
except Exception as e:
    print(f"Error reading bucket: {e}")