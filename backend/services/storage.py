import logging

import boto3

logger = logging.getLogger(__name__)

# SageMaker appends /{job_name}/output/ to whatever S3OutputPath we provide,
# and packages the container's /opt/ml/output/data/ as output.tar.gz.
# Given our output prefix of jobs/{job_id}/output, the tarball ends up here:
_OUTPUT_KEY_TEMPLATE = "jobs/{job_id}/output/{job_id}/output/output.tar.gz"

PRESIGNED_URL_EXPIRY = 3600  # 1 hour


class StorageClient:
    def __init__(self, region: str = "eu-west-1"):
        self.s3 = boto3.client("s3", region_name=region)

    def generate_presigned_url(self, bucket: str, job_id: str) -> str:
        key = _OUTPUT_KEY_TEMPLATE.format(job_id=job_id)
        url = self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=PRESIGNED_URL_EXPIRY,
        )
        logger.info(f"[storage] Generated presigned URL for {job_id}")
        return url

    def cleanup_job(self, bucket: str, job_id: str) -> None:
        prefix = f"jobs/{job_id}/"
        paginator = self.s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        objects = [
            {"Key": obj["Key"]}
            for page in pages
            for obj in page.get("Contents", [])
        ]

        if not objects:
            logger.info(f"[storage] Nothing to clean up for {job_id}")
            return

        self.s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": objects},
        )
        logger.info(f"[storage] Deleted {len(objects)} objects under {prefix}")
