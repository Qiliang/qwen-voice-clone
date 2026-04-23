import os
import boto3

from botocore.client import Config

bucket_name = 'public'

s3 = boto3.client(
    's3',
    endpoint_url='http://120.46.43.62:31676',
    aws_access_key_id='hollycrm',
    aws_secret_access_key='hollycrm',
    config=Config(signature_version='s3v4'),
    region_name='cn-beijing'
)

def upload_file(src_file_path):
    s3.upload_file(src_file_path, bucket_name, os.path.basename(src_file_path))
    return f'http://120.46.43.62:31676/public/{os.path.basename(src_file_path)}'