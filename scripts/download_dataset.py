# -*- coding: utf-8 -*-
"""
Download the COCO 2017 dataset from Kaggle
"""
import os

import kagglehub
os.environ["KAGGLEHUB_CACHE_DIR"] = r"D:\kaggle_datasets"
# Download latest version
print("Current cache directory:", os.getenv("KAGGLEHUB_CACHE_DIR"))
path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")

print("Path to dataset files:", path)
