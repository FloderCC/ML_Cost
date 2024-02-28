"""
This file contains auxiliary methods to simplify the understanding of the experiment file
"""

import numpy as np

from src.utils.dataset_descriptors import *


def preprocess_dataset(df: DataFrame) -> DataFrame:
    # replacing infinite values by the maximum allowed value
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if np.any(np.isinf(df[numeric_columns])):
        print(" - Replacing infinite values by the maximum allowed value")
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # replacing missing values by mean
    if df.isnull().any().any():
        print(" - Replacing missing values by mean")
        df.fillna(df.mean(), inplace=True)

    # encoding all no numerical columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for column in df.columns:
        if not df[column].dtype.kind in ['i', 'f']:
            print(f" - Encoding column {column}")
            df[column] = le.fit_transform(df[column].astype(str))

    return df


def describe_raw_dataset(df: DataFrame, class_name: str) -> list:
    return [
        get_class_imbalance_ratio(df, class_name),
        get_gini_impurity(df, class_name),
        get_entropy(df, class_name),
        get_number_of_samples(df),
        get_number_of_input_features(df),
        get_completeness(df, class_name),
        get_consistency(df, class_name),
        get_uniqueness(df),
    ]


def describe_codified_dataset(df: DataFrame, class_name: str) -> list:
    avg_correlation, std_correlation = get_redundancy(df, class_name)
    max_values, min_values = get_max_and_min(df, class_name)
    avg_values, std_values = get_avg_and_std(df, class_name)
    avg_of_avg, std_of_avg, avg_of_std, std_of_std = get_avg_and_std_of_features_avg_and_std(df, class_name)

    return [
        avg_correlation, std_correlation,
        max_values, min_values,
        avg_values, std_values,
        avg_of_avg, std_of_avg, avg_of_std, std_of_std
    ]
