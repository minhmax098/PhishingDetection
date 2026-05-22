import pandas as pd
import numpy as np

def load_and_harmonize_dataset(filepath, url_col, label_col, phishing_label, benign_label):
    """
    Loads a dataset, extracts URL and Label columns, and harmonizes labels to 1 (phishing) and 0 (benign).
    """
    df = pd.read_csv(filepath)
    if url_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Columns {url_col} or {label_col} not found in dataset.")

    df = df[[url_col, label_col]].copy()
    df.rename(columns={url_col: 'url', label_col: 'label'}, inplace=True)

    # Harmonize labels
    df['label'] = df['label'].apply(lambda x: 1 if str(x).strip().lower() == str(phishing_label).lower() else 0)
    return df

def merge_and_deduplicate(datasets):
    """
    Merges a list of datasets and removes duplicates.
    """
    merged_df = pd.concat(datasets, ignore_index=True)
    initial_len = len(merged_df)

    # Clean URLs (lowercase, strip whitespace)
    merged_df['url'] = merged_df['url'].astype(str).str.lower().str.strip()

    # Deduplication
    merged_df.drop_duplicates(subset=['url'], keep='first', inplace=True)
    final_len = len(merged_df)

    print(f"Merged {len(datasets)} datasets.")
    print(f"Removed {initial_len - final_len} duplicates. Final dataset size: {final_len}.")

    return merged_df

def split_dataset(df, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        df['url'], df['label'], test_size=test_size, random_state=random_state, stratify=df['label']
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Dataset Preparation module ready.")
