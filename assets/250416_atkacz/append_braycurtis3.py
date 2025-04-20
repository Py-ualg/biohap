#!/usr/bin/env python3
import sys
import os
import time
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from concurrent.futures import ProcessPoolExecutor
import sys

# File paths (all stored under the "saved_matrices" folder)
OUTPUT_FOLDER = "saved_matrices"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
MATRIX_FILE = os.path.join(OUTPUT_FOLDER, "braycurtis_matrix_columns.npy")
FEATURE_NAMES_FILE = os.path.join(OUTPUT_FOLDER, "feature_names.txt")
OLD_TABLE_FILE = os.path.join(OUTPUT_FOLDER, "old_asv_table.csv")

# A small epsilon to avoid division by zero
EPSILON = 1e-12

def update_progress(task, percent):
    """Print progress messages and flush immediately."""
    print(f"Progress Update: {task} - {percent}% complete", flush=True)

#############################
# Parallel functions for new table internal distances
#############################

# Global variables for new table internal computations.
new_table_global = None
m_global = None

def init_worker_new(table, m):
    """Initializer for workers computing distances within new table."""
    global new_table_global, m_global
    new_table_global = table
    m_global = m

def compute_new_col(i):
    """Compute Bray–Curtis distances for new table’s column i vs. columns j > i.
    Returns (i, result_vector) where result_vector is a NumPy array of length m_global.
    """
    row_result = np.zeros(m_global, dtype=np.float64)
    x = new_table_global.iloc[:, i].values
    for j in range(i+1, m_global):
        y = new_table_global.iloc[:, j].values
        # Compute sum(x+y) with safeguard
        denom = np.sum(x + y)
        if denom < EPSILON:
            d = 0.0
        else:
            d = np.sum(np.abs(x - y)) / denom
        row_result[j] = d
    return i, row_result

#############################
# Parallel functions for cross distances between new and old table
#############################

# Global variables for cross-computation.
old_table_global = None
new_table_global_cross = None
n_old_global = None
m_global_for_cross = None  # number of new features

def init_worker_cross(old_table, new_table, m):
    """Initializer for workers computing cross distances.
    Sets globals for old table and new table.
    """
    global old_table_global, new_table_global_cross, n_old_global, m_global_for_cross
    old_table_global = old_table
    new_table_global_cross = new_table
    n_old_global = old_table.shape[1]  # number of old features (columns)
    m_global_for_cross = m

def compute_cross(i):
    """For new table column i, compute Bray–Curtis distances to each old table column.
    Returns (i, result_vector) of length n_old_global.
    """
    result = np.zeros(n_old_global, dtype=np.float64)
    x = new_table_global_cross.iloc[:, i].values
    for j in range(n_old_global):
        y = old_table_global.iloc[:, j].values
        denom = np.sum(x + y)
        if denom < EPSILON:
            d = 0.0
        else:
            d = np.sum(np.abs(x - y)) / denom
        result[j] = d
    return i, result

#############################
# Main function
#############################
def main():
    if len(sys.argv) != 2:
        print("Usage: python append_braycurtis.py new_table.txt", flush=True)
        sys.exit(1)
    input_file = sys.argv[1]

    # Load new table using Dask (features remain as columns)
    update_progress("Loading new table with Dask", 0)
    with ProgressBar():
        new_ddf = dd.read_csv(input_file, sep="\t", assume_missing=True, sample=10_000_000)
        new_ddf = new_ddf.persist()
    # Use first column as index.
    new_ddf = new_ddf.set_index(new_ddf.columns[0])
    new_ddf = new_ddf.map_partitions(lambda df: df.apply(pd.to_numeric, errors="coerce"))
    with ProgressBar():
        non_nan = new_ddf.isnull().sum().compute() == 0
    good_cols = non_nan[non_nan].index.tolist()
    new_ddf = new_ddf[good_cols]
    update_progress("Dropped NaN columns in new table", 10)
    with ProgressBar():
        new_table = new_ddf.compute()
    update_progress("New table computed", 20)
    # We compare columns, so do not transpose.
    new_features = list(new_table.columns)
    m = len(new_features)
    update_progress("New table ready", 30)
    print("New features:", new_features, flush=True)

    # Check if an old table already exists.
    if os.path.exists(OLD_TABLE_FILE) and os.path.exists(FEATURE_NAMES_FILE) and os.path.exists(MATRIX_FILE):
        print("Old table exists. Entering append mode.", flush=True)
        old_table = pd.read_csv(OLD_TABLE_FILE, index_col=0)
        old_features = list(old_table.columns)
        n_old = len(old_features)
        print(f"Loaded old table with {n_old} features.", flush=True)
        # Load the existing BC matrix.
        existing_matrix = np.load(MATRIX_FILE)
    else:
        print("No old table found. This run will create the initial BC matrix.", flush=True)
        # Compute BC matrix for new table only (internal comparisons)
        with ProcessPoolExecutor(max_workers=54, initializer=init_worker_new, initargs=(new_table, m)) as executor:
            M_new = np.zeros((m, m), dtype=np.float64)
            for i, row_result in executor.map(compute_new_col, range(m)):
                M_new[i, :] = row_result
                for j in range(i+1, m):
                    M_new[j, i] = row_result[j]
                update_progress(f"Processed new feature {i+1} of {m}", int((i+1)/m*100))
        # Force symmetry and set diagonal to zero
        M_new = (M_new + M_new.T) / 2
        np.fill_diagonal(M_new, 0)
        # Save as the new BC matrix.
        np.save(MATRIX_FILE, M_new)
        with open(FEATURE_NAMES_FILE, "w") as f:
            for feat in new_features:
                f.write(feat + "\n")
        # Save new_table as old_table for future appends.
        new_table.to_csv(OLD_TABLE_FILE)
        print("Initial BC matrix computed and saved.", flush=True)
        update_progress("All tasks complete", 100)
        return

    # === Append mode ===
    # Compute internal distances among new table’s features (parallelized)
    with ProcessPoolExecutor(max_workers=54, initializer=init_worker_new, initargs=(new_table, m)) as executor:
        M_new = np.zeros((m, m), dtype=np.float64)
        for i, row_result in executor.map(compute_new_col, range(m)):
            M_new[i, :] = row_result
            for j in range(i+1, m):
                M_new[j, i] = row_result[j]
            update_progress(f"Processed new feature {i+1} of {m}", int((i+1)/m*100))
    # Force symmetry for new internal matrix:
    M_new = (M_new + M_new.T) / 2
    np.fill_diagonal(M_new, 0)

    # Compute cross distances: for each new feature vs. each old feature.
    with ProcessPoolExecutor(max_workers=54, initializer=init_worker_cross, initargs=(old_table, new_table, m)) as executor:
        M_cross = np.zeros((n_old, m), dtype=np.float64)
        for i, cross_result in executor.map(compute_cross, range(m)):
            M_cross[:, i] = cross_result
            update_progress(f"Processed cross distance for new feature {i+1} of {m}", int((i+1)/m*100))
    # Combine existing BC matrix (shape n_old x n_old), new internal matrix (m x m),
    # and cross distances (n_old x m) into a new matrix.
    new_n = n_old + m
    M_combined = np.zeros((new_n, new_n), dtype=np.float64)
    # Top-left: old matrix.
    M_combined[:n_old, :n_old] = existing_matrix
    # Top-right: cross matrix.
    M_combined[:n_old, n_old:] = M_cross
    # Bottom-left: transpose of cross.
    M_combined[n_old:, :n_old] = M_cross.T
    # Bottom-right: new internal distances.
    M_combined[n_old:, n_old:] = M_new
    # Force symmetry on the combined matrix and set diagonal to zero.
    M_combined = (M_combined + M_combined.T) / 2
    np.fill_diagonal(M_combined, 0)

    # Save combined BC matrix.
    np.save(MATRIX_FILE, M_combined)
    # Update feature names.
    combined_features = old_features + new_features
    with open(FEATURE_NAMES_FILE, "w") as f:
        for feat in combined_features:
            f.write(feat + "\n")
    # Also update the old table by concatenating the old and new tables (by columns).
    combined_table = pd.concat([old_table, new_table], axis=1)
    combined_table.to_csv(OLD_TABLE_FILE)
    print("BC matrix updated with new table. Combined matrix saved.", flush=True)
    update_progress("All tasks complete", 100)

if __name__ == '__main__':
    main()

