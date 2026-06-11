import os
import math
import pandas as pd


def _split_pickle_name(base_path, part_number):
    """
    Part 1 keeps the original filename.
    Later parts get _part002, _part003, etc.
    """
    root, ext = os.path.splitext(base_path)

    # Treat .pkl.gz as one extension
    if root.endswith(".pkl") and ext == ".gz":
        root = root[:-4]
        ext = ".pkl.gz"

    if part_number == 1:
        return base_path

    return f"{root}_part{part_number:03d}{ext}"


def to_pickle_split_by_size(
    df,
    base_path,
    max_size_gb=1.8,
    compression="gzip",
    initial_rows=None,
    safety=0.95,
):
    """
    Save a DataFrame into compressed pickle parts, each targeting max_size_gb.

    The first file keeps the original filename:
        data.pkl.gz

    Later files are numbered:
        data_part002.pkl.gz
        data_part003.pkl.gz

    Because compression ratios vary, this function estimates chunk sizes
    dynamically based on the actual compressed size of each written file.
    """
    if globals().get("SAVE_FILES_TO_REPO") is not True:
        print(f"Skipping save to {base_path}: SAVE_FILES_TO_REPO is not True")
        return []

    max_bytes = max_size_gb * 1024**3 * safety
    n = len(df)

    if n == 0:
        df.to_pickle(base_path, compression=compression)
        print(f"Saved empty DataFrame to {base_path}")
        return [base_path]

    # Start with a rough guess if none is provided.
    # This only affects the first chunk.
    if initial_rows is None:
        initial_rows = min(n, max(1, n // 10))

    files = []
    start = 0
    part = 1
    rows_per_chunk = initial_rows

    while start < n:
        rows_per_chunk = max(1, int(rows_per_chunk))
        end = min(start + rows_per_chunk, n)

        out = _split_pickle_name(base_path, part)

        # Write a tentative chunk.
        df.iloc[start:end].to_pickle(out, compression=compression)
        file_size = os.path.getsize(out)

        # If too large and chunk has more than 1 row, shrink and retry.
        while file_size > max_bytes and rows_per_chunk > 1:
            os.remove(out)

            shrink_factor = max_bytes / file_size
            rows_per_chunk = max(1, int(rows_per_chunk * shrink_factor * 0.95))
            end = min(start + rows_per_chunk, n)

            df.iloc[start:end].to_pickle(out, compression=compression)
            file_size = os.path.getsize(out)

        files.append(out)

        print(
            f"Saved {out}: rows {start:,} to {end:,} "
            f"({file_size / 1024**3:.2f} GB)"
        )

        rows_written = end - start
        start = end
        part += 1

        # Update estimate for the next chunk.
        # If this chunk was much smaller than the limit, increase rows next time.
        if file_size > 0:
            bytes_per_row = file_size / rows_written
            rows_per_chunk = int(max_bytes / bytes_per_row)

    return files