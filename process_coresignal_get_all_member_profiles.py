#!/usr/bin/env python3
"""
Process CoreSignal member profiles from CSV file.

This script:
1. Reads coresignal_member_education_latest.pkl to get all member IDs
2. Processes the large coresignal_member*.csv file (50GB) in chunks
3. Extracts profiles for members found in the education file
"""

## #grid_run --grid_mem=40g --grid_ncpus=6 /user/jag2367/.conda/envs/jgpriv/bin/python process_coresignal_get_all_member_profiles.py


import pickle
import pandas as pd
import pdb
import glob
import os
from pathlib import Path
import logging
import subprocess
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_member_ids_from_pickle(pickle_path):
    """
    Load member IDs from the pickle file.
    
    Args:
        pickle_path: Path to coresignal_member_education_latest.pkl
        
    Returns:
        Set of member IDs
    """
    logger.info(f"Loading member IDs from pickle file: {pickle_path}")
    
    try:
        with open(pickle_path, 'rb') as f:
            education_df = pickle.load(f)
        
        member_ids = set(education_df['member_id'].unique())
        logger.info(f"Loaded {len(member_ids)} unique member IDs from education data")
        
        return member_ids
    except Exception as e:
        logger.error(f"Error loading pickle file: {e}")
        raise


def find_member_csv_files(directory='.'):
    """
    Find all coresignal_member*.csv files in the directory.
    
    Args:
        directory: Directory to search in
        
    Returns:
        List of CSV file paths
    """
    pattern = os.path.join(directory, 'coresignal_member*.csv')
    csv_files = glob.glob(pattern)
    logger.info(f"Found {len(csv_files)} CSV file(s) matching pattern: {pattern}")
    
    return sorted(csv_files)


def get_memory_usage():
    """
    Get current process memory usage.
    
    Returns:
        Tuple of (RSS in GB, percentage of total memory)
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024**3)  # Convert to GB
    mem_percent = process.memory_percent()
    return rss_gb, mem_percent


def count_csv_lines(csv_path):
    """
    Count total lines in CSV file using wc -l.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Total number of lines (including header)
    """
    try:
        if "coresignal_member_1.csv" in csv_path:
            total_lines = 120_104_583
            logger.info(f"Using hardcoded line count for {csv_path}: {total_lines:,}")
            return total_lines
        else:
            result = subprocess.run(['wc', '-l', csv_path], 
                              capture_output=True, 
                              text=True, 
                              check=True)
            total_lines = int(result.stdout.split()[0])
            logger.info(f"Total lines in file (including header): {total_lines:,}")
            return total_lines
    except Exception as e:
        logger.warning(f"Could not count lines: {e}")
        return None


def process_large_csv_file(csv_path, member_ids, output_path=None, chunksize=500_000):
    """
    Process large CSV file efficiently by reading in chunks.
    
    Args:
        csv_path: Path to the CSV file
        member_ids: Set of member IDs to filter by
        output_path: Optional path to save matching profiles
        chunksize: Number of rows to read at a time
        
    Returns:
        DataFrame containing matching member profiles
    """
    logger.info(f"Processing CSV file: {csv_path}")
    logger.info(f"File size: {os.path.getsize(csv_path) / (1024**3):.2f} GB")
    
    # Log initial memory usage
    mem_gb, mem_pct = get_memory_usage()
    logger.info(f"Initial memory usage: {mem_gb:.2f} GB ({mem_pct:.1f}%)")
    
    # Count total lines in the file first
    total_lines = count_csv_lines(csv_path)
    total_data_rows = total_lines - 1 if total_lines else None  # Exclude header
    
    matched_profiles = []
    total_rows = 0
    matched_rows = 0
    
    try:
        # Read CSV in chunks to handle large files
        for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
            total_rows += len(chunk)
            
            #pdb.set_trace()
            # Filter for matching member IDs
            matching_chunk = chunk[chunk['id'].isin(member_ids)]
            matched_rows += len(matching_chunk)
            
            if len(matching_chunk) > 0:
                matched_profiles.append(matching_chunk)
            
            # Log progress
            if (chunk_num + 1) % 100 == 0:
                mem_gb, mem_pct = get_memory_usage()
                if total_data_rows:
                    progress_pct = (total_rows / total_data_rows) * 100
                    logger.info(f"Processed {total_rows:,} rows ({progress_pct:.2f}%), found {matched_rows:,} matches | Memory: {mem_gb:.2f} GB ({mem_pct:.1f}%)")
                else:
                    logger.info(f"Processed {total_rows:,} rows, found {matched_rows:,} matches | Memory: {mem_gb:.2f} GB ({mem_pct:.1f}%)")
        
        mem_gb, mem_pct = get_memory_usage()
        if total_data_rows:
            progress_pct = (total_rows / total_data_rows) * 100
            logger.info(f"Finished processing CSV. Total rows: {total_rows:,} ({progress_pct:.2f}%), Matched: {matched_rows:,} | Memory: {mem_gb:.2f} GB ({mem_pct:.1f}%)")
        else:
            logger.info(f"Finished processing CSV. Total rows: {total_rows:,}, Matched: {matched_rows:,} | Memory: {mem_gb:.2f} GB ({mem_pct:.1f}%)")
        
        if matched_profiles:
            result_df = pd.concat(matched_profiles, ignore_index=True)
            logger.info(f"Combined {len(matched_profiles)} chunks into single DataFrame")
            
            # Save to output file if specified
            if output_path:
                logger.info(f"Saving matched profiles to: {output_path}")
                result_df.to_pickle(output_path, index=False)
                logger.info(f"Saved {len(result_df)} member profiles")
            
            return result_df
        else:
            logger.warning("No matching profiles found")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise


def main():
    """Main execution function."""
    
    # Change working directory
    os.chdir('/shared/share_scp/coresignal')
    logger.info(f"Changed working directory to: {os.getcwd()}")

    # Configuration
    pickle_file = '/shared/share_scp/coresignal/coresignal_member_education_latest.pkl'
    output_file = 'processed_data2/coresignal_member_profiles_matched.pkl'
    
    # Step 1: Load member IDs from pickle file
    if not os.path.exists(pickle_file):
        logger.error(f"Pickle file not found: {pickle_file}")
        return
    
    member_ids = load_member_ids_from_pickle(pickle_file)
    logger.info(f"Target member IDs to find: {len(member_ids)}")
    
    # Step 2: Find CSV files
    csv_files = find_member_csv_files()
    if not csv_files:
        logger.error("No CSV files matching pattern coresignal_member*.csv found")
        return
    
    # Step 3: Process each CSV file
    all_profiles = []
    for csv_file in csv_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing file: {csv_file}")
        logger.info(f"{'='*60}")
        
        profiles_df = process_large_csv_file(csv_file, member_ids)
        if not profiles_df.empty:
            all_profiles.append(profiles_df)
    
    # Step 4: Combine results and save
    if all_profiles:
        logger.info(f"\n{'='*60}")
        logger.info("Combining results from all CSV files...")
        logger.info(f"{'='*60}")
        
        combined_df = pd.concat(all_profiles, ignore_index=True)
        
        # Remove duplicates if any
        combined_df = combined_df.drop_duplicates(subset=['member_id'], keep='first')
        
        logger.info(f"Total unique member profiles found: {len(combined_df)}")
        logger.info(f"Member IDs to find: {len(member_ids)}")
        logger.info(f"Coverage: {len(combined_df) / len(member_ids) * 100:.2f}%")
        
        # Save combined results
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Saved final results to: {output_file}")
        
        return combined_df
    else:
        logger.warning("No matching profiles found in any CSV files")
        return pd.DataFrame()


if __name__ == '__main__':
    main()
