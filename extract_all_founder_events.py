#grid_run --grid_mem=60g --grid_ncpus=6 /user/jag2367/.conda/envs/jgpriv/bin/python ../gitrepo_facebook/extract_all_founder_events.py
#grid_run --grid_mem=60g --grid_ncpus=6 --grid_submit=batch --grid_array=1-5 /user/jag2367/.conda/envs/jgpriv/bin/python ../gitrepo_facebook/extract_all_founder_events.py


#OVERVIEW:
#  
# This file extracts all founder events with a URL from the experience files, 
# independent if they are students in an undergraduate institution or not.


import pandas as pd
import os
import glob
import csv
import datetime
import polars as pl
import concurrent.futures
import pdb

# Set working directory
os.chdir('/shared/share_scp/coresignal')

def process_experience_file_large_chunk( csv_path, start_row=0, full_process_chunk_size=15_000_000 ,process_chunk_id=None):
    # Load member IDs from processed education data
    print(f"Process {process_chunk_id}: Loading file: {csv_path} start_row {start_row:,}", flush=True)
    read_chunk_size = 1_000_000

    skiprows = range(1, start_row + 1) if start_row > 0 else None
    nrows = full_process_chunk_size
    experience_matches = pd.DataFrame()      

    output_path = csv_path.replace('.csv', f'_linkedin_founder_events__START-{start_row}__SIZE-{full_process_chunk_size}.pkl') 
    output_path = os.path.join("/shared/share_scp/coresignal/processed_data2/", output_path)

    chunk_iter = pd.read_csv(
        csv_path,
        chunksize=read_chunk_size,
        skiprows=skiprows,
        nrows=nrows,
        low_memory=False,
        on_bad_lines='skip',  # Skip bad lines instead of failing
        quotechar='"',        # Use standard double quotes
        doublequote=True,     # Handle escaped quotes properly
        encoding_errors='replace',
        skipinitialspace=True,  # Skip whitespace after delimiter
        sep=',',               # Explicitly set separator
        dtype={'member_id': 'str'},  # Read as string initially
        quoting=1            # QUOTE_ALL to handle malformed quotes
    
    )
    

    counter = 0 
    try:
        for chunk in chunk_iter:
            counter += 1
            print(f"Process {process_chunk_id}: chunk_number {counter}", flush=True)
            
            mask = chunk['company_url'].notna() & (chunk['title'].str.contains('founder', case=False, na=False, regex=False) | 
                                                   chunk['title'].str.contains('owner', case=False, na=False, regex=False))
            chunk_filtered = chunk[mask]
            experience_matches = pd.concat([chunk_filtered, experience_matches], axis=0, ignore_index=True)
                    
            # Only save if experience_matches is not empty (finalized)
            if (counter % 10 == 0) and not experience_matches.empty:                    
                print(f"Process {process_chunk_id}: Saving progress to {output_path}", flush=True)
                experience_matches.to_pickle(output_path)
    except StopIteration:
        print(f"Process {process_chunk_id}: Iterator exhausted normally at chunk {counter}", flush=True)
    except pd.errors.ParserError as e:
        print(f"Process {process_chunk_id}: ParserError encountered at chunk {counter}: {e}", flush=True)
        print(f"Process {process_chunk_id}: Continuing with data processed so far ({len(experience_matches)} rows)", flush=True)
    except Exception as e:
        print(f"Process {process_chunk_id}: Unexpected error at chunk {counter}: {type(e).__name__}: {e}", flush=True)
        print(f"Process {process_chunk_id}: Continuing with data processed so far ({len(experience_matches)} rows)", flush=True)
        

    print(f"Process {process_chunk_id}: Saving progress to {output_path}", flush=True)
    experience_matches.to_pickle(output_path)
    return output_path




def process_experience_file(csv_path, count_total_rows=False, max_workers=10, run_single_process=False):
    '''
    Process the experience CSV file in large chunks.
    Utilizes multiprocessing to handle large files efficiently.

    This one calls process_experience_file_large_chunk in parallel for different segments of the CSV file.
    '''
    
    if count_total_rows:
        print("Counting the number of rows in experience file", flush=True)
        with open(csv_path, 'rb') as f:
            for i, _ in enumerate(f, 1):
                pass
        total_rows = i - 1  # subtract header
        print(f"{total_rows:,} rows found", flush=True)
    else:
        total_rows = 603_193_546
        print(f"Skipping estimates of total rows, using default value {603_193_546:,}", flush=True)


    
    output_path = csv_path.replace('.csv', '_founder_events.pkl')

    chunk_counter = 0
    latest_chunk = None

    large_chunk_size = 1_000_000  # Adjusted chunk size for better performance


    
    if run_single_process:
        chunk_id = 0      
        for start_row in range(0, total_rows, large_chunk_size):
            chunk_id += 1
            #pdb.set_trace()
            process_experience_file_large_chunk(csv_path, start_row, large_chunk_size, chunk_id)

    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            chunk_id = 0
            for start_row in range(0, total_rows, large_chunk_size):
                future = executor.submit(process_experience_file_large_chunk, csv_path, start_row, large_chunk_size, chunk_id)
                futures.append(future)
                chunk_id += 1

            # Collect filenames (no data merging)
            output_files = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    filename = future.result()
                    output_files.append(filename)
                except Exception as e:
                    print(f"Process failed with error: {type(e).__name__}: {e}", flush=True)
                    print(f"Continuing with remaining processes...", flush=True)
        
            print(f"Processing complete. Created {len(output_files)} chunk files:", flush=True)
            for filename in output_files:
                print(f"  - {filename}", flush=True)
            



if __name__ == "__main__":
    # Find all experience files
    experience_files = sorted(glob.glob('coresignal_member_experience_*.csv'), reverse=True )

        # Check for SGE_TASK_ID environment variable, which is passed during batch processes
    sge_task_id = os.environ.get('SGE_TASK_ID')

    run_single_process = (sge_task_id  == 'undefined')
    if sge_task_id != 'undefined':
        try:
            task_idx = int(sge_task_id) - 1  # SGE_TASK_ID is 1-based
            if 0 <= task_idx < len(experience_files):
                experience_files = [experience_files[task_idx]]
                print(f"SGE_TASK_ID detected: processing file {experience_files[0]}", flush=True)
            else:
                print(f"SGE_TASK_ID {sge_task_id} is out of range. Processing all files.", flush=True)
        except ValueError:
            print(f"Invalid SGE_TASK_ID: {sge_task_id}. Processing all files.", flush=True)
    else:
        print(f"SGE_TASK_ID not provided. Processing all files.", flush=True)
        
    
    count_total_rows = not run_single_process
    max_workers = 6


#    Process each experience file
    run_single_process = False #debugging, remove
    sge_task_id = 1 #debugging, remove

    for csv_path in experience_files:
        print(f"Processing file: {csv_path} with run_single_process={run_single_process} and count_total_rows={count_total_rows}", flush=True)
        process_experience_file(csv_path,count_total_rows=count_total_rows, max_workers=max_workers, run_single_process=run_single_process)