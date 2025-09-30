#Run under /shared/share_scp/coresignal/code_run
#grid_run --grid_mem=60g --grid_ncpus=6 --grid_submit=batch --grid_array=1-4 /user/jag2367/.conda/envs/jgpriv/bin/python ../gitrepo_facebook/process_coresignal_match_to_universities_parallel.py
#grid_run --grid_mem=60g --grid_ncpus=6 --grid_submit=batch --grid_array=1-4 /user/jag2367/.conda/envs/jgpriv/bin/python  ../gitrepo_facebook/process_coresignal_match_to_universities_parallel.py --graduation_year_start=1997 --graduation_year_end=2011
#grid_run --grid_mem=60g --grid_ncpus=6  /user/jag2367/.conda/envs/jgpriv/bin/python ../gitrepo_facebook/process_coresignal_match_to_universities_parallel.py --graduation_year_start=2009 --graduation_year_end=2015

import pdb 
import pandas as pd
import time
import glob
from fuzzywuzzy import fuzz
import rapidfuzz
from rapidfuzz import process as rapid_process
from collections import Counter
import os
from rapidfuzz.distance import Levenshtein
import datetime
import sys
from university_name_matcher import university_name_matcher


########
#  
#  Setup the code to start and Run effectively
# #######
os.chdir('/shared/share_scp/coresignal')

def update_parameters_from_argv():
    global PARAMETERS

    for arg in sys.argv[1:]:
        if arg.startswith('--') and '=' in arg:
            key, value = arg[2:].split('=', 1)
            print(f"Default parameter '{key}' updated to to '{value}'", flush=True)

            if key in PARAMETERS:
                # Try to infer type from existing value
                current_type = type(PARAMETERS[key])
                try:
                    if current_type is bool:
                        # Accept 'true'/'false' (case-insensitive)
                        PARAMETERS[key] = value.lower() == 'true'
                    else:
                        PARAMETERS[key] = current_type(value)
                except Exception:
                    PARAMETERS[key] = value  # fallback to string



PARAMETERS = {
    'graduation_year_start': 1995,
    'graduation_year_end': 2001,
    'output_file_prefix_base': 'linkedin_profiles_coresignal_matched',
    'output_file_folder': 'processed_data2',
    'debug_mode': False,  # Set to True to enable debug prints
    'full_process_chunk_size': 8_000_000,  # Number of rows to process in one go
    'file_read_chunk_size': 100_000  # Number of rows to read at a time from CSV
}
update_parameters_from_argv()

PARAMETERS['output_file_prefix'] = PARAMETERS['output_file_prefix_base'] + f'--_school_end_{PARAMETERS["graduation_year_start"]}_to_{PARAMETERS["graduation_year_end"]}'  

umatcher = university_name_matcher()
#
# End of code setup
########
 





#os.chdir('..')
# Load the list of universities with their geographic coordinates
universities_adopted_facebook = umatcher.load_university_data() 






def read_education_file_filter_to_school_years(chunk_data, start_year=PARAMETERS['graduation_year_start'], end_year=PARAMETERS['graduation_year_end']):
    '''
    Filters the education data to include only records with 'year_from' or 'year_to' between 2002 and 2008.
    '''
    edu_df = chunk_data.copy()
    edu_df['year_from'] = pd.to_numeric(edu_df['date_from'], errors='coerce').astype('Int64')
    edu_df['year_to'] = pd.to_numeric(edu_df['date_to'], errors='coerce').astype('Int64')
    edu_df = edu_df.dropna(subset=['year_from', 'year_to'])
    edu_df = edu_df.reset_index(drop=True)
    
    ### Updated to be only based on year_to edu_df = edu_df[(edu_df.year_from.between(start_year, end_year)) | (edu_df.year_to.between(start_year, end_year))]
    edu_df = edu_df[(edu_df.year_to.between(start_year, end_year))]

    return edu_df



def matching_add_university_unitid(linkedin_matches, url_matches, university_urls_to_unitid_correspondence):
    '''
    Adds the university ID variable based on 
    '''
    university_urls_to_unitid_correspondence = pd.concat([url_matches[['school_url', 'unitid']], university_urls_to_unitid_correspondence],
                                                         axis=0).drop_duplicates()
    university_urls_to_unitid_correspondence = university_urls_to_unitid_correspondence.dropna()
    linkedin_matches = pd.merge(linkedin_matches, university_urls_to_unitid_correspondence, on='school_url', how='left').drop_duplicates(subset=['member_id', 'unitid'])

    if debug_mode and linkedin_matches['unitid'].isnull().any():
        print("Stopping execution for review. Some linkedin matches have a missing unitid, this should not happen")
        pdb.set_trace()

    return linkedin_matches  , university_urls_to_unitid_correspondence





def match_education_file_to_universities(
    edu_df, universities_adopted_facebook, university_urls_keep=None, university_urls_remove=None, score_cutoff=85 , 
    university_urls_to_unitid_correspondence= pd.DataFrame()
):
    '''
    Selects only those people that attended universities that are in the universities_adopted_facebook dataset.
    '''
    if university_urls_keep is None:
        university_urls_keep = []
    if university_urls_remove is None:
        university_urls_remove = []

    linkedin_matches = edu_df.copy()
    #pdb.set_trace()
    linkedin_matches = linkedin_matches[~linkedin_matches['school_url'].isna()]
    linkedin_matches = linkedin_matches[~linkedin_matches['school_url'].str.endswith('linkedin.com/edu/school')]
    if university_urls_remove:
        linkedin_matches = linkedin_matches[~linkedin_matches['school_url'].isin(university_urls_remove)]
    linkedin_matches['keep'] = linkedin_matches['school_url'].isin(university_urls_keep)
    linkedin_matches['title'] = linkedin_matches['title'].str.lower().str.strip()
    

    unique_title_school_url = linkedin_matches[['title', 'school_url']].drop_duplicates()

    if universities_adopted_facebook is None:
        return None, None, None, None

    if debug_mode: print("Matching universities by tags and clean name with score cutoff:", score_cutoff)
    url_matches = umatcher.match_by_tags_and_clean_name(
        universities_adopted_facebook, unique_title_school_url, score_cutoff=score_cutoff
    )
    if debug_mode: print("Matching done")


    # If no matches found, return early
    if (
        url_matches is None or 
        url_matches.shape[0] == 0 or 
        'school_url' not in url_matches.columns
    ):
        if debug_mode: print("No matches found in this pass")
        return None, None, None, None


    for url in url_matches.school_url.unique():
        if url not in university_urls_keep:
            university_urls_keep.append(url)

    linkedin_matches.loc[linkedin_matches['school_url'].isin(url_matches['school_url']), 'keep'] = True

    for url in linkedin_matches[~linkedin_matches['keep']].school_url.unique():
        if url not in university_urls_keep and url not in university_urls_remove:
            university_urls_remove.append(url)

    
    linkedin_matches_to_return = linkedin_matches[linkedin_matches['keep']].drop_duplicates(subset=['id'])

    if debug_mode: print("Total matches to return in this pass (before adding unitid):", linkedin_matches_to_return.shape[0])
    linkedin_matches_to_return, university_urls_to_unitid_correspondence = matching_add_university_unitid(linkedin_matches_to_return, 
                                                                                                          url_matches, 
                                                                                                          university_urls_to_unitid_correspondence)
    if debug_mode: print("Total matches to return in this pass (after adding unitid):", linkedin_matches_to_return.shape[0])
    return linkedin_matches_to_return, university_urls_keep, university_urls_remove, university_urls_to_unitid_correspondence







def process_education_relation_file_large_chunk(csv_path, start_row=0, full_process_chunk_size=PARAMETERS['full_process_chunk_size'], process_chunk_id=None, default_num_rows=None):
    '''
    Gets the education file to keep only those that match to the right universities and years of study,
    to then develop a dataset of their linkedin profiles
    '''
    global debug_mode , debug_var_total_rows , debug_var_time_elapsed_matching, PARAMETERS

    chunk_size = PARAMETERS['file_read_chunk_size']
    university_urls_keep = []
    university_urls_remove = []
    university_urls_to_unitid_correspondence = pd.DataFrame()
    all_linkedin_matches = pd.DataFrame()

   
    # Read the header first to get column titles
    if debug_mode: print("reading titles")

    #with open(csv_path, 'r') as f:
    #    header = f.readline().strip().split(',')

    # Calculate rows to skip (excluding header)
    skiprows = range(1, start_row + 1) if start_row > 0 else None
    nrows = full_process_chunk_size

    if debug_mode: print("Opening iterator")

    chunk_iter = pd.read_csv(
        csv_path,
        header=0,
        usecols=['id','activities_and_societies','description','member_id', 'title', 'subtitle','date_from', 'date_to', 'school_url'],
        dtype={'id': int, 'member_id': int, 'title': str, 'activities_and_societies': str, 'subtitle':str,'date_from': str, 'date_to': str, 'school_url': str},
        skiprows=skiprows,
        nrows=nrows,
        low_memory=False,
        iterator=True,
        chunksize=chunk_size
    )
    if debug_mode: print("Iterator opened")

    pickle_file_name = csv_path.replace('.csv', f'{PARAMETERS["output_file_prefix"]}_START-{start_row}__SIZE-{full_process_chunk_size}.pkl')
    
    pickle_path = os.path.join(PARAMETERS['output_file_folder'], pickle_file_name)
    processed_row_count = 0
    included_row_count = 0

    if debug_mode: print(f"pickle path will be {pickle_path}")

    for chunk_data in chunk_iter:
        if debug_mode: print("Reading chunk")
        edu_df = read_education_file_filter_to_school_years(chunk_data)
        if debug_mode: print("Chunk read. Matching to universities")
        linkedin_matches, urls_keep, urls_remove, university_urls_to_unitid_correspondence = match_education_file_to_universities(
            edu_df, universities_adopted_facebook, university_urls_keep, university_urls_remove, 
            university_urls_to_unitid_correspondence=university_urls_to_unitid_correspondence
        )
        if linkedin_matches is None:
            print(f"\tValue Error parameters: \n" + "\n\t\t".join([
                f"csv_path = {csv_path}", 
                f"start_row = {start_row}",
                f"full_process_chunk_size={full_process_chunk_size}",
                f"process_chunk_id={process_chunk_id}",
                f"processed_row_count={processed_row_count}"
                ]))
            continue

        if debug_mode: print("Updating loop variables")
        # Update university_urls_keep with new urls, avoiding duplicates
        university_urls_keep = list(set(university_urls_keep).union(urls_keep))
        university_urls_remove = list(set(university_urls_remove).union(urls_remove))

        all_linkedin_matches = pd.concat([all_linkedin_matches, linkedin_matches], ignore_index=True)
        processed_row_count += chunk_data.shape[0]
        included_row_count += linkedin_matches.shape[0]
        share_included_row_count = included_row_count / processed_row_count * 100 if processed_row_count > 0 else 0
        share_included_row_count = round(share_included_row_count, 1)

        process_name = f"Process {process_chunk_id}" if process_chunk_id is not None else "Main Process"
        print(f"{process_name}: Processed {processed_row_count:,} rows, start {start_row:,} {included_row_count:,} included ({share_included_row_count}% inclusion rate).", flush=True)

        # Save progress every 10 loops or at the last chunk
        if processed_row_count % (chunk_size * 10) == 0 or chunk_data.shape[0] < chunk_size:
            all_linkedin_matches.to_pickle(pickle_path)
            print(f"{process_name}: Saved progress to {pickle_path} at {processed_row_count:,} rows.", flush=True)

        if debug_mode:
            debug_var_total_rows += chunk_data.shape[0]
            rows_per_second = round(debug_var_total_rows / debug_var_time_elapsed_matching, 1) if debug_var_time_elapsed_matching > 0 else 0
            print(f"{process_name}: Effective match rate {rows_per_second:,} rows/sec")

    # Final save
    all_linkedin_matches.to_pickle(pickle_path)
    print(f"Final save to {pickle_path} with {all_linkedin_matches.shape[0]} rows.")





def parallel_process_by_large_chunks(csv_path,  chunk_size=PARAMETERS['full_process_chunk_size'], max_workers=6):

    print(f"Estimating the number of rows in file {csv_path}")
    # Efficiently get total rows in the file (excluding header)
    with open(csv_path, 'rb') as f:
        for i, _ in enumerate(f, 1):
            pass
    total_rows = i - 1  # subtract header
    
    print(f"Detected {total_rows:,} rows.")

    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        chunk_id = 0
        for start_row in range(0, total_rows, chunk_size):
            future = executor.submit(process_education_relation_file_large_chunk, csv_path, start_row, chunk_size, chunk_id)
            futures.append(future)
            chunk_id += 1
        
        # Collect filenames (no data merging)
        output_files = []
        for future in concurrent.futures.as_completed(futures):
            filename = future.result()
            output_files.append(filename)
    
    print(f"Processing complete. Created {len(output_files)} chunk files:")
    for filename in output_files:
        print(f"  - {filename}")
    
    return output_files




import concurrent.futures

#parameters
run_concurrent_files = True # Set to True to run in parallel, False is necessary fro debugging
debug_mode = PARAMETERS['debug_mode'] # if true prints a bunch of messages
debug_var_total_rows = 0
debug_var_time_elapsed_matching = 0



university_urls_to_unitid_correspondence = pd.DataFrame() 

if __name__ == "__main__":

    # Update PARAMETERS from command line arguments
    
    
    print(f'files will be stored under folder {PARAMETERS["output_file_folder"]} with prefix {PARAMETERS["output_file_prefix"]}', flush=True)

    csv_files = sorted(glob.glob('coresignal_member_education_*.csv'))
    print(f"Found {len(csv_files)} files: {csv_files}", flush=True)

    # Check for SGE_TASK_ID environment variable, which is passed during batch processes
    sge_task_id = os.environ.get('SGE_TASK_ID')
    if sge_task_id is not None:
        try:
            task_idx = int(sge_task_id) - 1  # SGE_TASK_ID is 1-based
            if 0 <= task_idx < len(csv_files):
                csv_files = [csv_files[task_idx]]
                print(f"SGE_TASK_ID detected: processing file {csv_files[0]}")
            else:
                print(f"SGE_TASK_ID {sge_task_id} is out of range. Processing all files.")
        except ValueError:
            print(f"Invalid SGE_TASK_ID: {sge_task_id}. Processing all files.")
    else:
        print(f"SGE_TASK_ID not provided. Processing all files.")
        

    if run_concurrent_files is True:
       for csv_path in csv_files:
                parallel_process_by_large_chunks(csv_path, max_workers=5)
  
    else:
            print("Running in non-concurrent mode, one csv at a time")
            for csv_path in csv_files:
                
                process_education_relation_file_large_chunk(csv_path, start_row=192_000_000, default_num_rows=220_000_000)

