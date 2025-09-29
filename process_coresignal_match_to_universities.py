
#grid_run --grid_mem=40g --grid_ncpus=6 /user/jag2367/.conda/envs/jgpriv/bin/python process_coresignal_main.py
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


# Load the list of universities with their geographic coordinates
universities_adopted_facebook = pd.read_stata('universities_geo_for_jorge.dta')

# Split 'instnm' into words and count their incidence
words = universities_adopted_facebook['instnm'].str.lower().str.split().explode()
word_counts = Counter(words)
word_counts_df = pd.DataFrame(word_counts.most_common(), columns=['word', 'count'])
word_counts_df = word_counts_df[word_counts_df['word'].str.len() > 3]
top_10_tags = word_counts_df.head(10)


def clean_university_list(uni_list, name_col):
    '''
    Cleans the university list by extracting tags from the institution names and creating a clean name without those tags.
    The tags are derived from the top 10 most common words in the institution names.
    The 'instnm' column is expected to contain the institution names.
    '''

    global top_10_tags

    # Create 'tags' column: list of tags found in each university name
    uni_list['tags'] = uni_list[name_col].str.lower().apply(
        lambda x: [tag for tag in top_10_tags['word'].tolist() if pd.notnull(x) and tag in str(x)]
    )
    # Sort tags alphabetically and convert to lowercase
    uni_list['tags'] = uni_list['tags'].apply(
        lambda tag_list: sorted([tag.lower() for tag in tag_list])
    )
    # Create 'clean_name' column: remove tags from 'instnm'
    def remove_tags(name, tags):
        for tag in tags:
            name = name.replace(tag, '').strip()
        # Remove extra spaces
        name = ' '.join(name.split())
        return name

    uni_list['clean_name'] = uni_list.apply(
        lambda row: remove_tags(str(row[name_col]).lower() if pd.notnull(row[name_col]) else '', row['tags']),
        axis=1
    )

    return uni_list



def read_education_file_filter_to_school_years(chunk_data):
    '''
    Filters the education data to include only records with 'year_from' or 'year_to' between 2002 and 2008.
    '''
    edu_df = chunk_data.copy()
    edu_df['year_from'] = pd.to_numeric(edu_df['date_from'], errors='coerce').astype('Int64')
    edu_df['year_to'] = pd.to_numeric(edu_df['date_to'], errors='coerce').astype('Int64')
    edu_df = edu_df.dropna(subset=['year_from', 'year_to'])
    edu_df = edu_df.reset_index(drop=True)
    edu_df = edu_df[(edu_df.year_from.between(2002, 2008)) | (edu_df.year_to.between(2002, 2008))]

    return edu_df



def matching_add_university_unitid(linkedin_matches, url_matches):
    '''
    Adds the university ID variable based on 
    '''
    global university_urls_to_unitid_correspondence
    university_urls_to_unitid_correspondence = pd.concat([url_matches[['school_url', 'unitid']], university_urls_to_unitid_correspondence],
                                                         axis=0).drop_duplicates()
    university_urls_to_unitid_correspondence = university_urls_to_unitid_correspondence.dropna()
    linkedin_matches = pd.merge(linkedin_matches, university_urls_to_unitid_correspondence, on='school_url', how='left').drop_duplicates(subset=['member_id', 'unitid'])

    if debug_mode and linkedin_matches['unitid'].isnull().any():
        print("Stopping execution for review. Some linkedin matches have a missing unitid, this should not happen")
        pdb.set_trace()

    return linkedin_matches





def match_education_file_to_universities(
    edu_df, universities_adopted_facebook, university_urls_keep=None, university_urls_remove=None, score_cutoff=85
):
    '''
    Selects only those people that attended universities that are in the universities_adopted_facebook dataset.
    '''
    if university_urls_keep is None:
        university_urls_keep = []
    if university_urls_remove is None:
        university_urls_remove = []

    linkedin_matches = edu_df.copy()
    if university_urls_remove:
        linkedin_matches = linkedin_matches[~linkedin_matches['school_url'].isin(university_urls_remove)]
    linkedin_matches['keep'] = linkedin_matches['school_url'].isin(university_urls_keep)
    linkedin_matches['title'] = linkedin_matches['title'].str.lower().str.strip()
    

    unique_title_school_url = linkedin_matches[['title', 'school_url']].drop_duplicates()
    unique_title_school_url = clean_university_list(unique_title_school_url, 'title')
    universities_adopted_facebook = clean_university_list(universities_adopted_facebook, 'instnm')


    if debug_mode: print("Matching universities by tags and clean name with score cutoff:", score_cutoff)
    url_matches = match_by_tags_and_clean_name(
        universities_adopted_facebook, unique_title_school_url, score_cutoff=score_cutoff
    )
    if debug_mode: print("Matching done")

    for url in url_matches.school_url.unique():
        if url not in university_urls_keep:
            university_urls_keep.append(url)

    linkedin_matches.loc[linkedin_matches['school_url'].isin(url_matches['school_url']), 'keep'] = True

    for url in linkedin_matches[~linkedin_matches['keep']].school_url.unique():
        if url not in university_urls_keep and url not in university_urls_remove:
            university_urls_remove.append(url)

    
    linkedin_matches_to_return = linkedin_matches[linkedin_matches['keep']].drop_duplicates(subset=['member_id', 'school_url','date_from', 'date_to'])

    if debug_mode: print("Total matches to return in this pass (before adding unitid):", linkedin_matches_to_return.shape[0])
    linkedin_matches_to_return = matching_add_university_unitid(linkedin_matches_to_return, url_matches)
    if debug_mode: print("Total matches to return in this pass (after adding unitid):", linkedin_matches_to_return.shape[0])
    return linkedin_matches_to_return, university_urls_keep, university_urls_remove




def match_by_tags_and_clean_name(universities_adopted_facebook, linkedin_matches, score_cutoff=70):
    """
    Match rows between universities_adopted_facebook and linkedin_matches based on:
      (a) Exact match on 'tags' (as tuple or string)
      (b) Levenshtein similarity on 'clean_name' >= score_cutoff
      (c) Keep only the best match for each linkedin_matches row
    Returns a DataFrame with the best matches.
    """

    global debug_mode , debug_var_time_elapsed_matching

    if debug_mode: print("\tmatching: preparing data")
    # Ensure tags are comparable (convert lists to tuples or strings)
    ua = universities_adopted_facebook.copy()
    lm = linkedin_matches.copy()
    ua['tags_str'] = ua['tags'].apply(lambda x: ','.join(sorted(map(str, x))))
    lm['tags_str'] = lm['tags'].apply(lambda x: ','.join(sorted(map(str, x))))

    if debug_mode: print("\tmatching: filtering to rows with matched tags")
    # Filter to only rows with matching tags
    merged = lm.merge(ua, on='tags_str', suffixes=('_lm', '_ua'))

    start_time = datetime.datetime.now()
    
    if debug_mode: print("\tmatching: computing levenshtein distance")
    # Compute Levenshtein similarity for clean_name
    # Use rapidfuzz for much faster Levenshtein similarity calculation

    merged['lev_score'] = [
        Levenshtein.normalized_similarity(str(lm), str(ua)) * 100
        for lm, ua in zip(merged['clean_name_lm'], merged['clean_name_ua'])
    ]

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time    
    debug_var_time_elapsed_matching += elapsed.total_seconds()  # Convert to seconds
    if debug_mode: print(f"\tmatching: time elapsed {str(elapsed).split('.')[0]} (hh:mm:ss)")

    if debug_mode: print("\tmatching: filtering based on cutoff")
    # Filter by score_cutoff
    merged = merged[merged['lev_score'] >= score_cutoff]

    if debug_mode: print("\tmatching: getting the best matches")
    # For each linkedin_matches row, keep only the best match (highest lev_score)
    merged = merged.sort_values('lev_score', ascending=False)
    best_matches = merged.groupby(['clean_name_lm'], as_index=False).first()

    return best_matches

# Example usage:
# best_matches = match_by_tags_and_clean_name(universities_adopted_facebook, linkedin_matches, score_cutoff=70)





def process_education_relation_file(csv_path):
    '''
    Gets the education file to keep only those that match to the right universities and years of study,
    to then develop a dataset of their linkedin profiles
    '''
    global debug_mode , debug_var_total_rows , debug_var_time_elapsed_matching

    chunk_size = 40000
    university_urls_keep = []
    university_urls_remove = []
    all_linkedin_matches = pd.DataFrame()

    if debug_mode: print("Opening iterator")

    chunk_iter = pd.read_csv(
        csv_path,
        header=0,
        usecols=['member_id', 'title', 'date_from', 'date_to', 'school_url'],
        dtype={'member_id': str, 'title': str, 'date_from': str, 'date_to': str, 'school_url': str},
        chunksize=chunk_size,
        low_memory=False,
    )

    pickle_path = csv_path.replace('.csv', '_linkedin_matches_processed.pkl')
    processed_row_count = 0
    included_row_count = 0

    for chunk_data in chunk_iter:
        if debug_mode: print("Reading chunk")
        edu_df = read_education_file_filter_to_school_years(chunk_data)
        if debug_mode: print("Chunk read. Matching to universities")
        linkedin_matches, urls_keep, urls_remove = match_education_file_to_universities(
            edu_df, universities_adopted_facebook, university_urls_keep, university_urls_remove
        )

        if debug_mode: print("Updating loop variables")
        # Update university_urls_keep with new urls, avoiding duplicates
        university_urls_keep = list(set(university_urls_keep).union(urls_keep))
        university_urls_remove = list(set(university_urls_remove).union(urls_remove))

        all_linkedin_matches = pd.concat([all_linkedin_matches, linkedin_matches], ignore_index=True)
        processed_row_count += chunk_data.shape[0]
        included_row_count += linkedin_matches.shape[0]
        share_included_row_count = included_row_count / processed_row_count * 100 if processed_row_count > 0 else 0
        share_included_row_count = round(share_included_row_count, 1)

        print(
            f"\tProcessed {processed_row_count} rows, included {included_row_count} rows match criteria ({share_included_row_count}% inclusion rate)."
        )

        # Save progress every 10 loops or at the last chunk
        if processed_row_count % (chunk_size * 10) == 0 or chunk_data.shape[0] < chunk_size:
            all_linkedin_matches.to_pickle(pickle_path)
            print(f"Saved progress to {pickle_path} at {processed_row_count} rows.")

        if debug_mode:
            debug_var_total_rows += chunk_data.shape[0]
            rows_per_second = round(debug_var_total_rows / debug_var_time_elapsed_matching, 1) if debug_var_time_elapsed_matching > 0 else 0
            print(f"STATS: Effective match rate {rows_per_second} rows/sec")

    # Final save
    all_linkedin_matches.to_pickle(pickle_path)
    print(f"Final save to {pickle_path} with {all_linkedin_matches.shape[0]} rows.")







import concurrent.futures

#parameters
run_concurrent_files = False # Set to True to run in parallel, False is necessary fro debugging
run_threaded_extraction = True  
debug_mode = True # if true prints a bunch of messages
debug_var_total_rows = 0
debug_var_time_elapsed_matching = 0

university_urls_to_unitid_correspondence = pd.DataFrame() 

if __name__ == "__main__":
    csv_files = sorted(glob.glob('coresignal_member_education_*.csv'))
    print(f"Found {len(csv_files)} files: {csv_files}")

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
        # Set the number of parallel workers (adjust as needed)
        max_workers = min(5, len(csv_files))  # or os.cpu_count()

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for csv_path in csv_files:
                futures.append(executor.submit(process_education_relation_file, csv_path))
            for future in concurrent.futures.as_completed(futures):
                print(f"Finished processing: {future.result()}")

    else:
            print("Running in non-concurrent mode, one csv at a time")
            for csv_path in csv_files:
                process_education_relation_file(csv_path)
          