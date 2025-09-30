import pandas as pd
from fuzzywuzzy import fuzz
from rapidfuzz import process as rapid_process
from collections import Counter
from rapidfuzz.distance import Levenshtein
import datetime


class university_name_matcher:
    def __init__(self):
        global debug_mode, debug_var_time_elapsed_matching
        self.top_10_tags = None
        self.universities_adopted_facebook = None
        debug_mode = False
        debug_var_time_elapsed_matching = 0.0

    def load_university_data(self):
        '''
        Load the university data from the Stata file.
        '''
        self.universities_adopted_facebook = pd.read_stata('universities_geo_for_jorge.dta')
        # Remove all universities with names that repeat more than once as it would not be possible to match
        self.universities_adopted_facebook = self.universities_adopted_facebook[~self.universities_adopted_facebook['instnm'].duplicated(keep=False)]

        return self.universities_adopted_facebook

        # Load the dataset
  

    def common_tags_to_remove(self):
        # Split 'instnm' into words and count their incidence>q
        words = self.universities_adopted_facebook['instnm'].str.lower().str.split().explode()
        word_counts = Counter(words)
        word_counts_df = pd.DataFrame(word_counts.most_common(), columns=['word', 'count'])
        word_counts_df = word_counts_df[word_counts_df['word'].str.len() > 3]
        top_10_tags = word_counts_df.head(10)
        self.top_10_tags = top_10_tags  
        return top_10_tags



    def clean_university_list(self, uni_list, name_col):
        '''
        Cleans the university list by extracting tags from the institution names and creating a clean name without those tags.
        The tags are derived from the top 10 most common words in the institution names.
        The 'instnm' column is expected to contain the institution names.
        '''

        top_10_tags = self.top_10_tags

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

        try:
            uni_list['clean_name'] = uni_list.apply(
                lambda row: remove_tags(str(row[name_col]).lower() if pd.notnull(row[name_col]) else '', row['tags']),
                axis=1
            )
        except ValueError:

            print("VALUE ERROR -- IGNORING ")
            return None

        return uni_list



    def match_by_tags_and_clean_name(self,universities_adopted_facebook, linkedin_matches, score_cutoff=70):
        """
        Match rows between universities_adopted_facebook and linkedin_matches based on:
        (a) Exact match on 'tags' (as tuple or string)
        (b) Levenshtein similarity on 'clean_name' >= score_cutoff
        (c) Keep only the best match for each linkedin_matches row
        Returns a DataFrame with the best matches.
        """
        if universities_adopted_facebook is None or linkedin_matches is None:
            return pd.DataFrame()  # Return empty DataFrame if inputs are None


        debug_mode = True
        debug_var_time_elapsed_matching = 0

        if (self.universities_adopted_facebook is None):
            self.load_university_data()

 
        if (self.top_10_tags is None):
            self.common_tags_to_remove()
        #universities_adopted_facebook = self.clean_university_list(universities_adopted
        #top_10_tags = self.top_10_tags


        if debug_mode: print("\tmatching: preparing data")
        # Ensure tags are comparable (convert lists to tuples or strings)
        ua = self.clean_university_list(universities_adopted_facebook.copy(), 'instnm')
        lm = self.clean_university_list(linkedin_matches.copy(), 'title')

        if ua is None or lm is None:
            return pd.DataFrame()  # Return empty DataFrame if cleaning failed
        
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
        
        merged = merged[merged['lev_score'] >= score_cutoff]

        if debug_mode: print("\tmatching: getting the best matches")
       
        #Group by individual records and keep the best match
        best_matches = merged.sort_values('lev_score', ascending=False).groupby(['clean_name_lm'], as_index=False).first()

        return best_matches
