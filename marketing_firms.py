# Is Firm Facebook-Advertisable?
#
# Goal: Tag companies founded/owned by entrepreneurs as likely Facebook/Instagram
# advertisers -- i.e., B2C businesses whose product or service is visual or
# experiential and can be naturally promoted through a Facebook Page.
#
# What qualifies:
#   - Visual products: fashion, food, beauty, art, photography, home decor
#   - Consumer experiences: restaurants, cafes, fitness studios, salons, events, travel
#   - Consumer-facing digital brands: e-commerce, apps, media, subscription boxes
#   - Local services people choose with emotion: wedding planners, photographers, gyms
#
# What does NOT qualify:
#   - B2B/trade services (construction contractors, HVAC, trucking, accounting)
#   - Legal / medical professional practices
#   - Industrial / wholesale businesses
#
# Approach: Word2Vec embeddings (Google News) + logistic regression trained on GPT-4o labels,
# producing a 0-1 confidence score (fb_ad_prob) and a 5-level categorical label.
#
# Pipeline:
#   1. Load all_experience data
#   2. Filter to entrepreneurs (founder/owner flag)
#   3. Sample company names and get GPT-4o binary labels
#   4. Encode names with Word2Vec (Google News) and train logistic regression
#   5. Evaluate on hold-out set
#   6. Score all unique company names
#   7. Sample validation set and compare Word2Vec model vs GPT-4o
#   8. Apply manual overrides
#   9. Sanity check top/bottom companies
#   10. Save results

import os
import json
import time
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from gensim.models import KeyedVectors
import openai

warnings.filterwarnings('ignore')

# -- Constants -----------------------------------------------------------------
DATA_PATH         = '/shared/share_scp/coresignal/all_experience_AnalysisFile_latest.pkl'
OUT_ENTREPRENEURS = '/shared/share_scp/coresignal/entrepreneurs_marketing_oriented_latest.pkl'
OUT_SCORES        = '/shared/share_scp/coresignal/company_marketing_scores_latest.pkl'
OUT_SAMPLE_2019   = '/shared/share_scp/coresignal/sample_5000_companies_2019.csv'
WORD2VEC_PATH     = '/shared/share_scp/coresignal/GoogleNews-vectors-negative300.bin.gz'

LABEL_ORDER = ['highly likely', 'likely', 'unclear', 'unlikely', 'highly unlikely']

LABEL_SYSTEM_BINARY = """\
You are a business classifier.  For each company name, decide whether this company
would be likely to run paid ads on Facebook or Instagram.

Answer "yes" for visual / experiential B2C businesses:
  retail shops, beauty salons, gyms/yoga studios, restaurants, bakeries, cafes,
  fashion & apparel brands, photographers, florists, event planners, home decor
  brands, travel agencies, pet groomers, children's activity centers, wellness
  coaches, art studios, entertainment venues, e-commerce shops, candle companies.

Answer "no" for B2B, industrial, or professional-services businesses:
  plumbers, HVAC contractors, freight/trucking, law firms, accounting firms,
  IT consulting, construction companies, manufacturers, wholesale distributors,
  staffing agencies, financial advisors, medical/dental practices.

When truly ambiguous, prefer "no".

Respond ONLY with a JSON object mapping each name exactly as given to "yes" or "no".
No extra keys, no commentary."""

LABEL_SYSTEM_FIVEPOINT = """\
You are a business analyst helping classify company names.

The task: given a list of company names, decide how likely each company is to run
Facebook / Instagram paid advertising -- i.e. does it sell a visual or experiential
consumer product or service that would make a natural Facebook Page?

Examples that are POSITIVE (run FB ads):
  fashion boutiques, beauty salons, restaurants, bakeries, gyms/yoga studios,
  photographers, florists, event planners, home decor brands, travel agencies,
  e-commerce shops, candle companies, pet groomers, children's activity centers.

Examples that are NEGATIVE (unlikely to run FB ads):
  plumbers, HVAC contractors, freight/trucking companies, law firms, accounting
  firms, B2B software companies, medical practices, industrial manufacturers,
  janitorial services, funeral homes.

Assign each company EXACTLY one of these five labels (use the exact string):
  highly likely | likely | unclear | unlikely | highly unlikely

Return ONLY a JSON array of objects with keys "company_name" and "label".
No explanation, no markdown fences, just the raw JSON array."""


# -- Step 1: Load data ---------------------------------------------------------
def load_data(pkl_path=DATA_PATH):
    print('Loading all_experience...', flush=True)
    all_experience = pd.read_pickle(pkl_path)
    print(f'Loaded {len(all_experience):,} rows, {all_experience.shape[1]} columns')
    print('Columns:', list(all_experience.columns))
    return all_experience


# -- Step 2: Filter to entrepreneurs -------------------------------------------
def filter_entrepreneurs(all_experience):
    if 'is_founder_or_owner' in all_experience.columns:
        entrepreneurs = all_experience[all_experience['is_founder_or_owner'] == True].copy()
    elif 'is_founder_only' in all_experience.columns and 'is_owner_only' in all_experience.columns:
        entrepreneurs = all_experience[
            all_experience['is_founder_only'] | all_experience['is_owner_only']
        ].copy()
    else:
        raise ValueError('No founder/owner flag found. Check column names above.')

    print(f'Total entrepreneur rows     : {len(entrepreneurs):,}')
    print(f'Unique company names        : {entrepreneurs["company_name"].nunique():,}')
    print(f'Rows with null company_name : {entrepreneurs["company_name"].isna().sum():,}')

    unique_companies = (
        entrepreneurs['company_name']
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != '']
        .unique()
    )
    print(f'Unique non-null company names to score: {len(unique_companies):,}')
    return entrepreneurs, unique_companies


# -- Step 2b: Sample 2019 companies (for outreach list) ------------------------
def sample_companies_2019(entrepreneurs, out_csv=OUT_SAMPLE_2019, n=5000):
    entrepreneurs_2019 = entrepreneurs[entrepreneurs['job_from'] == 2019].copy()
    print(f"Entrepreneurs founded in 2019: {len(entrepreneurs_2019):,}")

    companies_2019 = (
        entrepreneurs_2019[['company_name', 'fb_ad_prob', 'fb_ad_likelihood', 'job_from']]
        .dropna(subset=['company_name'])
        .drop_duplicates(subset=['company_name'])
    )
    print(f"Unique companies founded in 2019: {len(companies_2019):,}")

    n_sample = min(n, len(companies_2019))
    sample_2019 = companies_2019.sample(n=n_sample, random_state=42).reset_index(drop=True)
    sample_2019.to_csv(out_csv, index=False)
    print(f"Saved {len(sample_2019):,} companies to {out_csv}")
    return sample_2019


# -- Step 3: Get GPT-4o binary labels for training set -------------------------
def get_gpt_training_labels(unique_companies, client, n_sample=1000, batch_size=50):
    rng = np.random.default_rng(42)
    sample_names = rng.choice(unique_companies, size=n_sample, replace=False).tolist()

    batches = [sample_names[i:i+batch_size] for i in range(0, len(sample_names), batch_size)]
    print(f"Sending {len(sample_names):,} names to GPT-4o in {len(batches)} batches...")

    gpt_labels: dict = {}
    for idx, batch in enumerate(batches):
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {'role': 'system', 'content': LABEL_SYSTEM_BINARY},
                        {'role': 'user',   'content': json.dumps(batch)},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                chunk = json.loads(resp.choices[0].message.content.strip())
                gpt_labels.update(chunk)
                print(f"  Batch {idx+1}/{len(batches)}: {len(chunk)} labels")
                break
            except Exception as e:
                print(f"  Batch {idx+1} attempt {attempt+1} failed: {e}")
                time.sleep(3)

    gpt_labels = {str(k).strip(): str(v).strip().lower() for k, v in gpt_labels.items()}
    yes_n = sum(1 for v in gpt_labels.values() if v == 'yes')
    no_n  = sum(1 for v in gpt_labels.values() if v == 'no')
    print(f"\nLabelled: {len(gpt_labels):,}  |  yes={yes_n:,}  no={no_n:,}")
    return sample_names, gpt_labels


# -- Step 4: Train Word2Vec + logistic regression ------------------------------
def _embed_names(names, wv):
    """Average Word2Vec vectors for each company name; zeros for OOV names."""
    rows = []
    for name in names:
        words = str(name).split()
        vecs  = [wv[w] for w in words if w in wv]
        rows.append(np.mean(vecs, axis=0) if vecs else np.zeros(wv.vector_size))
    return np.vstack(rows)


def train_word2vec_classifier(sample_names, gpt_labels):
    print(f"Loading Word2Vec model from {WORD2VEC_PATH} ...")
    wv = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
    print("Word2Vec model loaded.")

    train_names = [n for n in sample_names if gpt_labels.get(n) in ('yes', 'no')]
    train_y     = np.array([1 if gpt_labels[n] == 'yes' else 0 for n in train_names])
    print(f"Training examples: {len(train_names):,}  (yes={train_y.sum():,}  no={(1-train_y).sum():,})")

    print("Encoding training names with Word2Vec...")
    train_X = _embed_names(train_names, wv)

    print("Training logistic regression (5-fold CV)...")
    clf    = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    cv_auc = cross_val_score(clf, train_X, train_y, cv=5, scoring='roc_auc')
    print(f"  CV AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

    clf.fit(train_X, train_y)
    print("Model trained and ready.")
    return wv, clf, cv_auc


# -- Step 5: Evaluate on hold-out set ------------------------------------------
def evaluate_holdout(unique_companies, sample_names, gpt_labels, wv, clf, client,
                     n_holdout=200, batch_size=50):
    train_set     = set(sample_names)
    holdout_pool  = [n for n in unique_companies.tolist() if n not in train_set]
    rng           = np.random.default_rng(42)
    holdout_names = rng.choice(holdout_pool, size=n_holdout, replace=False).tolist()

    print(f"Sending {n_holdout} fresh names to GPT-4o for hold-out labels...")
    holdout_labels: dict = {}
    ho_batches = [holdout_names[i:i+batch_size] for i in range(0, n_holdout, batch_size)]
    for idx, batch in enumerate(ho_batches):
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {'role': 'system', 'content': LABEL_SYSTEM_BINARY},
                        {'role': 'user',   'content': json.dumps(batch)},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                chunk = json.loads(resp.choices[0].message.content.strip())
                holdout_labels.update(chunk)
                print(f"  Batch {idx+1}/{len(ho_batches)}: {len(chunk)} labels")
                break
            except Exception as e:
                print(f"  Batch {idx+1} attempt {attempt+1} failed: {e}")
                time.sleep(3)

    holdout_labels = {str(k).strip(): str(v).strip().lower() for k, v in holdout_labels.items()}
    clean_ho = [(n, holdout_labels[n]) for n in holdout_names if holdout_labels.get(n) in ('yes', 'no')]
    ho_names, ho_true = zip(*clean_ho)
    ho_y     = np.array([1 if v == 'yes' else 0 for v in ho_true])
    ho_X     = _embed_names(list(ho_names), wv)
    ho_probs = clf.predict_proba(ho_X)[:, 1]
    ho_auc   = roc_auc_score(ho_y, ho_probs)
    print(f"\nHold-out test  (n={len(ho_y)}, yes={ho_y.sum()}, no={(1-ho_y).sum()})")
    print(f"  AUC (out-of-sample): {ho_auc:.3f}")


# -- Step 6: Score all unique company names ------------------------------------
def prob_to_label(p):
    if   p >= 0.80: return 'highly likely'
    elif p >= 0.65: return 'likely'
    elif p >= 0.35: return 'unclear'
    elif p >= 0.20: return 'unlikely'
    else:           return 'highly unlikely'


def score_all_companies(unique_companies, wv, clf):
    print(f"Encoding {len(unique_companies):,} unique company names with Word2Vec...")
    all_X = _embed_names(unique_companies.tolist(), wv)
    probs = clf.predict_proba(all_X)[:, 1]

    company_scores = pd.DataFrame({
        'company_name'     : unique_companies,
        'fb_ad_prob'       : probs,
        'fb_ad_likelihood' : pd.Categorical(
                                 [prob_to_label(p) for p in probs],
                                 categories=LABEL_ORDER, ordered=True
                             ),
    })

    print("\nDistribution of fb_ad_likelihood:")
    for lbl in LABEL_ORDER:
        n = (company_scores['fb_ad_likelihood'] == lbl).sum()
        print(f"  {lbl:<20}: {n:8,}  ({n/len(company_scores):.1%})")
    return company_scores


# -- Step 6b: Merge scores onto entrepreneur rows ------------------------------
def merge_scores_to_entrepreneurs(entrepreneurs, company_scores):
    _drop = [c for c in ['raw_score', 'confidence', 'fb_ad_likelihood',
                         'fb_ad_likelihood_final', 'matched_keywords', 'fb_ad_prob']
             if c in entrepreneurs.columns]
    if _drop:
        entrepreneurs = entrepreneurs.drop(columns=_drop)

    entrepreneurs = entrepreneurs.merge(
        company_scores[['company_name', 'fb_ad_prob', 'fb_ad_likelihood']],
        on='company_name',
        how='left'
    )
    print(f"Entrepreneurs with a score: {entrepreneurs['fb_ad_prob'].notna().sum():,}")
    print("\nDistribution of fb_ad_likelihood across entrepreneur rows:")
    for lbl in LABEL_ORDER:
        n = (entrepreneurs['fb_ad_likelihood'] == lbl).sum()
        print(f"  {lbl:<20}: {n:8,}  ({n/len(entrepreneurs):.1%})")
    return entrepreneurs


# -- Step 7: Sample validation set ---------------------------------------------
def sample_validation_set(company_scores, n_per_label=40, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    frames = []
    for label in LABEL_ORDER:
        subset = company_scores[company_scores['fb_ad_likelihood'] == label]
        n = min(n_per_label, len(subset))
        if n > 0:
            frames.append(subset.sample(n=n, random_state=seed))
            print(f"  {label:<20}: sampled {n} of {len(subset):,}")

    validation_sample = (
        pd.concat(frames)
        .sample(frac=1, random_state=99)
        .reset_index(drop=True)
    )
    print(f"\nTotal validation sample: {len(validation_sample)} rows")
    return validation_sample


# -- Step 7b: GPT-4o independent 5-point classification -----------------------
def classify_batch_fivepoint(names, client, batch_size=50, pause=1.0):
    results = {}
    batches = [names[i:i+batch_size] for i in range(0, len(names), batch_size)]
    for batch_idx, batch in enumerate(batches):
        numbered = '\n'.join(f'{i+1}. {n}' for i, n in enumerate(batch))
        user_msg = (
            f'Classify these {len(batch)} company names:\n\n{numbered}\n\n'
            'Return a JSON array: [{"company_name": "...", "label": "..."}, ...]'
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {'role': 'system', 'content': LABEL_SYSTEM_FIVEPOINT},
                        {'role': 'user',   'content': user_msg},
                    ],
                    temperature=0,
                    max_tokens=4096,
                )
                raw = resp.choices[0].message.content.strip()
                if raw.startswith('```'):
                    raw = '\n'.join(raw.split('\n')[1:])
                if raw.endswith('```'):
                    raw = '\n'.join(raw.split('\n')[:-1])
                parsed = json.loads(raw)
                for item in parsed:
                    name  = item.get('company_name', '').strip()
                    label = item.get('label', '').strip().lower()
                    if label not in LABEL_ORDER:
                        label = 'unclear'
                    results[name] = label
                print(f'  Batch {batch_idx+1}/{len(batches)}: {len(parsed)} labels received')
                break
            except Exception as e:
                print(f'  Batch {batch_idx+1} attempt {attempt+1} failed: {e}')
                time.sleep(3)
        time.sleep(pause)
    return results


def validate_with_gpt(validation_sample, client):
    names_to_classify = validation_sample['company_name'].tolist()
    print(f'Sending {len(names_to_classify)} names to GPT-4o...')
    gpt_labels = classify_batch_fivepoint(names_to_classify, client)

    validation_sample = validation_sample.copy()
    validation_sample['gpt_label'] = validation_sample['company_name'].map(gpt_labels)
    print(f'\nGPT labels received for {validation_sample["gpt_label"].notna().sum()} / {len(validation_sample)} names')
    print('\nGPT label distribution:')
    print(validation_sample['gpt_label'].value_counts().reindex(LABEL_ORDER).to_string())
    return validation_sample


# -- Step 7c: Compare Word2Vec model vs GPT-4o --------------------------------
def compare_labels(validation_sample):
    compare = validation_sample[['company_name', 'fb_ad_likelihood', 'gpt_label', 'fb_ad_prob']].copy()
    compare = compare[compare['gpt_label'].notna()]
    compare['agree'] = compare['fb_ad_likelihood'] == compare['gpt_label']

    n_total = len(compare)
    n_agree = compare['agree'].sum()
    n_wrong = n_total - n_agree
    pct_ok  = n_agree / n_total * 100

    print('=' * 60)
    print(f'  Total compared  : {n_total}')
    print(f'  Model agree     : {n_agree}  ({pct_ok:.1f}%)')
    print(f'  Disagreements   : {n_wrong}  ({100 - pct_ok:.1f}%)')
    print('=' * 60)

    wrong = compare[~compare['agree']].copy()
    if len(wrong) == 0:
        print('\nPerfect agreement -- no disagreements!')
    else:
        print(f'\n{"#":<4}  {"Company Name":<40}  {"Word2Vec Model":<18}  {"GPT-4o":<18}  P(yes)')
        print('-' * 95)
        for i, (_, row) in enumerate(wrong.iterrows(), 1):
            print(f'{i:<4}  {str(row["company_name"]):<40}  '
                  f'{row["fb_ad_likelihood"]:<18}  {row["gpt_label"]:<18}  {row["fb_ad_prob"]:.3f}')

        print()
        print('Disagreement breakdown by direction:')
        direction = wrong.groupby(['fb_ad_likelihood', 'gpt_label']).size().rename('count').reset_index()
        direction.columns = ['model_said', 'gpt_said', 'count']
        direction = direction.sort_values('count', ascending=False)
        print(direction.to_string(index=False))


# -- Step 8: Apply manual overrides --------------------------------------------
def apply_manual_overrides(company_scores, force_label):
    # force_label: dict mapping company_name -> correct label (one of LABEL_ORDER)
    # Example: {'Acme Photo Studio': 'highly likely', 'Smith Plumbing LLC': 'highly unlikely'}
    company_scores = company_scores.copy()
    company_scores['fb_ad_likelihood_final'] = company_scores['fb_ad_likelihood'].copy()

    for name, label in force_label.items():
        if label not in LABEL_ORDER:
            print(f'WARNING: invalid label "{label}" for "{name}" -- skipping')
            continue
        mask = company_scores['company_name'] == name
        if mask.sum() == 0:
            print(f'WARNING: "{name}" not found in company_scores -- skipping')
        else:
            company_scores.loc[mask, 'fb_ad_likelihood_final'] = label

    changed = (company_scores['fb_ad_likelihood_final'] != company_scores['fb_ad_likelihood']).sum()
    print(f'Overrides applied: {changed}')
    print('\nUpdated distribution:')
    counts = company_scores['fb_ad_likelihood_final'].value_counts().reindex(LABEL_ORDER)
    for label, n in counts.items():
        pct = n / len(company_scores) * 100
        print(f'  {label:<20}: {n:8,}  ({pct:5.1f}%)')
    return company_scores


# -- Step 9: Sanity check top/bottom companies ---------------------------------
def sanity_check(company_scores, n=30):
    show_cols = ['company_name', 'fb_ad_likelihood', 'fb_ad_prob']

    print(f'=== TOP {n} most confidently HIGHLY LIKELY companies ===')
    top_likely = (
        company_scores[company_scores['fb_ad_likelihood'] == 'highly likely']
        .sort_values('fb_ad_prob', ascending=False)
        .head(n)[show_cols]
        .reset_index(drop=True)
    )
    print(top_likely.to_string())

    print(f'\n=== TOP {n} most confidently HIGHLY UNLIKELY companies ===')
    top_unlikely = (
        company_scores[company_scores['fb_ad_likelihood'] == 'highly unlikely']
        .sort_values('fb_ad_prob', ascending=True)
        .head(n)[show_cols]
        .reset_index(drop=True)
    )
    print(top_unlikely.to_string())

    print('\n=== Sample of UNCLEAR companies (closest to 0.50 probability) ===')
    unclear = (
        company_scores[company_scores['fb_ad_likelihood'] == 'unclear']
        .assign(_dist=lambda d: (d['fb_ad_prob'] - 0.5).abs())
        .sort_values('_dist')
        .head(n)[show_cols]
        .reset_index(drop=True)
    )
    print(unclear.to_string())


# -- Step 10: Save results -----------------------------------------------------
def save_results(entrepreneurs, company_scores,
                 out_entrepreneurs=OUT_ENTREPRENEURS, out_scores=OUT_SCORES):
    _drop = [c for c in ['fb_ad_likelihood', 'fb_ad_prob'] if c in entrepreneurs.columns]
    if _drop:
        entrepreneurs = entrepreneurs.drop(columns=_drop)

    label_col = ('fb_ad_likelihood_final'
                 if 'fb_ad_likelihood_final' in company_scores.columns
                 else 'fb_ad_likelihood')

    entrepreneurs = entrepreneurs.merge(
        company_scores[['company_name', 'fb_ad_prob', label_col]]
        .rename(columns={label_col: 'fb_ad_likelihood'}),
        on='company_name',
        how='left'
    )

    print('Final distribution across entrepreneur rows:')
    counts = entrepreneurs['fb_ad_likelihood'].value_counts().reindex(LABEL_ORDER)
    for label, n in counts.items():
        pct = n / len(entrepreneurs) * 100
        print(f'  {label:<20}: {n:8,}  ({pct:5.1f}%)')

    entrepreneurs.to_pickle(out_entrepreneurs)
    print(f'\nSaved {len(entrepreneurs):,} rows to {out_entrepreneurs}')

    company_scores.to_pickle(out_scores)
    print(f'Saved company scores ({len(company_scores):,} unique names) to {out_scores}')
    return entrepreneurs


# -- Main ----------------------------------------------------------------------
def main(
    data_path=DATA_PATH,
    out_entrepreneurs=OUT_ENTREPRENEURS,
    out_scores=OUT_SCORES,
    out_sample_2019=OUT_SAMPLE_2019,
    n_training_labels=1000,
    n_holdout=200,
    n_per_label_validation=40,
    force_label=None,   # dict of {company_name: label} manual overrides
):
    if force_label is None:
        force_label = {}

    # OpenAI client
    client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    # 1. Load data
    all_experience = load_data(data_path)

    # 2. Filter to entrepreneurs
    entrepreneurs, unique_companies = filter_entrepreneurs(all_experience)

    # 2b. Save 2019 outreach sample
    sample_companies_2019(entrepreneurs, out_csv=out_sample_2019)

    # 3. GPT-4o training labels
    sample_names, gpt_labels = get_gpt_training_labels(
        unique_companies, client, n_sample=n_training_labels
    )

    # 4. Train Word2Vec + logistic regression
    wv, clf, cv_auc = train_word2vec_classifier(sample_names, gpt_labels)
    print(f"  Train CV AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

    # 5. Hold-out evaluation
    evaluate_holdout(
        unique_companies, sample_names, gpt_labels,
        wv, clf, client, n_holdout=n_holdout
    )

    # 6. Score all companies and merge onto entrepreneurs
    company_scores = score_all_companies(unique_companies, wv, clf)
    entrepreneurs  = merge_scores_to_entrepreneurs(entrepreneurs, company_scores)

    # 7. Validate against GPT-4o independent labels
    validation_sample = sample_validation_set(company_scores, n_per_label=n_per_label_validation)
    validation_sample = validate_with_gpt(validation_sample, client)
    compare_labels(validation_sample)

    # 8. Apply manual overrides
    company_scores = apply_manual_overrides(company_scores, force_label)

    # 9. Sanity check
    sanity_check(company_scores)

    # 10. Save
    entrepreneurs = save_results(
        entrepreneurs, company_scores,
        out_entrepreneurs=out_entrepreneurs,
        out_scores=out_scores,
    )

    return entrepreneurs, company_scores


if __name__ == '__main__':
    main()
