# This code predicts whether a job is in finance
# and adds a flag on that to the graduates_job dataset

import os
import json
import time
import pdb

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import openai

DATA_DIR = '/shared/share_scp/coresignal'
DATA_FILE = 'graduates_with_education_job_level_staging.pkl'


# ### 2.2.1 Sample company names and get ChatGPT finance labels

# **Pipeline:**
# 1. Sample 5,000 unique company names → `training_finance_names.csv`
# 2. Send to ChatGPT → two binary labels per company:
#    - `is_finance_broad`: finance broadly defined (retail banking, insurance, Wall St, etc.)
#    - `is_wallstreet`: Wall Street / investment banking / asset management specifically
#    → `training_finance_predictions.csv`
# 3. Repeat with a separate 5,000-name hold-out set → `test_finance_names.csv` / `test_finance_predictions.csv`
# 4. Compute Word2Vec embeddings for all unique company names
# 5. Train logistic regression on training labels; report AUC on train & test
# 6. Predict finance probability for every firm; merge back to `graduates_with_education_job_level`
# 7. Estimate share going to finance by college × graduation year


# ── Step 1: Sample 5,000 training names and 5,000 test names ────────────────
def sample_companies(graduates_with_education_job_level):
    
    if os.path.exists('training_finance_names.csv') and os.path.exists('test_finance_names.csv'):
        print("Training and test name files already exist — skipping sampling.")
        return
    
    all_companies = graduates_with_education_job_level['company_name'].dropna().unique()
    print(f"Total unique company names: {len(all_companies):,}")

    rng = np.random.default_rng(seed=42)
    shuffled = rng.permutation(all_companies)

    train_names = shuffled[:5000]
    test_names  = shuffled[5000:10000]

    pd.DataFrame({'company_name': train_names}).to_csv('training_finance_names.csv', index=False)
    pd.DataFrame({'company_name': test_names}).to_csv('test_finance_names.csv', index=False)
    print(f"Saved {len(train_names):,} training names → training_finance_names.csv")
    print(f"Saved {len(test_names):,} test names     → test_finance_names.csv")





def openai_estimate_finance_companies():
    if (os.path.exists('training_finance_predictions.csv') and
        os.path.exists('test_finance_predictions.csv')):
        print("Training and test prediction files already exist — skipping OpenAI calls.")
        return

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=120,        # 60-second hard timeout per request
    )

    SYSTEM_PROMPT = (
        "You are a business analyst. For each company name given, return ONLY a "
        "JSON array where each element is an object with three keys:\n"
        "  'company_name': the exact name as provided\n"
        "  'is_finance_broad': 1 if the company is in finance broadly defined "
        "(includes retail banking, commercial banking, investment banking, asset "
        "management, hedge funds, private equity, venture capital, insurance, "
        "financial advisory, fintech, accounting firms, credit agencies), else 0\n"
        "  'is_wallstreet': 1 if the company is specifically on Wall Street "
        "(investment banks, hedge funds, PE/VC, asset managers, trading firms), "
        "else 0\n"
        "Return no other text — only the JSON array."
    )

    def parse_json_response(text):
        """Strip markdown fences then parse JSON."""
        text = text.strip()
        if text.startswith("```"):
            # remove opening fence (```json or ```)
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return json.loads(text.strip())

    def classify_batch(names, client, max_retries=3):
        prompt = "Classify the following company names:\n" + json.dumps(list(names))
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0,
                )
                text = resp.choices[0].message.content.strip()
                return parse_json_response(text)
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(5 * (attempt + 1))
        return [{"company_name": n, "is_finance_broad": None, "is_wallstreet": None} for n in names]

    def get_predictions(csv_path, out_path, batch_size=20):
        """Send company names to ChatGPT and save predictions.
        Saves a .partial file incrementally so interrupted runs can resume."""
        if os.path.exists(out_path):
            print(f"  {out_path} already exists — skipping API calls.")
            return pd.read_csv(out_path)

        partial_path = out_path + ".partial"
        names = pd.read_csv(csv_path)['company_name'].tolist()

        # Resume from partial progress if available
        if os.path.exists(partial_path):
            done_df = pd.read_csv(partial_path)
            done_names = set(done_df['company_name'])
            names = [n for n in names if n not in done_names]
            results = done_df.to_dict('records')
            print(f"  Resuming: {len(done_df):,} already done, {len(names):,} remaining.")
        else:
            results = []

        for i in range(0, len(names), batch_size):
            batch = names[i:i+batch_size]
            print(f"  Classifying {i+1}–{i+len(batch)} of {len(names)}...", end=" ", flush=True)
            rows = classify_batch(batch, client)
            results.extend(rows)
            print("done")
            # save partial progress every 5 batches
            if (i // batch_size) % 5 == 0:
                pd.DataFrame(results).to_csv(partial_path, index=False)
            time.sleep(0.3)

        df = pd.DataFrame(results)
        df.to_csv(out_path, index=False)
        if os.path.exists(partial_path):
            os.remove(partial_path)
        print(f"  Saved {len(df):,} predictions → {out_path}")
        return df

    print("Getting ChatGPT predictions for TRAINING set...")
    df_train_pred = get_predictions('training_finance_names.csv', 'training_finance_predictions.csv')
    print(f"Training predictions shape: {df_train_pred.shape}")
    print(df_train_pred[['is_finance_broad','is_wallstreet']].value_counts())

    print("Getting ChatGPT predictions for TEST set...")
    df_test_pred = get_predictions('test_finance_names.csv', 'test_finance_predictions.csv')
    print(f"Test predictions shape: {df_test_pred.shape}")
    print(df_test_pred[['is_finance_broad','is_wallstreet']].value_counts())

    df_train_pred.sum()
    return df_train_pred, df_test_pred


### 2.2.2 Word2Vec embeddings for all company names
def compute_word2vec_embeddings(graduates_with_education_job_level):
    WORD2VEC_PATH = "/shared/share_scp/coresignal/GoogleNews-vectors-negative300.bin.gz"  # Update path if needed
    EMBED_CACHE = "company_name_embeddings.pkl"

    if not os.path.exists(WORD2VEC_PATH):
        from pathlib import Path
        import urllib.request

        url = "https://github.com/mmihaltz/word2vec-GoogleNews-vectors/raw/refs/heads/master/GoogleNews-vectors-negative300.bin.gz?download="
        path = Path(WORD2VEC_PATH)

        urllib.request.urlretrieve(url, path)


    def get_company_embedding(name, wv):
        words = str(name).split()
        vectors = [wv[word] for word in words if word in wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(wv.vector_size)
            # word2vec model and batch size not needed
        
    print("Loading cached embeddings...")
    
    if os.path.exists(EMBED_CACHE):
        embed_df = pd.read_pickle(EMBED_CACHE)
    else:    
        all_unique = graduates_with_education_job_level['company_name'].dropna().unique()
        print(f"Loading Word2Vec model from {WORD2VEC_PATH} ...")
        wv = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
        print(f"Computing Word2Vec embeddings for {len(all_unique):,} unique company names...")
        embs = np.vstack([get_company_embedding(name, wv) for name in all_unique])
        embed_df = pd.DataFrame(embs, index=all_unique)
        embed_df.index.name = "company_name"
        embed_df.to_pickle(EMBED_CACHE)
        print(f"Saved embeddings → {EMBED_CACHE}  shape: {embed_df.shape}")
    return embed_df

### 2.2.3 Train logistic regression on Word2Vec embeddings; evaluate AUC
def evaluate_logistic_regression(embed_df):
    def build_xy(pred_csv, embed_df, label_col):
        """Return feature matrix X and label vector y aligned on company_name."""
        preds = pd.read_csv(pred_csv).dropna(subset=[label_col])
        preds[label_col] = preds[label_col].astype(int)
        # keep only companies that have embeddings
        preds = preds[preds['company_name'].isin(embed_df.index)]
        X = embed_df.loc[preds['company_name']].values
        y = preds[label_col].values
        return X, y

    results = {}
    for label in ['is_finance_broad', 'is_wallstreet']:
        print(f"\n── Label: {label} ───────────────────────────────────────")
        X_train, y_train = build_xy('training_finance_predictions.csv', embed_df, label)
        X_test,  y_test  = build_xy('test_finance_predictions.csv',     embed_df, label)

        scaler = StandardScaler() 
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs', n_jobs=-1)
        clf.fit(X_train_s, y_train)

        auc_train = roc_auc_score(y_train, clf.predict_proba(X_train_s)[:,1])
        auc_test  = roc_auc_score(y_test,  clf.predict_proba(X_test_s)[:,1])
        print(f"  Train AUC : {auc_train:.4f}")
        print(f"  Test  AUC : {auc_test:.4f}")
        print(f"  Positive rate (train): {y_train.mean():.3%}")
        print(f"  Positive rate (test) : {y_test.mean():.3%}")

        results[label] = {"clf": clf, "scaler": scaler}
    return results




# ── Step 6: Predict finance probability for ALL unique company names ─────────
def predict_finance_probabilities(graduates_with_education_job_level, results, embed_df):
    all_unique = graduates_with_education_job_level['company_name'].dropna().unique()
    # Only predict for companies that are in the embedding index
    in_embed = [n for n in all_unique if n in embed_df.index]
    X_all = embed_df.loc[in_embed].values

    prob_rows = []
    for label, obj in results.items():
        scaler = obj['scaler']
        clf    = obj['clf']
        probs  = clf.predict_proba(scaler.transform(X_all))[:,1]
        prob_rows.append(pd.Series(probs, index=in_embed, name=f"prob_{label}"))

    company_finance_probs = pd.concat(prob_rows, axis=1).reset_index()
    company_finance_probs.rename(columns={'index': 'company_name'}, inplace=True)
    print(f"Finance probability table: {company_finance_probs.shape}")
    print(company_finance_probs.describe())

    # Merge back into graduates_with_education_job_level
    graduates_with_education_job_level = graduates_with_education_job_level.merge(
        company_finance_probs, on='company_name', how='left')
    print(f"\nColumns added: prob_is_finance_broad, prob_is_wallstreet")
    pdb.set_trace()
    print(graduates_with_education_job_level[['prob_is_finance_broad','prob_is_wallstreet']].describe())
    return graduates_with_education_job_level


# ### 2.2.4 Estimate share going to finance by college × graduation year
# Focus on jobs that start within years_after_grad years of graduation.
# For each (college, graduation year) cell compute:
#   (a) average prob_is_finance_broad across all early jobs — estimated probability a graduate goes into finance
#   (b) (a) × number of graduates that year — estimated count going into finance
def compute_finance_by_school_year(graduates_with_education_job_level, years_after_grad=2):
    # ── Jobs that start within years_after_grad years of graduation ──────────
    early_jobs = graduates_with_education_job_level[
        graduates_with_education_job_level['job_from'] <=
        graduates_with_education_job_level['year_to'] + years_after_grad
    ].copy()
    print(f"Early-career jobs (within {years_after_grad} yrs of graduation): {len(early_jobs):,}")

    # ── (a) Average finance probability per college × graduation year ─────────
    finance_by_school_year = (
        early_jobs
        .groupby(['unitid', 'year_to'], dropna=False)
        .agg(
            n_jobs                    = ('member_id', 'count'),
            n_graduates               = ('member_id', 'nunique'),
            avg_prob_finance_broad    = ('prob_is_finance_broad', 'mean'),
            avg_prob_wallstreet       = ('prob_is_wallstreet',    'mean'),
        )
        .reset_index()
    )

    # ── (b) Estimated count going into finance ────────────────────────────────
    finance_by_school_year['est_count_finance_broad'] = (
        finance_by_school_year['avg_prob_finance_broad'] *
        finance_by_school_year['n_graduates']
    )
    finance_by_school_year['est_count_wallstreet'] = (
        finance_by_school_year['avg_prob_wallstreet'] *
        finance_by_school_year['n_graduates']
    )

    print(f"\nfinance_by_school_year shape: {finance_by_school_year.shape}")
    finance_by_school_year.to_stata('finance_by_school_year.dta',  version=118)
    finance_by_school_year.to_csv('finance_by_school_year.csv', index=False) 
    print("Saved → finance_by_school_year.csv")
    return finance_by_school_year


def summarize_results(finance_by_school_year):
    # Quick summary
    print("\nOverall stats:")
    print(f"  Mean avg_prob_finance_broad : {finance_by_school_year['avg_prob_finance_broad'].mean():.3%}")
    print(f"  Mean avg_prob_wallstreet    : {finance_by_school_year['avg_prob_wallstreet'].mean():.3%}")
    print(f"  Total est. finance (broad)  : {finance_by_school_year['est_count_finance_broad'].sum():,.0f}")
    print(f"  Total est. Wall Street      : {finance_by_school_year['est_count_wallstreet'].sum():,.0f}")

    # Top schools by est. Wall Street count
    top_ws = (
        finance_by_school_year
        .groupby('unitid')['est_count_wallstreet'].sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    print("\nTop 15 schools by estimated Wall Street placements:")
    print(top_ws)


def main(
    data_dir=DATA_DIR,
    data_file=DATA_FILE,
    train_names_csv='training_finance_names.csv',
    train_preds_csv='training_finance_predictions.csv',
    test_names_csv='test_finance_names.csv',
    test_preds_csv='test_finance_predictions.csv',
    embed_cache='company_name_embeddings.pkl',
    # bert_model removed (was for BERT, not needed for word2vec)
    embed_batch_size=256,
    clf_batch_size=20,
    n_train_names=5000,
    n_test_names=5000,
    years_after_grad=2,
):
    # ── Load data ────────────────────────────────────────────────────────────
    os.chdir(data_dir)
    print(f"Loading {data_file}...")
    graduates_with_education_job_level = pd.read_pickle(data_file)
    print(f"Loaded {len(graduates_with_education_job_level):,} rows")

    # ── Step 1: Sample company names ─────────────────────────────────────────
    sample_companies(graduates_with_education_job_level)

    # ── Step 2: Get ChatGPT finance labels ───────────────────────────────────
    openai_estimate_finance_companies()

    # ── Step 3: Word2Vec embeddings ──────────────────────────────────────────
    embed_df = compute_word2vec_embeddings(graduates_with_education_job_level)

    # ── Step 4: Train logistic regression and evaluate ───────────────────────
    results = evaluate_logistic_regression(embed_df)

    # ── Step 5: Predict finance probabilities for all companies ──────────────
    graduates_with_education_job_level = predict_finance_probabilities(
        graduates_with_education_job_level, results, embed_df
    )


    # ── Step 6: Compute finance share by school × year ───────────────────────
    finance_by_school_year = compute_finance_by_school_year(
        graduates_with_education_job_level, years_after_grad=years_after_grad
    )

    finance_by_school_year.to_pickle('finance_by_school_year.pkl')

    graduates_with_education_job_level = graduates_with_education_job_level.merge(
        finance_by_school_year,
        on=['unitid','year_to'],
        how='left'
    )

    graduates_with_education_job_level.to_pickle(data_file)

    # ── Step 7: Summarize results ──────────────────────────────────────────
    summarize_results(finance_by_school_year)

    return graduates_with_education_job_level


if __name__ == '__main__':
    main()

