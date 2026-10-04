import sys; sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import pandas as pd
from jansahay.retrieval.semantic import SemanticRetriever
from jansahay.evaluation.metrics import hit_at_k, reciprocal_rank

df = pd.read_csv('data/processed/schemes_clean.csv')
q_df = pd.read_csv('data/processed/queries.csv')
sem = SemanticRetriever.load('data/artifacts/semantic', current_df=df)

# Check all English queries
english_queries = q_df[q_df['language']=='English']
hits5, total, mrrs = 0, 0, []
for _, row in english_queries.iterrows():
    strict = set(int(x) for x in str(row['relevant_scheme_ids']).split(',') if x.strip())
    sr = sem.search(row['user_query'], top_k=5)
    ids = [r['scheme_id'] for r in sr.get('results', [])]
    h5 = hit_at_k(ids, strict, 5)
    rr = reciprocal_rank(ids, strict)
    print(f"  [{h5:.0f}] {row['user_query'][:55]}")
    print(f"        Relevant: {strict} | Retrieved: {ids[:3]}")
    hits5 += h5
    mrrs.append(rr)
    total += 1

print(f"\nEnglish Hit@5: {hits5}/{total} = {hits5/total:.3f}")
print(f"English MRR:   {sum(mrrs)/len(mrrs):.3f}")
