from finrag.data import load_all_filings, split_documents_with_strategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json

emb = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})

print('Loading text-only index...')
text_vs = FAISS.load_local('data/vector_stores/phase6_text_only', emb, allow_dangerous_deserialization=True)
print('Loaded text-only')

print('Loading table-aware index...')
table_vs = FAISS.load_local('data/vector_stores/phase6_tables_bge_base', emb, allow_dangerous_deserialization=True)
print('Loaded table-aware')

eval_set = json.load(open('eval/eval_set.json'))
q = next(item for item in eval_set if item['id'] == 'eval_36')
print('Test question:', q['question'])

text_retriever = text_vs.as_retriever(search_kwargs={'k': 5})
table_retriever = table_vs.as_retriever(search_kwargs={'k': 5})

import time
t0 = time.time()
text_docs = text_retriever.invoke(q['question'])
print(f'Text-only: {len(text_docs)} docs in {(time.time()-t0)*1000:.1f}ms')
for d in text_docs:
    print(f'  [{d.metadata.get("ticker")}] {d.metadata.get("section")} table={d.metadata.get("is_table")}')

t0 = time.time()
table_docs = table_retriever.invoke(q['question'])
print(f'Table-aware: {len(table_docs)} docs in {(time.time()-t0)*1000:.1f}ms')
for d in table_docs:
    print(f'  [{d.metadata.get("ticker")}] {d.metadata.get("section")} table={d.metadata.get("is_table")}')