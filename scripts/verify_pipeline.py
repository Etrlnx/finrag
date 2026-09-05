from finrag.pipeline import load_production_pipeline
pipeline = load_production_pipeline()
print('Pipeline loaded successfully')
print('Vector store:', pipeline.vector_store.index.ntotal)
docs = pipeline.retriever.invoke('What were Apple total net sales for fiscal 2025 versus fiscal 2024?')
print(f'Retrieved {len(docs)} docs')
for i, d in enumerate(docs[:3]):
    ticker = d.metadata.get('ticker')
    section = d.metadata.get('section')
    is_table = d.metadata.get('is_table')
    print(f'  {i+1}. [{ticker}] {section} table={is_table}')