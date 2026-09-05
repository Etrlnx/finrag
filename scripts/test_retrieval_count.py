from finrag.pipeline import load_production_pipeline
pipeline = load_production_pipeline()

question = 'What were Apple total net sales for fiscal 2025 versus fiscal 2024?'
docs = pipeline.retriever.invoke(question)
print(f'Retrieved {len(docs)} documents')
for i, d in enumerate(docs):
    ticker = d.metadata.get('ticker')
    section = d.metadata.get('section')
    is_table = d.metadata.get('is_table')
    print(f'  {i+1}. [{ticker}] {section} table={is_table}')