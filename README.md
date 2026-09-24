# Zepto-Data-AI-Platform-project
# Zepto Data AI Platform

Fixed Rate: 1 GBP = 105.50 INR

## Module 1 - Data Pipeline
- Scrape 3 categories: Travel, Mystery, Historical Fiction
- 72 books, clean price_gbp, rating, in_stock
- price_inr = price_gbp * 105.50
- DB: categories(PK) and books(FK), 2 tables
- 5 queries: avg/min/max, count by rating, count by category JOIN, top 10 expensive, rating>=4 and price<20
- Run: python data_pipeline/main.py
- Output: books.db, raw_books.csv, query_results.txt
- Git branch feature/data-pipeline -> 2 commits -> merge

## Module 2 - Analytics
- Data: sns.load_dataset('titanic') -> save titanic.csv
- Missing: age median, embarked mode, drop deck
- IQR outliers, mean/median/mode fare 32.2/14.4/8.05
- Crosstab survived vs sex, corr 6 cols, 4 charts with interpretation
- Standardization check std before 49.6 after 1.0
- Stratified 80/20, ColumnTransformer fit only train
- Classification: Logistic, DecisionTree, RandomForest + plot_tree, CM, accuracy, precision, recall, F1, ROC
- Imbalance: baseline vs balanced vs SMOTE train only
- Regression fare: MAE, RMSE, R2, AdjR2, residual plot
- Run: python analytics/analytics.py
- Output: titanic.csv, 8 pngs, best_pipeline.joblib

## Module 3 - Support Assistant
- 8 docs exact, all-MiniLM-L6-v2, ChromaDB
- Colab: EphemeralClient, Local: PersistentClient(path="support_assistant/chroma_db") + get_or_create_collection
- Prompt: ROLE, CONTEXT, TASK, FORMAT, LENGTH + Negative Constraint + Few-shot
- LangGraph 3 nodes: classify_intent, retrieve_and_answer, direct_answer + conditional edge
- Intent: keyword heuristic delivery, return, refund, membership, tracking, cancel, gift card, support hours
- Answer mock: Based on the retrieved context: <200 chars> + sources top3
- General: I can only answer questions about Zepto policies right now.
- Pydantic answer, sources, confidence + retry 3
- FastAPI POST /ask
- Dockerfile buildable
- Run: MOCK_LLM=1 python support_assistant/main.py
- Test: ask_zepto("What is return window for perishable?") -> doc_02 -> 24 hours

## Run All Colab
pip install requests beautifulsoup4 pandas lxml seaborn imbalanced-learn chromadb==0.4.24 sentence-transformers langgraph fastapi
rm -rf support_assistant/chroma_db
run each main.py
