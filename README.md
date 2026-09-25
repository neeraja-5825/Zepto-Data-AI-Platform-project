# Zepto Data AI Platform

## Module 1 - Data Pipeline
- Scrapes 3 categories from books.toscrape.com - Travel, Mystery, Historical Fiction
- Cleans price and rating, converts GBP to INR using fixed rate 105.50
- Creates SQLite DB with 2 tables: categories and books with PK FK
- Runs 5 SQL queries and does pd.merge check
- Run: python data_pipeline/main.py

## Module 2 - Analytics
- Uses titanic dataset from seaborn
- Handles missing values, finds outliers, does 4 charts
- Uses StandardScaler and stratified split
- Tries 3 models Logistic, DecisionTree, RandomForest
- Handles imbalance with balanced and SMOTE
- Does fare prediction with Linear Regression
- Saves best model as best_pipeline.joblib
- Run: python analytics/analytics.py

## Module 3 - Support Assistant
- 8 policy docs about delivery, return, refund etc
- Uses sentence-transformers and ChromaDB
- Uses LangGraph with intent classification
- FastAPI endpoint POST /ask
- Run: python support_assistant/main.py
