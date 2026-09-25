import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re

os.makedirs("data_pipeline", exist_ok=True)

print("scraping books.toscrape.com...")

BASE = "http://books.toscrape.com/"
cats_to_scrape = ["Travel", "Mystery", "Historical Fiction"] # 3 cats

def get_category_url(name):
    r = requests.get(BASE)
    soup = BeautifulSoup(r.text, 'html.parser')
    links = soup.select(".side_categories ul li ul li a")
    for a in links:
        if name.lower() in a.text.strip().lower():
            return BASE + a['href']
    return None

all_books = []
rating_map = {"One":1, "Two":2, "Three":3, "Four":4, "Five":5}

for cat_name in cats_to_scrape:
    cat_url = get_category_url(cat_name)
    print(f"scraping {cat_name} -> {cat_url}")
    url = cat_url
    count = 0
    while url and count < 30:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        books = soup.select("article.product_pod")
        for b in books:
            title = b.h3.a['title']
            price_text = b.select_one(".price_color").text
            stars_text = b.select_one(".star-rating")['class'][1]
            avail_text = b.select_one(".instock.availability").text.strip()
            all_books.append({
                "title": title,
                "price_text": price_text,
                "star_text": stars_text,
                "availability_text": avail_text,
                "category": cat_name
            })
            count += 1
        next_btn = soup.select_one("li.next a")
        if next_btn:
            next_href = next_btn['href']
            base_cat = cat_url.rsplit('/',1)[0] + "/"
            url = base_cat + next_href
        else:
            url = None
    print(f"got {count} from {cat_name}")

print(f"total scraped {len(all_books)}")
df = pd.DataFrame(all_books)


print("\n--- cleaning ---")
def clean_price(p):
    try:
        num = re.sub(r'[^0-9.]', '', p)
        return float(num)
    except:
        return None

df['price_gbp'] = df['price_text'].apply(clean_price)
df['rating'] = df['star_text'].map(rating_map)
df['in_stock'] = df['availability_text'].apply(lambda x: True if 'In stock' in x else False)

print("nulls before", df.isnull().sum().to_dict())
median_price = df['price_gbp'].median()
df['price_gbp'] = df['price_gbp'].fillna(median_price)
median_rating = df['rating'].median()
df['rating'] = df['rating'].fillna(median_rating)
df = df.dropna(subset=['title'])
print("after clean shape", df.shape)

FIXED_RATE = 105.50
df['price_inr'] = (df['price_gbp'] * FIXED_RATE).round(2)
print(f"converted using fixed rate {FIXED_RATE}")

df_clean = df[['title','price_gbp','price_inr','rating','in_stock','category']]

conn = sqlite3.connect("data_pipeline/books.db")
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS books")
cur.execute("DROP TABLE IF EXISTS categories")
cur.execute("CREATE TABLE categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)")
cur.execute("CREATE TABLE books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL, rating INTEGER, in_stock INTEGER, category_id INTEGER, FOREIGN KEY(category_id) REFERENCES categories(category_id))")

cats = df_clean['category'].unique()
for i, cname in enumerate(cats, start=1):
    cur.execute("INSERT OR IGNORE INTO categories(category_id, category_name) VALUES (?,?)", (i, cname))

cat_map = {row[1]:row[0] for row in cur.execute("SELECT category_id, category_name FROM categories").fetchall()}

for idx, row in df_clean.iterrows():
    cur.execute("INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id) VALUES (?,?,?,?,?,?)",
                (row['title'], row['price_gbp'], row['price_inr'], int(row['rating']), int(row['in_stock']), cat_map[row['category']]))

conn.commit()
print("db created with 2 tables PK FK")

queries = {
    "q1_select_where": "SELECT title, price_gbp FROM books WHERE price_gbp > 30;",
    "q2_order_by_limit": "SELECT title, rating FROM books ORDER BY rating DESC LIMIT 10;",
    "q3_distinct": "SELECT DISTINCT category_name FROM categories;",
    "q4_in_between": "SELECT title, price_gbp FROM books WHERE rating IN (4,5) AND price_gbp BETWEEN 20 AND 50;",
    "q5_join": "SELECT c.category_name, b.title, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id = c.category_id ORDER BY c.category_name, b.rating DESC LIMIT 10;"
}

for name, q in queries.items():
    print(f"\n--- {name} ---")
    print(q)
    cur.execute(q)
    rows = cur.fetchall()
    print(rows[:5])

print("\n--- pd.read_sql join ---")
join_sql = "SELECT c.category_name, b.title, b.rating, b.price_gbp FROM books b JOIN categories c ON b.category_id = c.category_id"
df_sql = pd.read_sql(join_sql, conn)
print(df_sql.head())

print("\n--- pd.merge in-memory no SQL ---")
df_books_mem = pd.read_sql("SELECT * FROM books", conn)
df_cats_mem = pd.read_sql("SELECT * FROM categories", conn)
df_merge = pd.merge(df_books_mem, df_cats_mem, left_on='category_id', right_on='category_id')[['category_name','title','rating','price_gbp']]
print(df_merge.head())

print("\nmatch check:", df_sql.shape == df_merge.shape)

conn.close()
print("\ndone")
