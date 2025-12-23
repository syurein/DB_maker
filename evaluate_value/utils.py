import os
import json
import re
import time
import random
import requests
import polars as pl  # 高速処理ライブラリ
import numpy as np
from scipy import stats
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from google import genai
from google.genai import types
from playwright.sync_api import sync_playwright
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
# --- 1. Vision AI (Gemini) ---
def extract_json(text: str):
    try:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match: text = match.group(1)
        start = text.find("{")
        if start == -1: return {}
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj
    except:
        return {}
class VisionAppraiser:
    def __init__(self,model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        

    def analyze_image(self, image: Image.Image) -> dict:
        if not client: return {"error": "API Key missing"}
        try:
            prompt = """
            この商品を特定し、Pythonのreモジュールで検索するための正規表現リストを作成してください。
            また、楽天市場で検索するための最適なキーワードも抽出してください。
            できるだけ多くヒットするよ追うなキーワードや正規表現リストを選んでください。
            
            【出力形式(JSON)】
            {
                "tentative_name": "商品名",
                "search_keyword": "楽天市場検索用単語",
                "search_queries": ["正規表現パターン1", "パターン2"],
                "ai_price_c": 3000
            }
            """
            response = client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return extract_json(response.text)
        except Exception as e:
            return {"error": str(e)}
        
# --- 2. Market Data Manager (Polars Adapter) ---
class MarketDataManager:
    def __init__(self, csv_path: str = "output.csv"):
        self.csv_path = csv_path
        self.df = self.load_data()

    def load_data(self):
        if not os.path.exists(self.csv_path):
            print(f"⚠️ {self.csv_path} not found.")
            return pl.DataFrame()
        
        try:
            # Polarsによる超高速読み込み
            df = pl.read_csv(self.csv_path, ignore_errors=True)
            
            # カラム名マッピング（日本語カラムへの対応）
            rename_dict = {"商品名": "product_name", "価格": "price", "URL": "item_url"}
            cols_to_rename = {k: v for k, v in rename_dict.items() if k in df.columns}
            df = df.rename(cols_to_rename)

            # 価格クレンジング: 文字列を数値に変換 (Polars Expression)
            if "price" in df.columns:
                df = df.with_columns(
                    pl.col("price").cast(pl.Utf8)
                    .str.replace_all(r"[^\d]", "")
                    .cast(pl.Int64, strict=False)
                ).filter(pl.col("price").is_not_null())
            
            print(f"🚀 Polars loaded {len(df)} records.")
            return df
        except Exception as e:
            print(f"❌ Polars Load Error: {e}")
            return pl.DataFrame()

    def fetch_market_data(self, regex_patterns: list) -> list:
        if self.df.is_empty() or not regex_patterns: return []
        
        try:
            # 正規表現を結合して一括検索
            combined_pattern = "|".join([str(p) for p in regex_patterns])
            
            # Polarsの高速フィルタリング
            filtered = self.df.filter(
                pl.col("product_name").str.contains(combined_pattern)
            )
            
            records = filtered.to_dicts()
            for r in records: r['source'] = 'CSV'
            return records
        except Exception as e:
            print(f"Search Error: {e}")
            return []

# --- 3. Rakuten Market Manager ---
class RakutenMarketManager:
    def __init__(self):
        self.GAS_URLS = [
            "https://script.google.com/macros/s/AKfycbz9QRefEYzM6P_WVNa5M1J_99Ak3RYNqbWfve61cLDwAUXHhwhgjfcpvR94BK18LbYD/exec",
            "https://script.google.com/macros/s/AKfycbw2qu9bdAQ70k3QozUzHUP6w3CQMZhR4BykMvmwpfloorz5UqlpeqVaOESgJ9SAnACi/exec",
            "https://script.google.com/macros/s/AKfycbwFSy4pEVeGdue98Ps6q3V4_L2I0gJP9A5wanoW7eKKWbTZKPdImRLJJHvJNQ0bl28V/exec"
        ]

    def fetch_data(self, keyword: str) -> list:
        if not keyword: return []
        try:
            url = random.choice(self.GAS_URLS)
            res = requests.get(url, params={"q": keyword}, timeout=10)
            items = res.json().get("prices", [])
            results = []
            for it in items:
                try:
                    price = int(it.get('price', 0))
                    if price > 0:
                        results.append({
                            "product_name": it.get('name'),
                            "price": price,
                            "item_url": it.get('url'),
                            "source": "Rakuten"
                        })
                except: continue
            return results
        except: return []


class janpara_price:
    def __init__(self):
        print('initialized')
    def fetch_price(self, product_name: str):
        self.product_name=product_name
        with sync_playwright() as p:
            browser =p.chromium.launch(headless=True)
            page=browser.new_page()
            page.goto(f'https://buy.janpara.co.jp/buy/search?keyword={self.product_name}', wait_until='networkidle')
            page.screenshot(path='janpara.png')
            if page.locator('text=該当商品は見つかりませんでした').is_visible():
                print('該当商品は見つかりませんでした')
            else:
                price=page.locator('text=円').all_inner_texts()
                price=[int(re.sub(r'\D', '', p)) for p in price if re.sub(r'\D', '', p) != '']
                print(price)
            

# --- 4. Statistical Engine ---
class StatisticalEngine:
    def calculate_stats_range(self, prices: list[int]) -> dict:
        if not prices: return None
        data = np.array(prices)
        n = len(data)
        
        # 外れ値除去 (IQR法)
        if n >= 4:
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            clean_data = data[(data >= q1 - 1.5*iqr) & (data <= q3 + 1.5*iqr)]
        else: clean_data = data

        n_clean = len(clean_data)
        if n_clean < 2:
            val = int(clean_data[0]) if n_clean == 1 else 0
            return {"min_a": val, "max_b": val, "mean": val, "n": n_clean}

        mean = np.mean(clean_data)
        sem = np.std(clean_data, ddof=1) / np.sqrt(n_clean)
        margin = stats.t.ppf(0.995, n_clean-1) * sem if n_clean < 100 else 2.58 * sem
        
        return {"min_a": int(mean - margin), "max_b": int(mean + margin), "mean": int(mean), "n": n_clean}

# --- 5. AI Filter & Estimator ---
class AI_Filter_Estimator:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name

    def filter_by_name_only(self, target_name: str, records: list) -> list:
        if not client or not records: return []
        candidates = "\n".join([f"{i}: {r['product_name']}" for i, r in enumerate(records)])
        print(target_name)
        prompt = f"商品名: {target_name}\nリスト:\n{candidates}\n上記から明らかに商品名が異なるものやケースなどのアクセサリ、付属品のみのインデックスを除外した『valid_indices』をJSONで返して。{{'valid_indices': [4, 5, 6, 7, 8, 9]}}"
        
        try:
            response = client.models.generate_content(
                model=self.model_name, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )

            res = json.loads(response.text)
            print(res)
            return [int(i) for i in res["valid_indices"] if int(i) < len(records)]
        except Exception as e:
            # 何が起きたか出力する
            print(f"Error during AI filtering: {e}")
            # エラー時は全件返す（または空を返す）安全策
            return list(range(len(records)))

    def estimate_final_price(self, target_name: str, filtered_records: list, stats_res: dict) -> dict:
        if not client: return {"final_ai_price": 0, "reasoning": "Error"}
        
        #records_json = json.dumps([{"n": r["product_name"], "p": r["price"]} for r in filtered_records[:20]], ensure_ascii=False)
        prompt = f"商品: {target_name}\n市場データ統計: {stats_res}\n最終的な買取価格1つを『final_ai_price』と『reasoning』で決定して。何か価格は決定して。また理由は日本語で出力して。"
        
        try:
            response = client.models.generate_content(
                model=self.model_name, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return extract_json(response.text)
        except: return {"final_ai_price": 0, "reasoning": "AI error"}
