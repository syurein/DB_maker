import pandas as pd
import requests
import google.generativeai as genai
import time
import json
import random
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 設定エリア
# ==========================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. 複数のGAS Manager URL (リスト形式)
# 複数ある場合はカンマ区切りで追加してください
GAS_MANAGER_URLS = [
    "https://script.google.com/macros/s/AKfycbz9QRefEYzM6P_WVNa5M1J_99Ak3RYNqbWfve61cLDwAUXHhwhgjfcpvR94BK18LbYD/exec",
    # "https://script.google.com/macros/s/xxxxx.../exec", 
]

# 2. 入出力ファイル
INPUT_CSV = "./MercariScraper/merged_data_total_6542.csv"
OUTPUT_CSV = "results_parallel.csv"

# 3. 正規表現フィルタ (処理したい商品名の条件)
# 例: ".*" (すべて), "ソニー|Sony", "iPhone.*128GB"
REGEX_PATTERN = ".*" 

# 4. 並列処理の設定
MAX_WORKERS = 3       # 同時に動かすスレッド数 (増やしすぎるとAPI制限にかかります)
SAVE_INTERVAL = 10    # 何件ごとに保存するか

# ==========================================
# AIセットアップ
# ==========================================
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_search_keyword(product_name):
    """商品名から検索キーワードを生成"""
    prompt = f"""
    あなたはECサイトの検索エンジニアです。
    以下の商品名から、楽天市場で価格調査をするための「最も精度の高い検索キーワード」を1つだけ抽出してください。
    型番がある場合は必ず含めてください。余計な説明は不要です。
    
    商品名: {product_name}
    キーワード:
    """
    try:
        # 並列処理時のAPI制限回避のため少しランダムに待機
        time.sleep(random.uniform(0.5, 1.5))
        response = model.generate_content(prompt)
        keyword = response.text.strip()
        return keyword
    except Exception as e:
        print(f"❌ AI Error ({product_name}): {e}")
        return product_name

def fetch_prices_from_gas(keyword):
    """ランダムなGAS Managerを選んで価格を取得"""
    target_url = random.choice(GAS_MANAGER_URLS) # URLをランダム選択して負荷分散
    try:
        response = requests.get(target_url, params={"q": keyword}, timeout=45)
        data = response.json()
        prices = data.get("prices", [])
        return prices
    except Exception as e:
        print(f"❌ GAS Error ({keyword}): {e}")
        return []

def process_single_item(row):
    """1行分の処理を行う関数 (並列実行用)"""
    original_name = row['商品名'] # CSVのヘッダーに合わせて変更
    
    # 1. AIでキーワード化
    search_keyword = generate_search_keyword(original_name)
    
    # 2. GASへ問い合わせ
    price_list = fetch_prices_from_gas(search_keyword)
    
    # 3. 統計計算
    count = len(price_list)
    if count > 0:
        min_price = min(price_list)
        max_price = max(price_list)
        avg_price = sum(price_list) / count
    else:
        min_price = max_price = avg_price = 0

    print(f"✅ Finished: {search_keyword} -> {count}件 (Min: {min_price}円)")

    return {
        "original_name": original_name,
        "search_keyword": search_keyword,
        "count": count,
        "min_price": min_price,
        "max_price": max_price,
        "avg_price": int(avg_price),
        "raw_prices": str(price_list)
    }

# ==========================================
# メイン処理
# ==========================================
def main():
    # 1. CSV読み込み
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"📂 CSV loaded: {len(df)} items")
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_CSV} が見つかりません。")
        return

    # 2. 正規表現フィルタリング
    if REGEX_PATTERN and REGEX_PATTERN != ".*":
        print(f"🔍 Filtering with regex: '{REGEX_PATTERN}'")
        df = df[df['商品名'].str.contains(REGEX_PATTERN, regex=True, case=False, na=False)]
    
    print(f"👉 Target items: {len(df)} items")
    
    results = []
    
    # 3. 並列処理の開始
    print(f"🚀 Starting parallel processing (Max workers: {MAX_WORKERS})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # タスクの登録
        future_to_row = {executor.submit(process_single_item, row): index for index, row in df.iterrows()}
        
        completed_count = 0
        
        for future in as_completed(future_to_row):
            try:
                data = future.result()
                results.append(data)
            except Exception as e:
                print(f"❌ Unexpected Error: {e}")
            
            completed_count += 1
            
            # 4. 定期保存 (SAVE_INTERVAL件ごと)
            if completed_count % SAVE_INTERVAL == 0:
                print(f"💾 Saving progress... ({completed_count}/{len(df)})")
                save_df = pd.DataFrame(results)
                save_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 5. 最終保存
    save_df = pd.DataFrame(results)
    save_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎉 All Done! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()