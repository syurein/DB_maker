import pandas as pd
import requests
import google.generativeai as genai
import time
import json
from dotenv import load_dotenv
import os
# ==========================================
# 設定エリア
# ==========================================
# 1. Google Gemini APIキー (AI用)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# 2. GAS ManagerのURL (さっきデプロイしたやつ)
GAS_MANAGER_URL = "https://script.google.com/macros/s/AKfycbz9QRefEYzM6P_WVNa5M1J_99Ak3RYNqbWfve61cLDwAUXHhwhgjfcpvR94BK18LbYD/exec"

# 3. 入力と出力のファイル名
INPUT_CSV = "./MercariScraper/merged_data_total_6542.csv"  # 読み込むCSV (ヘッダーに 'product_name' がある前提)
OUTPUT_CSV = "results.csv"  # 結果を保存するCSV

# ==========================================
# AIセットアップ
# ==========================================
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_search_keyword(product_name):
    """
    商品名から、楽天検索に最適なキーワードをAIに作らせる
    例: "中古のソニーのヘッドホン XM4" -> "Sony WH-1000XM4"
    """
    prompt = f"""
    あなたはECサイトの検索エンジニアです。
    以下の商品名から、楽天市場で価格調査をするための「最も精度の高い検索キーワード」を1つだけ抽出してください。
    余計な説明は不要です。キーワードのみを返してください。
    
    商品名: {product_name}
    キーワード:
    """
    try:
        response = model.generate_content(prompt)
        keyword = response.text.strip()
        print(f"🤖 AI Keyword: {product_name} -> {keyword}")
        return keyword
    except Exception as e:
        print(f"❌ AI Error: {e}")
        return product_name # エラーなら元の名前をそのまま使う

def fetch_prices_from_gas(keyword):
    """
    GAS Managerに問い合わせて価格リストを取得する
    """
    try:
        # GETリクエストでGASを叩く
        response = requests.get(GAS_MANAGER_URL, params={"q": keyword}, timeout=30)
        data = response.json()
        
        # GASからのレスポンス形式: { "prices": [1000, 1200, ...], ... }
        prices = data.get("prices", [])
        return prices
    except Exception as e:
        print(f"❌ GAS Error: {e}")
        return []

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

    results = []

    # 2. 1行ずつ処理
    for index, row in df.iterrows():
        original_name = row['商品名'] # CSVの列名に合わせて変更してください
        
        print(f"\n--- Processing {index + 1}/{len(df)}: {original_name} ---")

        # Step A: AIでキーワード化
        search_keyword = generate_search_keyword(original_name)
        
        # Step B: GASワーカーに問い合わせ
        price_list = fetch_prices_from_gas(search_keyword)
        
        # 統計データの計算 (最小、最大、平均、件数)
        count = len(price_list)
        if count > 0:
            min_price = min(price_list)
            max_price = max(price_list)
            avg_price = sum(price_list) / count
        else:
            min_price = max_price = avg_price = 0

        print(f"💰 Prices found: {count}件 (Min: {min_price}円)")

        # 結果をリストに追加
        results.append({
            "original_name": original_name,
            "search_keyword": search_keyword,
            "count": count,
            "min_price": min_price,
            "max_price": max_price,
            "avg_price": int(avg_price),
            "raw_prices": str(price_list) # 生データも文字として保存
        })

        # API制限への配慮 (少し待機)
        time.sleep(1) 

    # 3. 結果をCSVに保存
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ Done! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()