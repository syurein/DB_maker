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

# 1. GAS URLリスト (複数可)
GAS_MANAGER_URLS = [
    "https://script.google.com/macros/s/AKfycbz9QRefEYzM6P_WVNa5M1J_99Ak3RYNqbWfve61cLDwAUXHhwhgjfcpvR94BK18LbYD/exec",
    "https://script.google.com/macros/s/AKfycbw2qu9bdAQ70k3QozUzHUP6w3CQMZhR4BykMvmwpfloorz5UqlpeqVaOESgJ9SAnACi/exec",
    "https://script.google.com/macros/s/AKfycbwFSy4pEVeGdue98Ps6q3V4_L2I0gJP9A5wanoW7eKKWbTZKPdImRLJJHvJNQ0bl28V/exec"
]

# 2. 入出力ファイル名
# ※カラム名は入力CSVに合わせてコード内の row[...] 部分を調整してください
INPUT_CSV = "./merged_data_total_6542.csv" 

OUTPUT_CSV = "results_flat_data.csv"            # 結果CSV (1行1商品)
HISTORY_LOG_FILE = "processed_history.log"      # 履歴保存用ファイル

# 3. 動作設定
REGEX_PATTERN = ".*"  # フィルタ用 (全件なら ".*")
MAX_WORKERS = 9     # 並列数
SAVE_INTERVAL = 10    # 何商品ごとに保存するか

# ==========================================
# 準備
# ==========================================
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def load_history():
    """履歴ログを読み込み、処理済みの商品名セットを返す"""
    if not os.path.exists(HISTORY_LOG_FILE):
        return set()
    with open(HISTORY_LOG_FILE, 'r', encoding='utf-8') as f:
        # 改行を除去してセットに格納
        return set(line.strip() for line in f if line.strip())

def append_history(product_names):
    """処理した商品名をログに追記"""
    with open(HISTORY_LOG_FILE, 'a', encoding='utf-8') as f:
        for name in product_names:
            f.write(f"{name}\n")

def generate_search_keyword(product_name):
    """AIで検索キーワード生成"""
    prompt = f"""
    以下の商品名から、楽天市場で価格調査をするための「最も精度の高い検索キーワード」を1つだけ抽出してください。
    できるだけ短いワードで検索ヒット数が多くなるような単語にしてください。
    型番がある場合は必ず含め、表記ゆれをなくしてください。余計な説明は不要です。
    
    商品名: {product_name}
    キーワード:
    """
    try:
        time.sleep(random.uniform(0.5, 1.5))
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return product_name

def fetch_data_from_gas(keyword):
    """GASからデータ取得"""
    target_url = random.choice(GAS_MANAGER_URLS)
    try:
        # API制限やネットワークエラー対策でリトライ機構を入れるとなお良し
        response = requests.get(target_url, params={"q": keyword}, timeout=45)
        data = response.json()
        return data.get("prices", [])
    except Exception as e:
        print(f"❌ GAS Error ({keyword}): {e}")
        return []

def process_single_row_task(row):
    """
    並列処理用のタスク関数
    1つの入力行に対し、複数の結果行（リスト）を返す
    """
    # 列名の揺らぎ吸収
    original_name = row.get('product_name') or row.get('商品名')
    
    # 1. キーワード生成
    search_keyword = generate_search_keyword(original_name)
    
    # 2. 楽天データ取得
    items_list = fetch_data_from_gas(search_keyword)
    
    result_rows = []
    
    # 3. ヒットした商品を1つずつ行にする
    if items_list:
        for item in items_list:
            result_rows.append({
                "product_name": item.get('name'),       # 楽天の商品名
                "price": item.get('price'),             # 価格
                "image_url": item.get('image_url'),     # 画像URL
                "item_url": item.get('url'),            # 商品URL
                "data_source": "Rakuten"
            })
        print(f"✅ Hit: {search_keyword} -> {len(items_list)}件")
    else:
        # ヒットしなかった場合、CSV用のリスト(result_rows)には何も追加しない
        # これにより、CSVには書き込まれないが、original_nameは返されるので履歴には残る
        print(f"⚠️ No Hit: {search_keyword} (ログのみ記録)")

    return original_name, result_rows

# ==========================================
# メイン処理
# ==========================================
def main():
    # 1. 入力CSV読み込み
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_CSV} が見つかりません。")
        return

    # 2. 履歴読み込み & フィルタリング
    processed_history = load_history()
    print(f"📜 History loaded: {len(processed_history)} items processed.")

    # 処理対象のカラム特定
    target_col = 'product_name' if 'product_name' in df.columns else '商品名'
    
    # フィルタ: 正規表現 AND 未処理のもの
    if REGEX_PATTERN and REGEX_PATTERN != ".*":
        df = df[df[target_col].astype(str).str.contains(REGEX_PATTERN, regex=True, case=False, na=False)]
    
    # 履歴にあるものは除外
    df_target = df[~df[target_col].isin(processed_history)]
    
    total_targets = len(df_target)
    print(f"👉 Processing targets: {total_targets} items (Skipped: {len(df) - total_targets})")
    
    if total_targets == 0:
        print("🎉 全て処理済みです！")
        return

    all_results = []       # 結果データを溜めるリスト
    just_processed = []    # 今回のバッチで処理完了した商品名リスト

    # 3. 並列処理開始
    print(f"🚀 Starting processing...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # タスク登録
        future_to_row = {executor.submit(process_single_row_task, row): row for _, row in df_target.iterrows()}
        
        completed_count = 0
        
        for future in as_completed(future_to_row):
            try:
                # 結果を受け取る (元の商品名, 結果のリスト)
                orig_name, rows = future.result()
                
                # rowsが空（No Hit）の場合はリストに追加されない
                if rows:
                    all_results.extend(rows)
                
                # 検索自体は完了したので履歴リストには追加する
                just_processed.append(orig_name)
                
                completed_count += 1
                
                # 4. 定期保存
                if completed_count % SAVE_INTERVAL == 0:
                    print(f"💾 Saving chunk... ({completed_count}/{total_targets})")
                    
                    # 書き込むデータがある場合のみ保存処理を行う
                    if all_results:
                        output_df = pd.DataFrame(all_results)
                        
                        # 初回作成時と追記時のハンドリング
                        if os.path.exists(OUTPUT_CSV):
                            output_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding='utf-8-sig')
                        else:
                            output_df.to_csv(OUTPUT_CSV, mode='w', header=True, index=False, encoding='utf-8-sig')
                        
                        # メモリ解放のためリストをクリア
                        all_results = [] 
                    else:
                        print("  (No valid hits in this chunk to save)")

                    # 履歴ログ保存（ヒット有無に関わらず保存）
                    append_history(just_processed)
                    just_processed = [] # クリア

            except Exception as e:
                print(f"❌ Error in thread: {e}")

    # 5. 残りのデータを保存
    if all_results:
        output_df = pd.DataFrame(all_results)
        if os.path.exists(OUTPUT_CSV):
            output_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            output_df.to_csv(OUTPUT_CSV, mode='w', header=True, index=False, encoding='utf-8-sig')
    
    # 残りの履歴も保存
    if just_processed:
        append_history(just_processed)

    print(f"\n🎉 Process Complete! Log saved to {HISTORY_LOG_FILE}")

if __name__ == "__main__":
    main()