import os
import json
import re
import io
import time
import random
import requests # 追加
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from PIL import Image
from dotenv import load_dotenv
import gradio as gr
from google import genai
from google.genai import types

# --- 0. 設定 & 環境変数の読み込み ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 警告の抑制
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression, and has match groups")

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in .env file.")
    client = None
else:
    client = genai.Client(api_key=GOOGLE_API_KEY)

# --- Helper Logic ---
def extract_json(text: str):
    """JSON抽出ヘルパー (堅牢版)"""
    try:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match: text = match.group(1)
        start = text.find("{")
        if start == -1: return {}
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            return obj
        except:
            end = text.rfind("}")
            if end != -1: return json.loads(text[start:end+1])
            return {}
    except:
        return {}

# --- 1. Vision AI (Gemini) ---
class VisionAppraiser:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name

    def analyze_image(self, image: Image.Image) -> dict:
        if not client: return {"error": "API Key missing"}

        try:
            # Phase 1: Detective
            detective_prompt = """
            この商品を特定し、Pythonの `re` モジュールで検索するための「正規表現リスト」を作成してください。
            また、楽天市場で検索するための「最もヒットしやすい単語（キーワード）」も抽出してください。
            
            【作成ルール】
            1. **アニメ・キャラ名がある場合**: 「(作品名略称|正式名).*(商品種別|類義語)」や「(キャラ名).*(商品種別)」
            2. **型番がある場合**: 表記ゆれを吸収するパターン
            
            【出力フォーマット(JSON)】
            {
                "tentative_name": "正確な商品名",
                "search_keyword": "楽天市場検索用キーワード",
                "search_queries": ["正規表現パターン1", "正規表現パターン2"],
                "condition_rank": "B",
                "ai_price_c": 3000
            }
            """
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=[detective_prompt, image],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            d_result = extract_json(response.text)
            
            price = d_result.get("ai_price_c", 0)
            if not isinstance(price, (int, float)):
                 try: price = int(re.sub(r"[^\d]", "", str(price)))
                 except: price = 0
            d_result["ai_price_c"] = price

            return d_result

        except Exception as e:
            return {"error": str(e)}


# --- 2. Market Data Manager (CSV Adapter) ---
class MarketDataManager:
    def __init__(self, csv_path: str = "../merged_data_total_6542.csv"):
        self.csv_path = csv_path
        self.df = None
        self.load_csv_data()

    def load_csv_data(self):
        if os.path.exists(self.csv_path):
            try:
                self.df = pd.read_csv(self.csv_path, on_bad_lines='skip', engine='python')
                rename_map = {"商品名": "product_name", "価格": "price", "画像パス": "image_url", "URL": "item_url"}
                self.df = self.df.rename(columns=rename_map)
                
                if self.df['price'].dtype == object:
                    self.df['price'] = self.df['price'].astype(str).str.replace(',', '')
                    self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
                
                self.df = self.df.dropna(subset=['price', 'product_name'])
                self.df['price'] = self.df['price'].astype(int)
                print(f"CSV Loaded: {len(self.df)} records.")
            except Exception as e:
                print(f"CSV Load Error: {e}")
                self.df = pd.DataFrame(columns=["id", "product_name", "price"])
        else:
            self.df = pd.DataFrame(columns=["id", "product_name", "price"])

    def fetch_market_data(self, regex_patterns: list) -> list:
        """CSVから正規表現で検索し、標準フォーマットのリストを返す"""
        if self.df is None or self.df.empty:
            return []
        
        try:
            final_mask = pd.Series(False, index=self.df.index)
            for pattern in regex_patterns:
                try:
                    hit_mask = self.df['product_name'].str.contains(str(pattern), case=False, regex=True, na=False)
                    final_mask |= hit_mask
                except:
                    continue

            filtered = self.df[final_mask]
            if filtered.empty:
                return []

            # 辞書リストに変換 (sourceタグを追加)
            records = filtered[['product_name', 'price', 'item_url']].to_dict(orient='records')
            for r in records:
                r['source'] = 'CSV'
            return records
        except Exception as e:
            print(f"CSV Search Error: {e}")
            return []

# --- 3. Rakuten Market Manager (New Integration) ---
class RakutenMarketManager:
    def __init__(self):
        # 光さんのスクリプトにあるURLリスト
        self.GAS_URLS = [
            "https://script.google.com/macros/s/AKfycbz9QRefEYzM6P_WVNa5M1J_99Ak3RYNqbWfve61cLDwAUXHhwhgjfcpvR94BK18LbYD/exec",
            "https://script.google.com/macros/s/AKfycbw2qu9bdAQ70k3QozUzHUP6w3CQMZhR4BykMvmwpfloorz5UqlpeqVaOESgJ9SAnACi/exec",
            "https://script.google.com/macros/s/AKfycbwFSy4pEVeGdue98Ps6q3V4_L2I0gJP9A5wanoW7eKKWbTZKPdImRLJJHvJNQ0bl28V/exec"
        ]

    def fetch_data(self, keyword: str) -> list:
        """GAS経由で楽天データを取得し、標準フォーマットのリストを返す"""
        if not keyword:
            return []
        
        target_url = random.choice(self.GAS_URLS)
        print(f"🌐 Searching Rakuten for: {keyword} ...")
        
        try:
            response = requests.get(target_url, params={"q": keyword}, timeout=15) # リアルタイム性を考慮してタイムアウト短め
            data = response.json()
            items_list = data.get("prices", [])
            
            formatted_records = []
            for item in items_list:
                # 価格の型変換
                try:
                    p = int(item.get('price', 0))
                except:
                    p = 0
                
                if p > 0:
                    formatted_records.append({
                        "product_name": item.get('name'),
                        "price": p,
                        "item_url": item.get('url'),
                        "image_url": item.get('image_url'),
                        "source": "Rakuten" # データソースを明記
                    })
            
            print(f"✅ Rakuten Hit: {len(formatted_records)} items")
            return formatted_records

        except Exception as e:
            print(f"❌ Rakuten API Error: {e}")
            return []

# --- 4. Statistical Engine ---
class StatisticalEngine:
    def calculate_stats_range(self, prices: list[int]) -> dict:
        if not prices: return None
        data = np.array(prices)
        n = len(data)
        
        # IQR除去
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
        margin = 2.58 * sem if n_clean >= 100 else stats.t.ppf(0.995, n_clean-1) * sem
        
        return {"min_a": int(mean - margin), "max_b": int(mean + margin), "mean": int(mean), "n": n_clean}

# --- 5. AI Filter & Estimator ---
class AI_Filter_Estimator:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name

    def filter_by_name_only(self, target_name: str, records: list) -> list:
        """
        Phase 1: 価格を見ずに、商品名だけで不適切なものを弾く
        """
        if not client or not records: return []

        # データ量を減らすため、インデックスと名前だけを渡す
        name_list = [r["product_name"] for r in records]
        candidates_str = "\n".join([f"{i}: {name}" for i, name in enumerate(name_list)])

        prompt = f"""
        【ターゲット商品】: {target_name}
        
        【判定対象リスト】
        {candidates_str}

        【タスク】
        ターゲット商品と「明らかに異なるもの」や「付属品（ケース、箱のみなど）」のインデックスを特定してください。
        
        【出力(JSON)】
        {{
            "valid_indices": [0, 1, 3] 
        }}
        """

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            res = extract_json(response.text)
            
            raw_indices = res.get("valid_indices", [])
            clean_indices = []
            for x in raw_indices:
                try:
                    clean_indices.append(int(x))
                except (ValueError, TypeError):
                    continue
            
            return clean_indices

        except Exception as e:
            print(f"Filter Error: {e}")
            return list(range(len(records)))

    def estimate_final_price(self, target_name: str, filtered_records: list, stats_res: dict) -> dict:
        """
        Phase 2: フィルタ済みデータと統計結果を元に、最終価格を決める
        """
        if not client: return {"final_ai_price": 0, "reason": "API Key Error"}

        # トークン節約のため、必要な情報だけに絞る
        simple_records = []
        for r in filtered_records:
            simple_records.append({
                "name": r["product_name"],
                "price": r["price"],
                "source": r.get("source", "Unknown")
            })
        records_str = json.dumps(simple_records, ensure_ascii=False, indent=2)
        
        stats_info = "統計データなし"
        if stats_res and stats_res["n"] > 0:
            stats_info = (
                f"【統計データ（信頼度99%）】\n"
                f"- 適正範囲: ¥{stats_res['min_a']:,} 〜 ¥{stats_res['max_b']:,}\n"
                f"- 平均値: ¥{stats_res['mean']:,}\n"
                f"- サンプル数: {stats_res['n']}件"
            )

        prompt = f"""
        あなたはプロの鑑定士です。以下のデータを統合して、最終的な買取/販売想定価格を決定してください。

        【商品名】: {target_name}
        
        {stats_info}

        【参照市場データ（ノイズ除去済み）】
        {records_str}

        【指示】
        1. 「市場のデータ」を最も重視してください。
        2. 「統計データの適正範囲」を次に重視してください
        3. 最終的に「ひとつ」の価格を決定してください。

        【出力(JSON)】
        {{
            "final_ai_price": 5000,
            "reasoning": "市場の金額がこのぐらいであり、統計範囲がX〜Yであるため..."
        }}
        """

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = extract_json(response.text)
            
            price = result.get("final_ai_price", 0)
            if not isinstance(price, (int, float)):
                 try: price = int(re.sub(r"[^\d]", "", str(price)))
                 except: price = 0
            result["final_ai_price"] = price
            
            return result
        except Exception as e:
            return {"final_ai_price": 0, "reasoning": f"Error: {e}"}

# --- 6. The Brain (Controller) ---
class HybridBrain:
    def __init__(self):
        self.vision = VisionAppraiser()
        self.market_csv = MarketDataManager()      # 既存のCSV
        self.market_rakuten = RakutenMarketManager() # 追加: 楽天
        self.stats = StatisticalEngine()
        self.filter_estimator = AI_Filter_Estimator()

    def process(self, image: Image.Image):
        # 1. Vision AI: 画像から商品名と検索ワードを特定
        vision_res = self.vision.analyze_image(image)
        if "error" in vision_res: return {"error": vision_res["error"]}
        
        tentative_name = vision_res.get("tentative_name", "Unknown")
        regex_queries = vision_res.get("search_queries", [])
        search_keyword = vision_res.get("search_keyword", tentative_name) # 楽天用のきれいな単語

        # 2. Data Gathering (Hybrid)
        # A. CSV検索 (正規表現)
        csv_records = self.market_csv.fetch_market_data(regex_queries)
        
        # B. 楽天検索 (キーワード)
        rakuten_records = self.market_rakuten.fetch_data(search_keyword)
        
        # データを結合
        raw_records = csv_records + rakuten_records
        
        final_price = 0
        filter_reason = ""
        valid_records = []
        stats_res = None

        if not raw_records:
            # データがない場合はVisionの初期推定を採用
            final_price = vision_res["ai_price_c"]
            filter_reason = "市場データ(CSV/Rakuten)なし。Vision推定を採用。"
        else:
            # 3. AI Filtering (Name Only!) - 価格バイアス排除
            valid_indices = self.filter_estimator.filter_by_name_only(tentative_name, raw_records)
            
            # Python側でフィルタリング実行
            valid_records = [raw_records[i] for i in valid_indices if i < len(raw_records)]
            
            if not valid_records:
                final_price = vision_res["ai_price_c"]
                filter_reason = "フィルタリングにより全データ除外。Vision推定を採用。"
            else:
                # 4. Statistics (Recalculate on Clean Data) - 精度向上
                valid_prices = [r["price"] for r in valid_records]
                stats_res = self.stats.calculate_stats_range(valid_prices)
                
                # 5. Final Estimation (Price & Stats Aware)
                est_res = self.filter_estimator.estimate_final_price(
                    tentative_name, valid_records, stats_res
                )
                final_price = est_res.get("final_ai_price", 0)
                filter_reason = est_res.get("reasoning", "")

        # 6. Final Decision Logic
        final_min, final_max = int(final_price * 0.8), int(final_price * 1.2)
        score = '☆'
        logic = "AI Only"

        if stats_res and stats_res["n"] > 0:
            a, b = stats_res["min_a"], stats_res["max_b"]
            logic = "Hybrid (Clean Data)"
            
            # 統計範囲内なら高信頼度
            if a <= final_price <= b:
                score = '☆☆☆'
                final_min, final_max = a, b
            else:
                score = '☆☆'
                final_min = min(a, final_price)
                final_max = max(b, final_price)
        
        # データの出典内訳を集計
        source_count = {"CSV": 0, "Rakuten": 0}
        for r in valid_records:
            src = r.get("source", "Unknown")
            source_count[src] = source_count.get(src, 0) + 1

        return {
            "product_info": vision_res,
            "market_stats": stats_res,
            "market_records": raw_records,
            "valid_records_count": len(valid_records),
            "source_breakdown": source_count,
            "ai_filter_res": {"final_ai_price": final_price, "filter_reasoning": filter_reason},
            "final_decision": {
                "range_min": final_min, "range_max": final_max, 
                "confidence_score": score, "logic": logic
            }
        }

# --- UI ---
brain = HybridBrain()

def appraisal_interface(image):
    if image is None: return "画像をアップロードしてください"
    
    # 処理開始時間を計測
    start_time = time.time()
    
    res = brain.process(Image.fromarray(image))
    if "error" in res: return f"エラーが発生しました: {res['error']}"

    elapsed = time.time() - start_time
    final = res["final_decision"]
    filter_data = res["ai_filter_res"]
    stats_data = res.get("market_stats")
    src_cnt = res.get("source_breakdown", {})
    
    md = f"""
    # 🛍️ 査定結果: ¥{final['range_min']:,} 〜 ¥{final['range_max']:,}
    - 信頼度: {final['confidence_score']} ({final['logic']})
    - AI決定価格: ¥{filter_data.get('final_ai_price', 0):,}
    - 処理時間: {elapsed:.2f}秒
    - 理由: {filter_data.get('filter_reasoning', '')}
    
    ---
    ### 📊 市場データ処理詳細
    1. **検索ヒット**: {len(res['market_records'])}件
       - 📁 CSV: {sum(1 for r in res['market_records'] if r.get('source') == 'CSV')}件
       - 🌐 楽天: {sum(1 for r in res['market_records'] if r.get('source') == 'Rakuten')}件
    2. **AIフィルタリング**: 商品名のみで判定 → **{res['valid_records_count']}件** に厳選
       (内訳: CSV {src_cnt.get('CSV',0)}件, 楽天 {src_cnt.get('Rakuten',0)}件)
    """
    
    if stats_data and stats_data["n"] > 0:
        md += f"""
    3. **統計再計算**:
        - 適正範囲: ¥{stats_data['min_a']:,} 〜 ¥{stats_data['max_b']:,}
        - 平均価格: ¥{stats_data['mean']:,}
        """
    else:
        md += "\n*統計計算に必要な有効データ不足*\n"

    return md

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# 🐉 Dragon Eye (Hybrid Edition: CSV + Rakuten)")
        gr.Markdown("画像をアップロードすると、Gemini Visionが商品を特定し、ローカルCSVと楽天市場から価格を調査して査定します。")
        inp = gr.Image(type="numpy")
        out = gr.Markdown()
        gr.Button("査定開始").click(appraisal_interface, inp, out)
    
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))