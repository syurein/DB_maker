import os
import json
import re
import io
import pandas as pd
import numpy as np
from scipy import stats
from PIL import Image
from dotenv import load_dotenv
import gradio as gr
from google import genai
from google.genai import types

# --- 0. 設定 & 環境変数の読み込み ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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
            # Phase 1: Detective (ここを修正: 正規表現リストを作成させる)
            detective_prompt = """
            この商品を特定し、Pythonの `re` モジュールで検索するための「正規表現リスト」を作成してください。
            検索漏れを防ぐため、表記ゆれや類義語を `|` (OR) で含めてください。
            
            【作成ルール】
            検索クエリのリスト (`search_queries`) を作成してください。
            
            1. **アニメ・キャラ名がある場合**
               - 「(作品名略称|正式名).*(商品種別|類義語)」のパターン
               - 「(キャラ名).*(商品種別|類義語)」のパターン
               - 例: リゼロのレムのキーホルダーの場合
                 `["(Re:?ゼロ|リゼロ).*(キーホルダー|ストラップ|アクキー)", "(レム|ラム).*(キーホルダー|ストラップ|アクキー)"]`
            
            2. **型番がある場合**
               - 型番の表記ゆれを吸収するパターン
               - 例: WF-1000XM4の場合
                 `["WF.?1000XM4", "ソニー.*イヤホン.*ノイズキャンセリング"]`

            3. **その他**
               - メーカー名と広い商品カテゴリ
               - 例: `["(Sony|ソニー).*(イヤホン|ヘッドホン)"]`

            【出力フォーマット(JSON)】
            {
                "visual_cues": "特徴",
                "tentative_name": "商品名",
                "search_queries": ["正規表現パターン1", "正規表現パターン2"],
                "condition_rank": "B",
                "condition_note": "状態メモ"
            }
            """
            
            response1 = client.models.generate_content(
                model=self.model_name,
                contents=[detective_prompt, image],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            d_result = extract_json(response1.text)
            if isinstance(d_result, list): d_result = d_result[0] if d_result else {}

            # Phase 2: Appraiser
            tentative = d_result.get("tentative_name", "Unknown")
            rank = d_result.get("condition_rank", "B")
            
            appraiser_prompt = f"""
            商品: {tentative} (状態: {rank})
            中古市場価格(C)を推定。0円禁止。
            出力JSON: {{ "ai_price_c": 3000, "trend_note": "理由" }}
            """
            
            response2 = client.models.generate_content(
                model=self.model_name,
                contents=appraiser_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            a_result = extract_json(response2.text)
            if isinstance(a_result, list): a_result = a_result[0] if a_result else {}

            price = a_result.get("ai_price_c", 0)
            if not isinstance(price, (int, float)):
                try: price = int(re.sub(r"[^\d]", "", str(price)))
                except: price = 0

            return {
                "official_name": tentative,
                "search_queries": d_result.get("search_queries", [tentative]), 
                "condition_rank": rank,
                "condition_note": d_result.get("condition_note", ""),
                "ai_price_c": int(price),
                "trend_note": a_result.get("trend_note", "")
            }

        except Exception as e:
            return {"error": str(e)}


# --- 2. Market Data Manager (DB Adapter) ---
class MarketDataManager:
    def __init__(self, csv_path: str = "../MercariScraper/merged_data_total_6542.csv", mode: str = "csv"):
        self.csv_path = csv_path
        self.mode = mode
        self.df = None
        if self.mode == "csv": self.load_csv_data()

    def load_csv_data(self):
        if os.path.exists(self.csv_path):
            try:
                self.df = pd.read_csv(self.csv_path)
                rename_map = {"商品名": "product_name", "価格": "price", "画像パス": "image_url", "URL": "item_url"}
                self.df = self.df.rename(columns=rename_map)
                
                if self.df['price'].dtype == object:
                    self.df['price'] = self.df['price'].astype(str).str.replace(',', '')
                    self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
                
                self.df = self.df.dropna(subset=['price'])
                self.df['price'] = self.df['price'].astype(int)
                if 'id' not in self.df.columns: self.df['id'] = range(1, len(self.df) + 1)
                print(f"Loaded {len(self.df)} records.")
            except:
                self.df = pd.DataFrame(columns=["id", "product_name", "price"])
        else:
            self.df = pd.DataFrame(columns=["id", "product_name", "price"])

    def fetch_market_data(self, regex_patterns: list) -> dict:
        """
        検索ロジック (正規表現リスト版):
        リスト内の正規表現の「どれか」にヒットすれば採用する (OR条件)
        """
        if self.mode == "api": return {"prices": [], "records": [], "source": "api"}

        if self.df is None or self.df.empty:
            return {"prices": [], "records": [], "source": "csv_empty"}
        
        try:
            if 'product_name' not in self.df.columns:
                 return {"prices": [], "records": [], "source": "error"}

            # 全体がFalseのマスクを作成
            final_mask = pd.Series([False] * len(self.df))
            
            # 各正規表現パターンごとに検索し、結果をOR結合 (|=) していく
            for pattern in regex_patterns:
                try:
                    # regex=True で正規表現検索を実行
                    hit_mask = self.df['product_name'].str.contains(str(pattern), case=False, regex=True, na=False)
                    final_mask |= hit_mask
                except re.error:
                    print(f"Invalid regex from AI: {pattern}")
                    continue

            filtered = self.df[final_mask]

            if filtered.empty:
                return {"prices": [], "records": [], "source": "csv_no_hit"}

            prices = filtered['price'].tolist()
            records = filtered.to_dict(orient='records')
            
            return {"prices": prices, "records": records, "source": "csv"}
        except Exception as e:
            print(f"Search Error: {e}")
            return {"prices": [], "records": [], "source": "error"}


# --- 3. Statistical Engine (変更なし) ---
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


# --- 4. The Brain (Controller) ---
class HybridBrain:
    def __init__(self):
        self.vision = VisionAppraiser()
        self.market = MarketDataManager()
        self.stats = StatisticalEngine()

    def process(self, image: Image.Image):
        # 1. Vision AI
        vision_res = self.vision.analyze_image(image)
        if "error" in vision_res: return {"error": vision_res["error"]}

        # 正規表現リストを取得 (例: ["(Re:?ゼロ|リゼロ).*(キーホルダー|ストラップ)", ...])
        queries = vision_res.get("search_queries", [])
        ai_price_c = vision_res["ai_price_c"]

        # 2. 市場データ取得
        market_res = self.market.fetch_market_data(queries)
        market_prices = market_res["prices"]
        
        # 3. 統計計算
        stats_res = self.stats.calculate_stats_range(market_prices)

        # 4. 判定ロジック
        result = {
            "product_info": vision_res,
            "market_stats": stats_res,
            "market_records": market_res["records"][:5],
            "final_decision": {}
        }

        if stats_res and stats_res["n"] > 0:
            a, b = stats_res["min_a"], stats_res["max_b"]
            final_min, final_max = a, b
            
            if ai_price_c < a: final_min = ai_price_c
            elif b < ai_price_c: final_max = ai_price_c
            if ai_price_c == 0: final_min, final_max = a, b

            mu = (a + b) / 2
            diff = abs(mu - ai_price_c)
            score = 50 if ai_price_c == 0 else min(100, int(10 * (mu / (diff if diff!=0 else 1))))
            
            result["final_decision"] = {
                "range_min": final_min, "range_max": final_max,
                "confidence_score": score, "logic": "Hybrid"
            }
        else:
            conf = 0 if ai_price_c == 0 else 20
            result["final_decision"] = {
                "range_min": int(ai_price_c*0.8), "range_max": int(ai_price_c*1.2),
                "confidence_score": conf, "logic": "AI Only"
            }
        return result


# --- UI: Gradio ---
brain = HybridBrain()

def appraisal_interface(image):
    if image is None: return "画像をアップロードしてください"
    res = brain.process(Image.fromarray(image))
    if "error" in res: return f"エラー: {res['error']}"

    info = res["product_info"]
    final = res["final_decision"]
    stats_data = res.get("market_stats")
    
    # 正規表現リストを表示
    queries_str = "\n".join([f"- `{q}`" for q in info.get('search_queries', [])])

    output_md = f"""
    # 🛍️ 査定結果
    ## 🎯 ¥{final['range_min']:,} 〜 ¥{final['range_max']:,}
    - 信頼度: {final['confidence_score']}/100 ({final['logic']})
    
    ---
    ## 🤖 AI分析
    - 商品名: {info['official_name']}
    - AI予測: ¥{info['ai_price_c']:,} ({info['trend_note']})
    
    ### 🔑 使用した検索パターン (正規表現)
    {queries_str}
    
    ## 📊 市場データ
    """
    if stats_data:
        output_md += f"- ヒット: {stats_data['n']}件 (平均 ¥{stats_data['mean']:,})\n"
        output_md += "### 🔍 ヒットした商品例\n"
        for r in res.get('market_records', []):
            output_md += f"- {r['product_name']}: ¥{r['price']:,}\n"
    else:
        output_md += "\n*データなし (条件に合う商品が見つかりませんでした)*\n"
    return output_md

if __name__ == "__main__":
    with gr.Blocks(title="Dragon Eye") as demo:
        gr.Markdown("# 🐉 Dragon Eye")
        with gr.Row():
            input_img = gr.Image(type="numpy", label="画像")
            btn = gr.Button("査定", variant="primary")
        output_area = gr.Markdown()
        btn.click(fn=appraisal_interface, inputs=input_img, outputs=output_area)
    demo.launch()