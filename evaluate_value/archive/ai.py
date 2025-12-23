import os
import json
import re
from google import genai
from google.genai import types
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv('../.env')

# 新しいクライアントの初期化
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Helper: JSON抽出 ---
def extract_json(text: str):
    """JSONを抽出・パースする。戻り値は dict または list"""
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    json_str = match.group(1) if match else text
    json_str = json_str.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"raw_content": text}

class VisionAppraiser:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name

    def analyze_image(self, image_data: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # ---------------------------------------------------------
            # Phase 1: Detective (画像分析)
            # ---------------------------------------------------------
            print(f"Phase 1: Detective Work ({self.model_name})...")
            
            detective_prompt = """
            この商品を特定するための情報を抽出してください。
            出力は以下のキーを持つ単一のJSONオブジェクトにしてください（リストにしないでください）。
            
            1. visual_cues: 検索キーワードとなる特徴（型番、ロゴ、製品名）のみを抽出。
            2. tentative_name: 推測される商品名。
            3. condition_rank: S/A/B/C/J判定。
            """

            response1 = client.models.generate_content(
                model=self.model_name,
                contents=[detective_prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.4,
                )
            )
            
            d_result = extract_json(response1.text)
            
            # ★★★【修正ポイント】リストで返ってきた場合のガード処理 ★★★
            if isinstance(d_result, list):
                if len(d_result) > 0:
                    d_result = d_result[0] # リストの先頭要素を取り出す
                else:
                    d_result = {} # 空リストなら空辞書にする
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

            if "raw_content" in d_result:
                cues = str(d_result["raw_content"])[:500]
                tentative = "Unknown"
                rank = "Unknown"
            else:
                cues = d_result.get("visual_cues", "")
                tentative = d_result.get("tentative_name", "Unknown")
                rank = d_result.get("condition_rank", "B")

            print(f"  -> Guess: {tentative}")

            # ---------------------------------------------------------
            # Phase 2: Appraiser (検索 & 査定)
            # ---------------------------------------------------------
            print("Phase 2: Market Research (Google Search)...")
            
            appraiser_prompt = f"""
            以下の商品の正式名称を特定し、中古相場を査定してください。

            【鑑識情報】
            - 特徴: {cues}
            - 仮名: {tentative}
            - 状態: {rank}

            【指示】
            1. Google検索で正式名称を特定。
            2. その状態での中古買取相場(C)を調査。
            
            【出力フォーマット】
            ```json
            {{
                "final_official_name": "商品名",
                "ai_price_c": 10000,
                "trend_note": "根拠"
            }}
            ```
            """

            response2 = client.models.generate_content(
                model=self.model_name,
                contents=appraiser_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )

            a_result = extract_json(response2.text)
            
            # Phase 2 も念のためリスト対策を入れておきます
            if isinstance(a_result, list):
                a_result = a_result[0] if len(a_result) > 0 else {}

            price = 0
            if "ai_price_c" in a_result:
                try: price = int(a_result["ai_price_c"])
                except: pass

            final_result = {
                "official_name": a_result.get("final_official_name", tentative),
                "condition_rank": rank,
                "ai_price_c": price,
                "trend_note": a_result.get("trend_note", ""),
                "visual_cues": cues
            }
            return final_result

        except Exception as e:
            print(f"Pipeline Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "official_name": "System Error",
                "ai_price_c": 0,
                "trend_note": str(e),
                "condition_rank": "Unknown"
            }

if __name__ == "__main__":
    image_path = r"C:\Users\hikar\Downloads\DBscrayper\MercariScraper\downloaded_images\0_0_0_1765035531.jpg"
    
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
             img_bytes = f.read()
        
        appraiser = VisionAppraiser()
        result = appraiser.analyze_image(img_bytes)
        print("\n--- Final Appraisal Result ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))