import os
import json
import re
import io
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

# .envの読み込み
load_dotenv()

# --- Config ---
app = Flask(__name__)
# 日本語文字化け対策
app.json.ensure_ascii = False 

# Google GenAI Client Setup
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Helper Logic (The Brain) ---
def extract_json(text: str):
    """JSON抽出ヘルパー"""
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
            
            # Phase 1: Detective
            print(f"Phase 1: Detective Work ({self.model_name})...")
            detective_prompt = """
            この商品を特定するための情報を抽出してください。
            出力は以下のキーを持つ単一のJSONオブジェクトにしてください（リスト禁止）。
            
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
            if isinstance(d_result, list):
                d_result = d_result[0] if d_result else {}

            if "raw_content" in d_result:
                cues = str(d_result["raw_content"])[:500]
                tentative = "Unknown"
                rank = "Unknown"
            else:
                cues = d_result.get("visual_cues", "")
                tentative = d_result.get("tentative_name", "Unknown")
                rank = d_result.get("condition_rank", "B")

            # Phase 2: Appraiser
            print(f"Phase 2: Market Research (Google Search) for {tentative}...")
            appraiser_prompt = f"""
            以下の商品の正式名称を特定し、中古相場を査定してください。
            【鑑識情報】特徴: {cues}, 仮名: {tentative}, 状態: {rank}
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
            if isinstance(a_result, list):
                a_result = a_result[0] if a_result else {}

            price = 0
            if "ai_price_c" in a_result:
                try: price = int(a_result["ai_price_c"])
                except: pass

            return {
                "official_name": a_result.get("final_official_name", tentative),
                "condition_rank": rank,
                "ai_price_c": price,
                "trend_note": a_result.get("trend_note", ""),
                "visual_cues": cues
            }

        except Exception as e:
            print(f"Error: {e}")
            return {"error": str(e)}

# インスタンス生成（サーバー起動時に1回だけ実行）
appraiser = VisionAppraiser()

# --- API Routes ---

@app.route('/')
def home():
    return "AI Appraisal API is Running!"

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    """
    POST /analyze
    Form-Data:
      image: (file binary)
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 画像バイナリを読み込んでAIに渡す
        img_bytes = file.read()
        result = appraiser.analyze_image(img_bytes)
        
        # 結果をJSONで返す
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 開発用サーバー起動 (ポート5000)
    app.run(debug=True, host='0.0.0.0', port=5000)