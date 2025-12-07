import os
import json
import re
import typing_extensions as typing
import google.generativeai as genai
from google.generativeai import protos
from PIL import Image
import io
import argparse
import traceback
from dotenv import find_dotenv, load_dotenv

# --- 設定 ---
# 実行時に利用可能なモデルのリストが出力されます。
# 下記のVISION_MODELを、リストに表示されたVision対応モデル（'gemini-pro-vision'など）に書き換えてみてください。
VISION_MODEL = "models/gemini-2.5-flash-image"
TEXT_MODEL = "models/gemini-pro-latest" # ツール利用に安定しているテキストモデル

# --- 初期化 ---
load_dotenv(find_dotenv())
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("環境変数 `GOOGLE_API_KEY` が見つかりません。.envファイルを確認してください。")
genai.configure(api_key=api_key)


# --- Helper Function: 安全なJSONパース ---
def safe_parse_json(text: str, default_key: str = "content") -> dict:
    """
    AIの出力からJSONを抽出・パースする。失敗した場合は生テキストを返す。
    """
    cleaned_text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        print(f"Warning: JSON Parse Failed. Using raw text fallback.\nRaw: {text[:80]}...")
        return {default_key: cleaned_text}

# --- Schemas ---
class DetectiveSchema(typing.TypedDict):
    visual_cues: str
    tentative_name: str
    condition_rank: str

class VisionAppraiser:
    def __init__(self, vision_model: str, text_model: str):
        self.vision_model_name = vision_model
        self.text_model_name = text_model
        
        self.shared_config = genai.GenerationConfig(temperature=0.4)

        # 1. Detective (Vision) - 画像認識担当
        self.detective_model = genai.GenerativeModel(
            model_name=self.vision_model_name,
            generation_config=self.shared_config,
            system_instruction=(
                "あなたは鑑識官です。画像から商品を特定するための特徴（ロゴ、型番、形状、キズ）を詳細に言語化してください。"
                "回答は必ず下記のキーを持つJSON形式で出力してください: visual_cues, tentative_name, condition_rank"
            )
        )

        # 2. Appraiser (Search) - テキストベースの検索と査定担当
        self.appraiser_model = genai.GenerativeModel(
            model_name=self.text_model_name,
            generation_config=self.shared_config,
            tools=['google_search'], # モデルによっては protos.Tool(...) が必要になる場合がある。エラーが出たら切り替えてみてください。
            system_instruction=(
                "あなたは市場分析のプロです。検索機能を使用して正確な査定を行ってください。"
                "回答は必ず正しいJSON形式の文字列のみを出力してください。"
            )
        )

    def analyze_image(self, image_data: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # --- Phase 1: Detective ---
            print(f"Phase 1: Analyzing visual cues (Model: {self.vision_model_name})...")
            detective_prompt = """
            この商品を特定するための情報を抽出し、JSON形式で出力してください。
            1. **visual_cues**: 検索の手がかりになる特徴（ロゴ、文字情報、形状）を具体的に。
            2. **tentative_name**: 推測される商品名。
            3. **condition_rank**: 商品の状態をS(新品同様), A(美品), B(良品), C(使用感あり), J(ジャンク)の5段階で判定。
            """
            
            d_resp = self.detective_model.generate_content([detective_prompt, image])
            d_result = safe_parse_json(d_resp.text, default_key="visual_cues")

            cues = d_result.get("visual_cues", str(d_result))
            tentative = d_result.get("tentative_name", "Unknown")
            rank = d_result.get("condition_rank", "B")

            print(f"  -> Cues: {cues[:100]}...")
            print(f"  -> Guess: {tentative}")
            print(f"  -> Rank: {rank}")

            # --- Phase 2: Appraiser ---
            print(f"Phase 2: Verifying and Pricing via Google Search (Model: {self.text_model_name})...")
            appraiser_prompt = f"""
            以下の商品の正式名称を特定し、中古相場を査定してください。

            【鑑識官からの報告】
            - 視覚的特徴: {cues}
            - 仮の名称: {tentative}
            - 状態ランク: {rank}

            【指示】
            1. Google検索を必ず使用し、特徴に合致する正式な商品を特定してください。
            2. 特定した商品名と状態ランク[{rank}]を元に、現在の中古市場価格(C)を調査してください。
            
            【出力フォーマット】
            以下のキーを持つJSONのみを出力してください。
            {{
                "final_official_name": "商品名",
                "ai_price_c": 12345,
                "trend_note": "価格判断の根拠や市場動向"
            }}
            """
            
            a_resp = self.appraiser_model.generate_content(appraiser_prompt)
            a_result = safe_parse_json(a_resp.text, default_key="trend_note")

            try:
                price = int(a_result.get("ai_price_c", 0))
            except (ValueError, TypeError):
                price = 0

            final_result = {
                "official_name": a_result.get("final_official_name", tentative),
                "condition_rank": rank,
                "ai_price_c": price,
                "trend_note": a_result.get("trend_note", "解析情報なし"),
                "visual_cues": cues[:100] + "..." if len(cues) > 100 else cues
            }
            return final_result

        except Exception as e:
            print(f"Pipeline Critical Error: {e}")
            traceback.print_exc()
            return {
                "official_name": "System Error",
                "ai_price_c": 0,
                "trend_note": str(e),
                "condition_rank": "Unknown"
            }

if __name__ == "__main__":
    # 利用可能なモデルをリストアップ
    print("--- 利用可能なモデル ---")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"モデルのリストを取得できませんでした: {e}")
    print("------------------------\n")

    parser = argparse.ArgumentParser(description="Vision-based Value Appraiser")
    parser.add_argument("image_path", type=str, help="Path to the image file to be evaluated.")
    args = parser.parse_args()

    if os.path.exists(args.image_path):
        with open(args.image_path, "rb") as f:
             img_bytes = f.read()
        
        # 設定したモデル名で実行
        appraiser = VisionAppraiser(vision_model=VISION_MODEL, text_model=TEXT_MODEL)
        result = appraiser.analyze_image(img_bytes)
        
        print("\n--- 最終査定結果 ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Error: 画像が見つかりません: {args.image_path}")
