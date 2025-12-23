import os
import re
import json
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from google import genai

# utils.py から必要なクラスと関数をインポート
# ※ extract_json も utils.py に移動させている前提です
from utils import (
    VisionAppraiser, 
    MarketDataManager, 
    RakutenMarketManager, 
    StatisticalEngine, 
    AI_Filter_Estimator,
    Janpara_price
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
# --- 0. 基本設定 ---
load_dotenv()
app = Flask(__name__)
# --- 0. 基本設定とログ ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 【対策1】ファイルサイズ制限: 5MB以上のリクエストは即座に拒否
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 

# 【対策2】レートリミットの設定
# IPアドレスごとにリクエスト回数を制限
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["100 per day", "30 per hour"],
    storage_uri="memory://", # 本番環境で複数サーバーにするなら Redis 等を推奨
)



# --- 1. Hybrid Brain (Controller) ---
class HybridBrain:
    def __init__(self):
        # VisionAppraiser に client を注入して NameError を防ぐ
        self.vision = VisionAppraiser()
        self.market_csv = MarketDataManager()
        self.market_rakuten = RakutenMarketManager()
        self.stats = StatisticalEngine()
        self.filter_estimator = AI_Filter_Estimator()
        self.janpara_price = Janpara_price()
    def process(self, image: Image.Image, option: str) :
        # 1. 画像解析 (Gemini Vision)
        v_res = self.vision.analyze_image(image)
        if "error" in v_res:
            return v_res
        
        t_name = v_res.get("tentative_name", "Unknown")
        
        # 2. 市場データ収集 (CSV + 楽天)
        csv_data = self.market_csv.fetch_market_data(v_res.get("search_queries", []))
        rakuten_data = self.market_rakuten.fetch_data(v_res.get("search_keyword", t_name))
        janpara_data = self.janpara_price.fetch_price(t_name)
        raw_records = csv_data + rakuten_data + janpara_data
        print(f'csv:{csv_data}, rakuten:{rakuten_data}, janpara:{janpara_data}')
        # --- パターンA: 市場データが見つからない場合 ---
        if not raw_records:
            # AIの推定価格を数値として取得
            raw_ai_price = v_res.get("ai_price_c", 0)
            try:
                ai_p = int(re.sub(r'\D', '', str(raw_ai_price))) if raw_ai_price else 0
            except:
                ai_p = 0

            return {
                "product_info": v_res, 
                "valid_records_count": 0,
                "ai_filter_res": {
                    "final_ai_price": ai_p, 
                    "filter_reasoning": "市場データが見つかりませんでした。AIによる推定値を表示します。"
                },
                "final_decision": {
                    "range_min": int(ai_p * 0.8),
                    "range_max": int(ai_p * 1.2),
                    "confidence_score": "☆",
                    "logic": "Vision Only"
                }
            }

        # --- パターンB: 市場データがある場合 ---
        # 3. データのフィルタリングと統計計算
        valid_idx = self.filter_estimator.filter_by_name_only(t_name, raw_records)
        valid_records = [raw_records[i] for i in valid_idx]
        
        v_prices = [r["price"] for r in valid_records]
        stats_res = self.stats.calculate_stats_range(v_prices)
        
        # 4. AIによる最終価格推定
        if option == 'default':
            est_res = self.filter_estimator.estimate_final_price(t_name, valid_records, stats_res)
            
            # 型エラー対策: final_ai_price を確実に数値にする
            raw_final_p = est_res.get("final_ai_price", 0)
            try:
                # 文字列（"15,000"など）なら数字以外を消してint化
                if isinstance(raw_final_p, str):
                    final_p = int(re.sub(r'\D', '', raw_final_p))
                else:
                    final_p = int(raw_final_p)
            except (ValueError, TypeError):
                final_p = 0

            return {
                "product_info": v_res,
                "valid_records_count": len(valid_records),
                "ai_filter_res": {
                    "final_ai_price": final_p, 
                    "filter_reasoning": est_res.get("reasoning")
                },
                "final_decision": {
                    # 統計データがあればそれを使用、なければAI価格から算出
                    "range_min": stats_res["min_a"] if stats_res else int(final_p * 0.8),
                    "range_max": stats_res["max_b"] if stats_res else int(final_p * 1.2),
                    "confidence_score": "☆☆☆" if stats_res and stats_res["n"] > 2 else "☆☆",
                    "logic": "Hybrid (Polars + Gemini)"
                }
            }
        else:
            return {
                "product_info": v_res,
                "valid_records_count": len(valid_records),
                "ai_filter_res": {
                    "final_ai_price": 0, 
                    "filter_reasoning": "過去データからの統計推定によって算出しました。"
                },
                "final_decision": {
                    "range_min": stats_res["min_a"] if stats_res else 0,
                    "range_max": stats_res["max_b"] if stats_res else 0,
                    "confidence_score": "☆" if stats_res else "☆☆",
                    "logic": "Statistical Only"
                }
            }




# --- 2. Flask Routes ---
# クライアントを注入してインスタンス化
brain = HybridBrain()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/appraise', methods=['POST'])
# 【対策3】特定のエンドポイントに厳格なリミットを適用
# 1分間に5回、かつ1時間に30回までに制限（査定は重い処理なので厳しめ）
@limiter.limit("5 per minute; 30 per hour")
def appraise():
    # 【対策4】オプションのバリデーション
    option = request.args.get('option', 'default')
    if option not in ['default', 'non']:
        option = 'default'

    # 【対策5】ファイルの存在と形式チェック
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # MIMEタイプが画像であることを確認
    if not file.content_type.startswith('image/'):
        return jsonify({"error": "File type not supported. Please upload an image."}), 400

    try:
        # 画像を開く（ここでMAX_CONTENT_LENGTHにより巨大画像は既に弾かれている）
        img = Image.open(file.stream).convert("RGB")
        
        # 【対策6】内部で画像をリサイズ（計算負荷軽減）
        # 1024px以上に大きい場合は縮小するなどの処理を入れるとより安全
        img.thumbnail((1024, 1024))

        # 脳内処理開始
        res = brain.process(img, option=option) 
        return jsonify(res)

    except Exception as e:
        logger.error(f"Appraisal Error: {str(e)}")
        return jsonify({"error": "Internal server error during analysis"}), 500

# 【対策7】レートリミット超過時のエラーハンドリング
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Too many requests. Please wait a moment."}), 429

# 【対策8】ファイルサイズ超過時のエラーハンドリング
@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "File is too large (Max 5MB)"}), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))