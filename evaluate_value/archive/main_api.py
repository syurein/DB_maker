import os
import io
from flask import Flask, request, jsonify
from PIL import Image
import traceback

# main.pyからHybridBrainクラスをインポート
# 同じディレクトリにあることを想定
try:
    from main import HybridBrain
except ImportError:
    print("エラー: main.py から HybridBrain をインポートできませんでした。")
    print("カレントディレクトリが 'evaluate_value' であることを確認してください。")
    # 代替としてダミークラスを定義し、起動だけはできるようにする
    class HybridBrain:
        def process(self, image):
            return {"error": "HybridBrain class not found. Check project structure."}

# Flaskアプリケーションの初期化
app = Flask(__name__)

# HybridBrainのインスタンスをグローバルに作成
# 起動時に一度だけモデルをロードするため
try:
    print("Initializing HybridBrain... (This may take a moment)")
    brain = HybridBrain()
    print("HybridBrain initialized successfully.")
except Exception as e:
    print(f"Failed to initialize HybridBrain: {e}")
    brain = None

@app.route('/evaluate', methods=['POST'])
def evaluate_image():
    """
    画像を受け取り、査定結果をJSONで返すAPIエンドポイント
    """
    if brain is None:
        return jsonify({"error": "Brain not initialized. Check server logs."}), 500

    # 1. リクエストから画像ファイルを取得
    if 'image' not in request.files:
        return jsonify({"error": "Request does not contain an image file"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 2. 画像をPIL.Imageオブジェクトに変換
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # 3. HybridBrainで処理を実行
        print(f"Processing image: {file.filename}")
        result = brain.process(pil_image)
        print("Processing complete.")

        # 4. 結果をJSONで返す
        if "error" in result:
            return jsonify(result), 500
        else:
            # numpyのデータ型が含まれているとjsonifyでエラーになる場合があるため、
            # Pythonの標準データ型に変換する
            # (このスクリプトでは既に対策済みだが、念のため)
            return jsonify(result)

    except Exception as e:
        print("An error occurred during evaluation:")
        traceback.print_exc()
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # デバッグモードでサーバーを起動
    # host='0.0.0.0' でローカルネットワーク内からアクセス可能に
    app.run(host='0.0.0.0', port=5001, debug=True)
