import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import pandas as pd
import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import tempfile
import traceback
import threading
import uuid
import shutil
import time
from datetime import datetime

# ==========================================
# 設定・定数
# ==========================================
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "processed_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# モデルのロード (グローバル)
# ==========================================
MODEL_NAME = "openai/clip-vit-base-patch32"
print(f"Loading CLIP model: {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print(f"Model loaded on {device}.")
except Exception as e:
    print(f"Model load failed: {e}")

# ==========================================
# ベクトル生成ロジック
# ==========================================
def generate_image_vector(image_filename, base_dir):
    """画像パスとベースディレクトリからCLIPベクトルを生成"""
    if pd.isna(image_filename) or str(image_filename).strip() == "" or str(image_filename) == "nan":
        return None, "Empty filename"

    filename = str(image_filename).strip()
    full_path = os.path.join(base_dir, filename)
    
    # フォルダ探索ロジック
    if not os.path.exists(full_path):
        alt_path = os.path.join(base_dir, "downloaded_images", filename)
        if os.path.exists(alt_path):
            full_path = alt_path
        else:
            return None, f"File not found"

    try:
        image = Image.open(full_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten().tolist(), None
        
    except Exception as e:
        return None, f"Image Error: {str(e)}"

def generate_text_vector(text):
    """商品名などのテキストからCLIPベクトルを生成"""
    if pd.isna(text) or str(text).strip() == "":
        return None, "Empty text"
    
    try:
        inputs = processor(text=str(text), return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten().tolist(), None
        
    except Exception as e:
        return None, f"Text Error: {str(e)}"

def clean_price(price_str):
    if pd.isna(price_str): return 0
    try:
        return int(str(price_str).replace(',', '').replace('¥', '').strip())
    except:
        return 0

# ==========================================
# ジョブ実行関数 (ワーカーから呼ばれる)
# ==========================================
def process_job_logic(job_id, input_csv_path, image_dir_path, mode, progress_callback):
    """実際のCSV処理ロジック"""
    log_messages = []
    
    # パスチェック
    if mode == "画像から生成":
        if not os.path.exists(image_dir_path):
            return None, f"エラー: 画像フォルダが見つかりません: {image_dir_path}"
    
    try:
        df = pd.read_csv(input_csv_path)
        
        # 前処理
        if '価格' in df.columns:
            df['price'] = df['価格'].apply(clean_price)
        if '商品名' in df.columns:
            df['product_name'] = df['商品名']
        
        # カラムチェック
        if mode == "画像から生成" and '画像パス' not in df.columns:
            return None, "エラー: CSVに「画像パス」列がありません。"
        if mode == "商品名から生成" and '商品名' not in df.columns:
            return None, "エラー: CSVに「商品名」列がありません。"

        vectors = []
        success_count = 0
        fail_count = 0
        total = len(df)
        
        for index, row in df.iterrows():
            vec = None
            error_msg = None

            if mode == "画像から生成":
                img_file = row.get('画像パス')
                vec, error_msg = generate_image_vector(img_file, image_dir_path)
            else: 
                text_data = row.get('商品名')
                vec, error_msg = generate_text_vector(text_data)
            
            vectors.append(vec)
            
            if vec is not None:
                success_count += 1
            else:
                fail_count += 1
                if fail_count <= 5: 
                    log_messages.append(f"Row {index} Skip: {error_msg}")
            
            # 進捗更新 (10件ごとまたは最後)
            if index % 10 == 0 or index == total - 1:
                progress_callback(index + 1, total)

        df['feature_vector'] = vectors
        if '画像パス' in df.columns:
            df['image_url'] = df['画像パス']
        else:
            df['image_url'] = ""

        cols_to_save = ['product_name', 'price', 'image_url', 'feature_vector']
        final_cols = [c for c in cols_to_save if c in df.columns]
        output_df = df[final_cols]
        
        # 結果保存
        mode_label = "img" if mode == "画像から生成" else "txt"
        filename = f"{job_id}_{mode_label}_vec.csv"
        output_path = os.path.join(OUTPUT_DIR, filename)
        output_df.to_csv(output_path, index=False)

        status_text = f"完了 (成功:{success_count}, 失敗:{fail_count})"
        if fail_count > 0 and len(log_messages) > 0:
            status_text += f" ※エラー例: {log_messages[0]}"
        
        return output_path, status_text

    except Exception as e:
        err = f"システムエラー: {str(e)}"
        print(err)
        traceback.print_exc()
        return None, err

# ==========================================
# ジョブキュー管理システム
# ==========================================
class JobQueueManager:
    def __init__(self):
        self.queue = [] # List of dicts
        self.lock = threading.Lock()
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def add_job(self, file_obj, image_dir, mode):
        if file_obj is None:
            return None, "ファイルがありません"

        # ファイルを安全な場所にコピー
        job_id = str(uuid.uuid4())[:8]
        name=os.path.splitext(file_obj.name)[0]
        ext = os.path.splitext(file_obj.name)[1]
        safe_input_path = os.path.join(UPLOAD_DIR, f"{name}_{job_id}{ext}")
        shutil.copy(file_obj.name, safe_input_path)

        with self.lock:
            job = {
                "id": job_id,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "status": "待機中",
                "mode": mode,
                "input_file": safe_input_path,
                "image_dir": image_dir,
                "progress": "0/0",
                "result_file": "",
                "message": ""
            }
            self.queue.append(job)
        return job_id, "キューに追加しました"

    def get_job_list(self):
        with self.lock:
            # UI表示用にデータフレーム向けのリストを返す
            # [ID, 時刻, モード, 状態, 進捗, 結果ファイル]
            data = []
            for job in self.queue:
                res_file_name = os.path.basename(job["result_file"]) if job["result_file"] else ""
                data.append([
                    job["id"],
                    job["timestamp"],
                    job["mode"],
                    job["status"],
                    job["progress"],
                    res_file_name
                ])
            return pd.DataFrame(data, columns=["ID", "登録時刻", "モード", "状態", "進捗", "結果ファイル名"])

    def get_result_path(self, job_id):
        with self.lock:
            for job in self.queue:
                if job["id"] == job_id:
                    return job["result_file"]
        return None

    def _worker_loop(self):
        while self.is_running:
            job_to_run = None
            with self.lock:
                for job in self.queue:
                    if job["status"] == "待機中":
                        job_to_run = job
                        job["status"] = "処理中"
                        break
            
            if job_to_run:
                self._execute_job(job_to_run)
            else:
                time.sleep(1)

    def _execute_job(self, job):
        def update_progress(current, total):
            with self.lock:
                job["progress"] = f"{current}/{total}"

        output_path, status_msg = process_job_logic(
            job["id"], job["input_file"], job["image_dir"], job["mode"], update_progress
        )

        with self.lock:
            if output_path:
                job["status"] = "完了"
                job["result_file"] = output_path
                job["message"] = status_msg
            else:
                job["status"] = "エラー"
                job["message"] = status_msg

# グローバルインスタンス
job_manager = JobQueueManager()

# ==========================================
# UI イベントハンドラ
# ==========================================
def submit_job(file, image_dir, mode):
    if file is None:
        return job_manager.get_job_list(), "CSVファイルをアップロードしてください。"
    
    job_id, msg = job_manager.add_job(file, image_dir, mode)
    return job_manager.get_job_list(), f"{msg} (ID: {job_id})"

def refresh_table():
    return job_manager.get_job_list()

def on_select_row(evt: gr.SelectData, current_df):
    # 行がクリックされたら、その行のジョブIDを取得してファイルを返す
    if evt.index is None: return None
    
    row_index = evt.index[0]
    # データフレームからIDを取得 (0列目と仮定)
    # gr.Dataframeの値はそのままリストのリストではない場合があるため注意
    try:
        # current_df が DataFrame の場合
        job_id = current_df.iloc[row_index][0] 
        path = job_manager.get_result_path(job_id)
        if path and os.path.exists(path):
            return path
    except:
        pass
    return None

# ==========================================
# Gradio UI構築
# ==========================================
with gr.Blocks(title="ベクトル化ツール (Queue)") as demo:
    gr.Markdown("## 🛍️ 商品データ ベクトル生成ツール (予約実行版)")
    gr.Markdown("複数のCSVを予約実行できます。下のテーブルで行をクリックすると結果ファイルをダウンロードできます。")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="CSVファイル", file_types=[".csv"])
            mode_input = gr.Radio(
                choices=["画像から生成", "商品名から生成"], 
                value="画像から生成", 
                label="ベクトル化の対象"
            )
            image_dir_input = gr.Textbox(
                label="画像フォルダ (画像モード時)", 
                value=".", 
                placeholder="例: downloaded_images",
            )
            add_btn = gr.Button("キューに追加 (予約)", variant="primary")
            msg_box = gr.Markdown("")

        with gr.Column(scale=1):
            gr.Markdown("### 📋 ジョブ一覧 (クリックしてダウンロード)")
            status_table = gr.Dataframe(
                headers=["ID", "登録時刻", "モード", "状態", "進捗", "結果ファイル名"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                row_count=10
            )
            download_output = gr.File(label="選択した結果のダウンロード")

    # イベント定義
    add_btn.click(
        submit_job, 
        inputs=[file_input, image_dir_input, mode_input], 
        outputs=[status_table, msg_box]
    )

    # 自動更新タイマー (2秒毎)
    timer = gr.Timer(2)
    timer.tick(refresh_table, outputs=status_table)
    
    # テーブルクリックでダウンロード
    status_table.select(
        on_select_row,
        inputs=[status_table],
        outputs=[download_output]
    )

if __name__ == "__main__":
    demo.queue().launch(share=False, server_port=8000)