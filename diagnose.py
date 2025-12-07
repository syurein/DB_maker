import gradio as gr
import pandas as pd
import os

def diagnose_paths(csv_file, target_dir):
    logs = []
    
    def log(message):
        logs.append(message)
    
    log("=== 🔍 パス診断ツール (Gradio版) ===")
    
    # 1. 入力チェック
    if csv_file is None:
        return "エラー: CSVファイルをアップロードしてください。"
    
    target_dir = str(target_dir).strip()
    if not target_dir:
        target_dir = "." # デフォルト
    
    # 2. 実行環境とフォルダ確認
    current_dir = os.getcwd()
    log(f"📂 現在のスクリプト実行場所 (Current Dir): {current_dir}")
    
    abs_target_dir = os.path.abspath(target_dir)
    log(f"📂 実際に探しに行くフォルダ (絶対パス): {abs_target_dir}")
    
    if not os.path.exists(abs_target_dir):
        log("\n❌ 【致命的エラー】 指定されたフォルダ自体が存在しません。")
        log("   → 入力したパスが間違っているか、相対パスの基準場所が違います。")
        log("   → 確実なのは「絶対パス」(例: C:\\Users\\...\\images) を入力することです。")
        return "\n".join(logs)

    # 3. フォルダの中身確認
    try:
        files_in_dir = os.listdir(abs_target_dir)
        # 画像っぽい拡張子のファイルを探す
        jpg_files = [f for f in files_in_dir if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        
        log(f"\nℹ️ 指定フォルダ内のファイル総数: {len(files_in_dir)}")
        log(f"ℹ️ 画像ファイルの数: {len(jpg_files)}")
        
        if len(jpg_files) == 0:
            log("⚠️ 【警告】 指定フォルダに画像ファイルが1つもありません。")
            
            # サブフォルダチェック
            sub_dirs = [d for d in files_in_dir if os.path.isdir(os.path.join(abs_target_dir, d))]
            if sub_dirs:
                log(f"   💡 以下のサブフォルダが見つかりました: {sub_dirs}")
                if "downloaded_images" in sub_dirs:
                    log("   🚀 「downloaded_images」というフォルダがあります！")
                    log(f"   → もしかして、パス欄に '{os.path.join(target_dir, 'downloaded_images')}' と入力すべきではありませんか？")
        else:
            log(f"   ℹ️ ファイル名の例 (先頭3件): {jpg_files[:3]}")
            
    except Exception as e:
        log(f"❌ フォルダアクセスエラー: {e}")
        return "\n".join(logs)

    # 4. CSV照合
    log("\n=== 📄 CSVとの照合テスト ===")
    try:
        df = pd.read_csv(csv_file.name)
        if '画像パス' not in df.columns:
            log("❌ CSVに「画像パス」列がありません。列名を確認してください。")
            return "\n".join(logs)
            
        log(f"データ件数: {len(df)}件")
        log("--- 先頭5件のパスチェック ---")
        
        found_count = 0
        check_limit = 5
        
        for i, row in df.head(check_limit).iterrows():
            filename = str(row['画像パス']).strip()
            if not filename or filename == "nan":
                log(f"[{i}] スキップ (ファイル名なし)")
                continue
                
            expected_path = os.path.join(abs_target_dir, filename)
            exists = os.path.exists(expected_path)
            
            status = "✅ OK" if exists else "❌ NOT FOUND"
            log(f"[{i}] CSV上のファイル名: '{filename}'")
            log(f"    探した場所: {expected_path}")
            log(f"    結果: {status}")
            
            if exists:
                found_count += 1
            elif not exists:
                # 典型的なミスのヒント: サブフォルダにあるか？
                sub_path = os.path.join(abs_target_dir, "downloaded_images", filename)
                if os.path.exists(sub_path):
                    log(f"    💡 ヒント: サブフォルダ 'downloaded_images' の中にならありました！")
                    log(f"    → パス欄の指定を '{os.path.join(target_dir, 'downloaded_images')}' に変えてください。")
            
            log("-" * 20)
            
        if found_count == 0 and len(df) > 0:
            log("\n⚠️ 【結論】 先頭5件すべて見つかりませんでした。")
            log("パスの設定が間違っています。ログの「探した場所」と、実際の画像がある場所を見比べてください。")
        elif found_count == check_limit:
            log("\n🎉 【結論】 パス設定は正しいようです！この設定で本番ツールを動かしてください。")
        else:
            log("\n⚠️ 【結論】 一部は見つかりましたが、一部が見つかりません。ファイル名の不一致などを確認してください。")

    except Exception as e:
        log(f"CSV読み込みエラー: {e}")

    return "\n".join(logs)

# UI構築
with gr.Blocks(title="パス設定診断ツール") as demo:
    gr.Markdown("## 🔍 パス設定診断ツール")
    gr.Markdown("「画像が見つからない」原因を特定します。対象のCSVと、本番ツールに入力しようとしている画像フォルダパスを指定してください。")
    
    with gr.Row():
        file_input = gr.File(label="CSVファイル", file_types=[".csv"])
        path_input = gr.Textbox(
            label="画像フォルダのパス (検証したいパス)", 
            value=".", 
            placeholder="例: . または downloaded_images または C:/Users/.../images"
        )
    
    btn = gr.Button("診断開始", variant="primary")
    output_log = gr.Textbox(label="診断ログ", lines=20)
    
    btn.click(diagnose_paths, inputs=[file_input, path_input], outputs=[output_log])

if __name__ == "__main__":
    demo.launch(share=False, server_port=8001) # ポートが被らないように8001にしています