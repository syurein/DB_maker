import gradio as gr
import pandas as pd
import os
import tempfile

def merge_csv_files(files, unique_col_name):
    """
    複数のCSVファイルを読み込んで結合し、指定カラムで重複削除を行う
    """
    if not files:
        return None, "エラー: ファイルがアップロードされていません。"

    dfs = []
    total_rows_before = 0
    file_logs = []

    # 1. 各ファイルを読み込む
    for file in files:
        try:
            # encoding='utf-8' で試してだめなら 'cp932' (Shift_JIS) で読むなどの配慮
            try:
                df = pd.read_csv(file.name, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file.name, encoding='cp932')
            
            dfs.append(df)
            total_rows_before += len(df)
            file_logs.append(f"・{os.path.basename(file.name)}: {len(df)}件")
        except Exception as e:
            file_logs.append(f"× 読み込み失敗 {os.path.basename(file.name)}: {e}")

    if not dfs:
        return None, "有効なCSVファイルがありませんでした。\n" + "\n".join(file_logs)

    # 2. 結合 (concat)
    # カラムが不揃いでも、列名が同じなら自動的に縦に繋がります
    merged_df = pd.concat(dfs, ignore_index=True)
    
    log_msg = "【結合レポート】\n" + "\n".join(file_logs)
    log_msg += f"\n----------------\n結合後の合計行数: {len(merged_df)}件\n"

    # 3. 重複削除
    if unique_col_name and unique_col_name in merged_df.columns:
        before_dedup = len(merged_df)
        
        # 指定カラム(URLなど)で重複を削除。keep='last' で新しい方(リストの後ろ)を残すか、'first'で最初を残すか
        # ここでは 'first' (先に読み込んだファイルを優先) にしています
        merged_df = merged_df.drop_duplicates(subset=[unique_col_name], keep='first')
        
        removed_count = before_dedup - len(merged_df)
        log_msg += f"重複削除 ({unique_col_name}): -{removed_count}件\n"
        log_msg += f"最終的な行数: {len(merged_df)}件\n"
    elif unique_col_name:
        log_msg += f"⚠️ 警告: カラム「{unique_col_name}」が見つからないため、重複削除は行われませんでした。\n"

    # 4. 保存
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"merged_data_total_{len(merged_df)}.csv")
    
    # 日本語文字化け防止のため utf-8-sig
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    return output_path, log_msg

# UI構築
with gr.Blocks(title="CSV結合ツール") as demo:
    gr.Markdown("## 🔗 複数CSV 結合＆重複削除ツール")
    gr.Markdown("分割して作成したCSVファイルをまとめて、一つのファイルにします。")
    
    with gr.Row():
        with gr.Column():
            # file_count="multiple" で複数選択可能にする
            file_input = gr.File(
                label="結合したいCSVファイル (複数選択可)", 
                file_count="multiple", 
                file_types=[".csv"]
            )
            
            unique_col_input = gr.Textbox(
                label="重複削除をする基準のカラム名 (空欄なら削除なし)", 
                value="URL", 
                placeholder="例: URL または 商品名"
            )
            
            btn = gr.Button("結合を実行", variant="primary")

        with gr.Column():
            log_output = gr.Textbox(label="実行ログ", lines=10)
            file_output = gr.File(label="結合済みCSVのダウンロード")

    btn.click(
        merge_csv_files, 
        inputs=[file_input, unique_col_input], 
        outputs=[file_output, log_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=8002)