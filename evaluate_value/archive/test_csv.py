import gradio as gr
import pandas as pd
import re
import os

# --- 設定: CSVパス ---
# メインアプリと同じパスを指定してください
CSV_PATH = "../merged_data_total_6542.csv"

class DataTester:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = self.load_data()

    def load_data(self):
        """CSVデータを読み込み、前処理を行う（メインアプリと同等の処理）"""
        if not os.path.exists(self.csv_path):
            return None
        
        try:
            df = pd.read_csv(self.csv_path)
            # カラム名の統一処理
            rename_map = {"商品名": "product_name", "価格": "price"}
            df = df.rename(columns=rename_map)
            
            # 価格のクリーニング
            if df['price'].dtype == object:
                df['price'] = df['price'].astype(str).str.replace(',', '')
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            df = df.dropna(subset=['price', 'product_name'])
            df['price'] = df['price'].astype(int)
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None

    def search(self, regex_pattern):
        """入力された正規表現で検索を実行"""
        if self.df is None:
            return "CSVファイルが見つかりません", pd.DataFrame()
        
        if not regex_pattern.strip():
            return "正規表現を入力してください", pd.DataFrame()

        try:
            # メインアプリと同じロジック: case=False (大文字小文字無視), regex=True
            mask = self.df['product_name'].str.contains(regex_pattern, case=False, regex=True, na=False)
            results = self.df[mask]
            
            count = len(results)
            message = f"✅ ヒット数: {count} 件"
            
            if count == 0:
                message = "⚠️ ヒットなし。条件を緩めるか、OR(|)を活用してください。"
            
            # 表示用にカラムを絞る
            display_cols = ['product_name', 'price']
            # もし元のCSVに画像URLなどがあればそれも含めるなど調整可能
            
            return message, results[display_cols].head(100) # 重くなるので最大100件表示

        except re.error as e:
            return f"❌ 正規表現エラー: {e}", pd.DataFrame()
        except Exception as e:
            return f"❌ エラー発生: {e}", pd.DataFrame()

    def get_random_samples(self):
        """データの中身を確認するためのランダムサンプリング"""
        if self.df is None: return pd.DataFrame()
        return self.df[['product_name', 'price']].sample(10)

# --- インスタンス化 ---
tester = DataTester(CSV_PATH)

# --- UI構築 ---
with gr.Blocks(title="Regex Sandbox") as demo:
    gr.Markdown("## 🧪 Dragon Eye: 正規表現テストラボ")
    gr.Markdown("AIが生成する予定の「正規表現」を入力して、実際にCSVのどの商品にヒットするか実験できます。")

    with gr.Row():
        with gr.Column(scale=1):
            regex_input = gr.Textbox(
                label="正規表現パターンを入力",
                placeholder="例: (Sony|ソニー).*(イヤホン|ヘッドホン)",
                lines=2
            )
            search_btn = gr.Button("検索実行 (Search)", variant="primary")
            
            with gr.Accordion("📝 正規表現チートシート", open=True):
                gr.Markdown("""
                - **OR検索 (いずれかを含む)**: `(A|B)`  
                  例: `(リゼロ|Re:Zero)` → 「リゼロ」か「Re:Zero」どちらかあればOK
                - **AND検索 (間に文字が入る)**: `A.*B`  
                  例: `ソニー.*イヤホン` → 「ソニー」の後に「イヤホン」があるもの
                - **数字の曖昧検索**: `1000.?XM4`  
                  例: `1000XM4` にも `1000-XM4` にもヒット
                """)

        with gr.Column(scale=2):
            result_msg = gr.Markdown("ここに結果が表示されます")
            result_table = gr.Dataframe(label="検索結果 (最大100件)")

    # データの中身確認用
    with gr.Accordion("🔍 データベースの中身を覗く (ランダム10件)", open=False):
        sample_btn = gr.Button("ランダム表示")
        sample_table = gr.Dataframe()
        sample_btn.click(fn=tester.get_random_samples, outputs=sample_table)

    # イベントハンドラ
    search_btn.click(
        fn=tester.search,
        inputs=regex_input,
        outputs=[result_msg, result_table]
    )

if __name__ == "__main__":
    # ポートを変えてメインアプリと競合しないようにする
    demo.launch(server_port=7861)