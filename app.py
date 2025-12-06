import os
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright
import gradio as gr
from urllib.parse import quote

# 設定読み込み
load_dotenv()

# --- 定数・設定 ---
IMAGE_DIR = "downloaded_images"
os.makedirs(IMAGE_DIR, exist_ok=True)
SELECTORS_PATH = "selectors.json"

# AI設定
DEFAULT_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "True")

# カテゴリマップ読み込み
CATEGORY_CSV_PATH = "メルカリカテゴリ一覧.csv"

def load_category_map(csv_path):
    cat_map = {}
    if os.path.exists(csv_path):
        try:
            try:
                df = pd.read_csv(csv_path, header=None, encoding='utf-8')
            except:
                df = pd.read_csv(csv_path, header=None, encoding='cp932')
            for _, row in df.iterrows():
                if pd.notna(row[1]): cat_map[str(row[1]).strip()] = int(row[0])
                if len(row) > 3 and pd.notna(row[3]): cat_map[f"{row[1]} > {row[3]}"] = int(row[2])
        except: pass
    return cat_map

CATEGORY_MAP = load_category_map(CATEGORY_CSV_PATH)
CATEGORY_CHOICES = list(CATEGORY_MAP.keys()) if CATEGORY_MAP else []

# --- セレクタ管理クラス ---
class SelectorManager:
    def __init__(self):
        self.selectors = self._load()
        # 初期値がない場合はデフォルトを設定
        updated = False
        if "item_container" not in self.selectors:
            self.selectors["item_container"] = ["li[data-testid='item-cell']", "div[data-testid='item-cell']"]
            updated = True
        if "title" not in self.selectors:
            self.selectors["title"] = ["img[alt]", "[data-testid='thumbnail-image']"]
            updated = True
        if "price" not in self.selectors:
            self.selectors["price"] = [".number__6b270ca7", "[data-testid='price']"]
            updated = True
        
        # 初期値を入れた場合も即保存
        if updated:
            self.save()

    def _load(self):
        if os.path.exists(SELECTORS_PATH):
            try:
                with open(SELECTORS_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {}

    def save(self):
        """現在のセレクタ情報をJSONファイルに書き込む"""
        try:
            with open(SELECTORS_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.selectors, f, indent=2, ensure_ascii=False)
            print(f"💾 セレクタ設定をJSONに保存しました: {SELECTORS_PATH}")
        except Exception as e:
            print(f"❌ JSON保存エラー: {e}")

    def get_candidates(self, key):
        val = self.selectors.get(key, [])
        if isinstance(val, str): return [val]
        return val

    def add_prioritized(self, key, new_selector):
        """新しいセレクタをリストの先頭に追加して保存する"""
        print(f"🔄 セレクタ更新・優先順位変更: {key} -> {new_selector}")
        current = self.get_candidates(key)
        # 重複を除きつつ、新しいセレクタを先頭に追加
        new_list = [new_selector] + [x for x in current if x != new_selector]
        self.selectors[key] = new_list
        # ★ここで保存を実行
        self.save()

# --- スクレイパー本体 ---
class MercariSmartScraper:
    def __init__(self, api_key, base_url, model_name):
        self.selector_manager = SelectorManager()
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _clean_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup(['script', 'style', 'svg', 'path', 'noscript', 'iframe', 'meta', 'link']):
            tag.decompose()
        return str(soup)[:30000]

    def _ask_ai_for_selector(self, html_snippet, target_description, failed_selectors=None):
        if not self.client:
            print("⚠️ API KeyがないためAI修復をスキップします")
            return None

        print(f"🚑 AI Healingリクエスト: {target_description}")
        
        system_prompt = "あなたはWebスクレイピングの専門家です。CSSセレクタのみをJSON形式で返してください。"
        user_prompt = f"""
        以下のHTMLコンテキストから、「{target_description}」を特定するCSSセレクタを見つけてください。
        
        【除外リスト】
        {json.dumps(failed_selectors)}

        【条件】
        - 安定した属性（data-testid, aria-label等）を優先。
        - なければclass属性などを使用。
        - 1つだけ提案してください。

        【HTML】
        {html_snippet}

        【出力形式 (JSON)】
        {{"selector": "あなたの答え"}}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            cleaned = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)
            new_selector = result.get("selector")
            print(f"✨ AI提案: {new_selector}")
            return new_selector
        except Exception as e:
            print(f"❌ AI Error: {e}")
            return None

    def _find_elements_with_healing(self, page, key, description):
        candidates = self.selector_manager.get_candidates(key)
        
        # 1. キャッシュされた候補を試す
        for sel in candidates:
            try:
                count = page.locator(sel).count()
                if count > 0:
                    return page.locator(sel).all()
            except: pass
        
        # 2. 全候補試して全部0件 -> AI Healing発動
        print(f"⚠️ {key} の既存セレクタでは0件でした。AI修復を実行します...")
        
        try:
            html_content = page.content()
            if not html_content or len(html_content) < 100:
                print("❌ ページの内容が空です。修復できません。")
                return []
        except: return []

        html_context = self._clean_html(html_content)
        new_sel = self._ask_ai_for_selector(html_context, description, candidates)
        
        if new_sel:
            # ★ここで新しいセレクタを追加＆保存
            self.selector_manager.add_prioritized(key, new_sel)
            try:
                count = page.locator(new_sel).count()
                if count > 0:
                    print(f"✅ 修復成功！ {count}件見つかりました。")
                    return page.locator(new_sel).all()
            except: pass
            
        return []

    def _get_text_with_healing(self, item_locator, key, description):
        candidates = self.selector_manager.get_candidates(key)
        for sel in candidates:
            try:
                target = item_locator.locator(sel).first
                if key == "title" and "img" in sel:
                    text = target.get_attribute("alt")
                else:
                    text = target.inner_text()
                if text and text.strip(): return text.strip()
            except: pass
        
        # 取得失敗時のHealing (アイテム単体)
        try:
            item_html = item_locator.inner_html()
            # print(f"⚠️ {key} が空でした。修復中...") 
            new_sel = self._ask_ai_for_selector(self._clean_html(item_html), description, candidates)
            
            if new_sel:
                # ★ここで新しいセレクタを追加＆保存
                self.selector_manager.add_prioritized(key, new_sel)
                try:
                    target = item_locator.locator(new_sel).first
                    if key == "title" and "img" in new_sel:
                        return target.get_attribute("alt")
                    return target.inner_text()
                except: pass
        except: pass

        return "" 

    def run(self, keyword, category_id, status, price_min, price_max, limit, progress=gr.Progress()):
        results = []
        status_param = "on_sale%7Csold_out" if status == "すべて" else ("sold_out" if status == "売り切れ" else "on_sale")
        safe_kw = "".join([c for c in keyword if c.isalnum()])
        csv_filename = f"{safe_kw}_{limit}件.csv"
        pd.DataFrame(columns=["商品名", "価格", "画像パス", "URL"]).to_csv(csv_filename, index=False, encoding="utf-8-sig")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=HEADLESS_MODE.lower() == "true")
            context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
            page = context.new_page()

            count = 0
            page_idx = 0
            
            while count < limit:
                page_token = f"v1%3A{page_idx}"
                url = f"https://jp.mercari.com/search?keyword={quote(keyword)}&status={status_param}&sort=created_time&order=desc&page_token={page_token}"
                if category_id: url += f"&category_id={category_id}"
                if price_min: url += f"&price_min={price_min}"
                if price_max: url += f"&price_max={price_max}"

                print(f"🌍 Accessing Page {page_idx}: {url}")
                
                try:
                    page.goto(url, timeout=30000)
                    page.wait_for_load_state("networkidle", timeout=5000)
                    time.sleep(2)
                except Exception as e:
                    if "Timeout" in str(e):
                        print("⚠️ タイムアウト発生: 読み込みを強制停止し、現在の表示状態で解析を試みます...")
                        try:
                            page.evaluate("window.stop()")
                        except: pass
                    else:
                        print(f"❌ 致命的なエラー: {e}")
                        break

                items = self._find_elements_with_healing(
                    page, 
                    "item_container", 
                    "検索結果一覧の個々の商品を囲むコンテナ要素(liタグやdivタグ)"
                )

                if not items:
                    print("❌ 商品が見つかりませんでした（AI修復後も0件）。")
                    break

                print(f"✅ {len(items)}件の商品を検出 (Page {page_idx})")

                page_results = []
                for item in items:
                    if count >= limit: break
                    try:
                        title = self._get_text_with_healing(item, "title", "商品名のテキストまたは画像のalt属性")
                        if title: title = title.replace("のサムネイル", "").strip()
                        price = self._get_text_with_healing(item, "price", "商品の価格（数字を含む要素）")
                        
                        try: img_src = item.locator("img").first.get_attribute("src")
                        except: img_src = ""
                        try: href = item.locator("a").first.get_attribute("href")
                        except: href = ""
                        product_url = f"https://jp.mercari.com{href}" if href else ""

                        title = title or "取得失敗"
                        price = price or "0"

                        img_filename = ""
                        if img_src:
                            try:
                                img_data = requests.get(img_src, timeout=5).content
                                safe_name = f"{count}_{int(time.time())}.jpg"
                                img_path = os.path.join(IMAGE_DIR, safe_name)
                                with open(img_path, "wb") as f: f.write(img_data)
                                img_filename = safe_name
                            except: pass

                        row = {"商品名": title, "価格": price, "画像パス": img_filename, "URL": product_url}
                        page_results.append(row)
                        results.append(row)
                        count += 1
                        progress(count / limit, desc=f"取得中... {count}/{limit}件")
                    except Exception: continue
                
                if page_results:
                    pd.DataFrame(page_results).to_csv(csv_filename, mode='a', header=False, index=False, encoding="utf-8-sig")
                
                page_idx += 1
                if len(items) == 0: break

            browser.close()
            
        return f"完了！ {len(results)}件取得しました。\nファイル: {csv_filename}", csv_filename

# --- Gradio UI ---
def start_scraping(api_key, keyword, category_name, limit, status, price_min, price_max):
    use_api_key = api_key if api_key else DEFAULT_API_KEY
    if not use_api_key: return "エラー: AIのAPIキーが必要です。", None
    scraper = MercariSmartScraper(use_api_key, DEFAULT_BASE_URL, DEFAULT_MODEL)
    cat_id = CATEGORY_MAP.get(category_name)
    return scraper.run(keyword, cat_id, status, price_min, price_max, int(limit))

with gr.Blocks() as demo:
    gr.Markdown("## メルカリAIスクレイピング (学習機能付き)")
    gr.Markdown("AIが修復したセレクタは `selectors.json` に保存され、次回から自動的に使用されます。")
    with gr.Accordion("API設定", open=False):
        api_key_input = gr.Textbox(label="API Key", type="password")
    with gr.Row():
        keyword_input = gr.Textbox(label="検索キーワード", value="ニンテンドー3DS")
        limit_input = gr.Number(label="目標取得件数", value=50, precision=0)
    with gr.Row():
        category_input = gr.Dropdown(label="カテゴリ", choices=CATEGORY_CHOICES)
        status_input = gr.Dropdown(label="状態", choices=["販売中", "売り切れ", "すべて"], value="販売中")
    with gr.Row():
        price_min_input = gr.Number(label="価格下限")
        price_max_input = gr.Number(label="価格上限")
    btn = gr.Button("開始", variant="primary")
    output_log = gr.Textbox(label="ログ")
    output_file = gr.File(label="CSV")
    btn.click(start_scraping, inputs=[api_key_input, keyword_input, category_input, limit_input, status_input, price_min_input, price_max_input], outputs=[output_log, output_file])

if __name__ == "__main__":
    demo.queue().launch()