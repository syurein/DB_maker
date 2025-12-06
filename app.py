# --- 依存ライブラリのインストール（初回のみ） ---
# !pip install playwright gradio openai pandas beautifulsoup4 python-dotenv nest_asyncio
# !playwright install chromium
# !playwright install-deps

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

import concurrent.futures
import math

# 1. Google Driveマウント (Colab用)
if os.path.exists('/content/drive'):
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/MercariScraper"
else:
    BASE_DIR = "MercariScraper" # ローカル実行用

os.makedirs(BASE_DIR, exist_ok=True)
IMAGE_DIR = os.path.join(BASE_DIR, "downloaded_images")
os.makedirs(IMAGE_DIR, exist_ok=True)
SELECTORS_PATH = os.path.join(BASE_DIR, "selectors.json")
CATEGORY_CSV_PATH = os.path.join(BASE_DIR, "メルカリカテゴリ一覧.csv")
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

# AI設定
DEFAULT_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")

# 定数
TIMEOUT_MS = 150000  # 読み込みタイムアウト

# --- カテゴリマップ ---
def load_category_map(csv_path):
    cat_map = {}
    if os.path.exists(csv_path):
        try:
            try: df = pd.read_csv(csv_path, header=None, encoding='utf-8')
            except: df = pd.read_csv(csv_path, header=None, encoding='cp932')
            for _, row in df.iterrows():
                if pd.notna(row[1]): cat_map[str(row[1]).strip()] = int(row[0])
                if len(row) > 3 and pd.notna(row[3]): cat_map[f"{row[1]} > {row[3]}"] = int(row[2])
        except: pass
    return cat_map

CATEGORY_MAP = load_category_map(CATEGORY_CSV_PATH)
CATEGORY_CHOICES = list(CATEGORY_MAP.keys()) if CATEGORY_MAP else []

# --- セレクタ管理 ---
class SelectorManager:
    def __init__(self):
        self.selectors = self._load()
        updated = False
        if "item_container" not in self.selectors:
            # bs4用セレクタ
            self.selectors["item_container"] = ["li[data-testid='item-cell']", "div[data-testid='item-cell']", ".ItemCell__Item-sc-1"]
            updated = True
        if "title" not in self.selectors:
            self.selectors["title"] = ["img[alt]", "[data-testid='thumbnail-image'] img", ".thumbnail-image img"]
            updated = True
        if "price" not in self.selectors:
            self.selectors["price"] = [".number__6b270ca7", "[data-testid='price']", "span[aria-label*='円']"]
            updated = True
        if updated: self.save()

    def _load(self):
        if os.path.exists(SELECTORS_PATH):
            try:
                with open(SELECTORS_PATH, 'r', encoding='utf-8') as f: return json.load(f)
            except: pass
        return {}

    def save(self):
        try:
            with open(SELECTORS_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.selectors, f, indent=2, ensure_ascii=False)
        except: pass

    def get_candidates(self, key):
        self.selectors = self._load()
        val = self.selectors.get(key, [])
        return [val] if isinstance(val, str) else val

    def add_prioritized(self, key, new_selector):
        current = self.get_candidates(key)
        new_list = [new_selector] + [x for x in current if x != new_selector]
        self.selectors[key] = new_list
        self.save()

# --- 高速解析ロジック (BeautifulSoup版) ---
class FastScraperLogic:
    def __init__(self, api_key, base_url, model_name):
        self.selector_manager = SelectorManager()
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.model_name = model_name

    def _clean_html_for_ai(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup(['script', 'style', 'svg', 'path', 'noscript', 'iframe', 'meta', 'link']):
            tag.decompose()
        return str(soup)[:20000]

    def _ask_ai_for_selector(self, html_snippet, target_description, failed_selectors=None):
        if not self.client: return None
        print(f"🚑 AI Healing (BS4): {target_description}")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a CSS Selector expert. Return JSON: {\"selector\": \"...\"}."}, 
                    {"role": "user", "content": f"Find a standard CSS selector compatible with BeautifulSoup4 for '{target_description}' in this HTML:\n{html_snippet}\nAvoid these: {failed_selectors}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            res = json.loads(response.choices[0].message.content)
            return res.get("selector")
        except: return None

    def parse_page(self, html_content):
        """
        HTMLを受け取り、BeautifulSoupオブジェクトを返す。
        """
        return BeautifulSoup(html_content, 'html.parser')

    def find_items(self, soup):
        """
        Soupから商品コンテナリストを探す (AI修復付き)
        """
        key = "item_container"
        max_retries = 5
        retries = 0
        
        while retries < max_retries:
            candidates = self.selector_manager.get_candidates(key)
            for sel in candidates:
                items = soup.select(sel)
                if items:
                    return items
            
            # AI修復
            print(f"⚠️ {key} が見つかりません。AI修復を実行します... ({retries + 1}/{max_retries})")
            html_snippet = self._clean_html_for_ai(str(soup))
            new_sel = self._ask_ai_for_selector(html_snippet, "Item container element (li or div) in the search result grid", candidates)
            
            if new_sel:
                self.selector_manager.add_prioritized(key, new_sel)
            else:
                # AIがセレクタを提案できなかった場合は、無駄なループを防ぐために終了
                break
                
            retries += 1
            
        return []

    def extract_text(self, item_soup, key, description):
        """
        商品コンテナ(soup)からテキスト/属性を抽出 (AI修復付き)
        """
        max_retries = 5
        retries = 0

        while retries < max_retries:
            candidates = self.selector_manager.get_candidates(key)
            for sel in candidates:
                target = item_soup.select_one(sel)
                if target:
                    if key == "title" and target.name == 'img':
                        val = target.get('alt')
                    else:
                        val = target.get_text(strip=True)
                    if val:
                        return val

            # AI修復
            print(f"⚠️ {key} が見つかりません。AI修復を実行します... ({retries + 1}/{max_retries})")
            item_html_snippet = self._clean_html_for_ai(str(item_soup))
            new_sel = self._ask_ai_for_selector(item_html_snippet, description, candidates)

            if new_sel:
                self.selector_manager.add_prioritized(key, new_sel)
            else:
                break
            
            retries += 1
            
        return ""
    
    def extract_image_url(self, item_soup):
        # 画像URL取得（高速化のため固定ロジック＋簡易探索）
        # Mercariは通常 img タグの src または data-src
        img = item_soup.select_one("img")
        if img:
            return img.get('src') or img.get('data-src')
        return ""
    
    def extract_product_url(self, item_soup):
        a_tag = item_soup.select_one("a")
        if a_tag:
            href = a_tag.get('href')
            if href: return f"https://jp.mercari.com{href}"
        return ""

# --- 画像ダウンロード関数 (requests使用) ---
def download_image_fast(url, save_path):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            with open(save_path, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

# --- 並列ワーカー (Playwright -> BS4) ---
def worker_process(worker_id, keyword, category_id, status_param, price_min, price_max, sort_val, order_val, start_page, limit_per_worker, api_key, base_url, model, download_images):
    print(f"🚀 Worker {worker_id}: 開始")
    
    logic = FastScraperLogic(api_key, base_url, model)
    results = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        # 画像読み込みをブロックしない（遅延ロード画像を取得するため）
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        count = 0
        current_page_idx = start_page
        
        while count < limit_per_worker:
            page_token = f"v1%3A{current_page_idx}"
            url = f"https://jp.mercari.com/search?keyword={quote(keyword)}&status={status_param}&sort={sort_val}&order={order_val}&page_token={page_token}"
            if category_id: url += f"&category_id={category_id}"
            if price_min: url += f"&price_min={price_min}"
            if price_max: url += f"&price_max={price_max}"

            print(f"🌍 Worker {worker_id}: Accessing Page {current_page_idx}...")
            
            try:
                # ページ遷移
                page.goto(url, timeout=TIMEOUT_MS, wait_until="domcontentloaded")
                
                # ★ここが重要: 画像の遅延読み込み (Lazy Loading) を発火させるために少しスクロール
                # Mercariはスクロールしないとimg srcが空の場合がある
                page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                time.sleep(0.5)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(3.0) # 画像ロード待ち
                
            except Exception as e:
                print(f"⚠️ Worker {worker_id}: 読み込みタイムアウト/エラー (HTML解析は続行) - {e}")
                try: page.evaluate("window.stop()")
                except: pass

            # ★ここから爆速パート: HTMLを文字列として取得してBS4に渡す
            html = page.content()
            soup = logic.parse_page(html)
            
            # BS4で解析
            items = logic.find_items(soup)
            
            if not items:
                print(f"❌ Worker {worker_id}: 商品なし (終了)")
                break

            print(f"⚡ Worker {worker_id}: BS4で {len(items)} 件を解析中...")

            for item in items:
                if count >= limit_per_worker: break
                
                try:
                    # BS4でテキスト抽出（メモリ処理なので一瞬）
                    title = logic.extract_text(item, "title", "商品名")
                    if title: title = title.replace("のサムネイル", "").strip()
                    
                    price = logic.extract_text(item, "price", "価格")
                    
                    img_src = logic.extract_image_url(item)
                    product_url = logic.extract_product_url(item)

                    title = title or "取得失敗"
                    price = price or "0"
                    
                    img_filename = "SKIP"
                    if download_images and img_src:
                        safe_name = f"{worker_id}_{current_page_idx}_{count}_{int(time.time())}.jpg"
                        save_path = os.path.join(IMAGE_DIR, safe_name)
                        if download_image_fast(img_src, save_path):
                            img_filename = safe_name
                    
                    row = {"商品名": title, "価格": price, "画像パス": img_filename, "URL": product_url}
                    results.append(row)
                    count += 1
                except Exception as e:
                    continue
            
            current_page_idx += 1
            
        browser.close()
    
    print(f"✅ Worker {worker_id}: 完了 ({len(results)}件)")
    return results

# --- メインクラス ---
class MercariFastScraper:
    def __init__(self, api_key, base_url, model_name):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def run(self, keyword, category_id, status, price_min, price_max, sort_order, total_limit, num_workers, download_images, progress=gr.Progress()):
        status_param = "on_sale%7Csold_out" if status == "すべて" else ("sold_out" if status == "売り切れ" else "on_sale")
        sort_map = {
            "おすすめ順": ("score", "desc"), "新しい順": ("created_time", "desc"),
            "価格の安い順": ("price", "asc"), "価格の高い順": ("price", "desc"), "いいね！順": ("num_likes", "desc")
        }
        sort_val, order_val = sort_map.get(sort_order, ("score", "desc"))

        safe_kw = "".join([c for c in keyword if c.isalnum()])
        csv_filename = os.path.join(BASE_DIR, f"{safe_kw}_{total_limit}件_爆速版.csv")
        
        # 1ページあたり約100件と仮定して、Workerへの割り振りを計算
        limit_per_worker = math.ceil(total_limit / num_workers)
        
        print(f"🔥 爆速スクレイピング開始: {num_workers} workers, BS4解析, 画像DL={download_images}")
        
        futures = []
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i in range(num_workers):
                # 開始ページをずらす (Worker 1: 0~, Worker 2: 5~ ...)
                start_page = i * 5 
                futures.append(
                    executor.submit(
                        worker_process, 
                        worker_id=i,
                        keyword=keyword,
                        category_id=category_id,
                        status_param=status_param,
                        price_min=price_min,
                        price_max=price_max,
                        sort_val=sort_val,
                        order_val=order_val,
                        start_page=start_page,
                        limit_per_worker=limit_per_worker,
                        api_key=self.api_key,
                        base_url=self.base_url,
                        model=self.model_name,
                        download_images=download_images
                    )
                )
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                all_results.extend(res)
                completed += 1
                progress(completed / num_workers, desc=f"Worker完了: {completed}/{num_workers}")

        if all_results:
            df = pd.DataFrame(all_results)
            # 重複除去（URLベース）
            df = df.drop_duplicates(subset=["URL"], keep='first')
            df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
            return f"完了！ 合計{len(df)}件取得しました。\nファイル: {csv_filename}", csv_filename
        else:
            return "エラー: データなし", None

# --- UI ---
def start_scraping(api_key, keyword, category_name, limit, status, price_min, price_max, sort_order, workers, download_images):
    use_api_key = api_key if api_key else DEFAULT_API_KEY
    workers = int(workers)
    if workers > 4: workers = 4
    
    scraper = MercariFastScraper(use_api_key, DEFAULT_BASE_URL, DEFAULT_MODEL)
    cat_id = CATEGORY_MAP.get(category_name)
    
    return scraper.run(keyword, cat_id, status, price_min, price_max, sort_order, int(limit), workers, download_images)

with gr.Blocks() as demo:
    gr.Markdown("## メルカリAIスクレイピング (爆速 BS4ハイブリッド版)")
    gr.Markdown("Playwrightでロードし、BeautifulSoupで瞬時に解析します。")
    
    with gr.Accordion("API設定", open=False):
        api_key_input = gr.Textbox(label="API Key", type="password")
    
    with gr.Row():
        keyword_input = gr.Textbox(label="検索キーワード", value="ニンテンドー3DS")
        limit_input = gr.Number(label="目標合計取得件数", value=100, precision=0)
    
    with gr.Row():
        category_input = gr.Dropdown(label="カテゴリ", choices=CATEGORY_CHOICES)
        status_input = gr.Dropdown(label="状態", choices=["販売中", "売り切れ", "すべて"], value="販売中")
    
    with gr.Row():
        price_min_input = gr.Number(label="価格下限")
        price_max_input = gr.Number(label="価格上限")
        sort_input = gr.Dropdown(label="並び順", choices=["おすすめ順", "新しい順", "価格の安い順", "価格の高い順", "いいね！順"], value="おすすめ順")

    with gr.Row():
        workers_input = gr.Slider(label="並列ワーカー数", minimum=1, maximum=4, value=2, step=1)
        image_dl_input = gr.Checkbox(label="画像をダウンロードする", value=True)

    btn = gr.Button("開始", variant="primary")
    output_log = gr.Textbox(label="ログ")
    output_file = gr.File(label="CSV")

    btn.click(
        start_scraping, 
        inputs=[
            api_key_input, keyword_input, category_input, limit_input, 
            status_input, price_min_input, price_max_input, sort_input, 
            workers_input, image_dl_input
        ], 
        outputs=[output_log, output_file]
    )

demo.queue().launch(share=True, debug=True)