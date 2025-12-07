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
import threading
import uuid
from datetime import datetime

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
TIMEOUT_MS = 300000  # 読み込みタイムアウト

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
    def __init__(self, api_key, base_url, model_name, use_ai_healing=True):
        self.selector_manager = SelectorManager()
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.model_name = model_name
        self.use_ai_healing = use_ai_healing

    def _clean_html_for_ai(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup(['script', 'style', 'svg', 'path', 'noscript', 'iframe', 'meta', 'link']):
            tag.decompose()
        return str(soup)[:200000]

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
        return BeautifulSoup(html_content, 'html.parser')

    def find_items(self, soup):
        key = "item_container"
        max_retries = 5
        retries = 0
        
        while retries < max_retries:
            candidates = self.selector_manager.get_candidates(key)
            for sel in candidates:
                items = soup.select(sel)
                if items:
                    return items
            
            if not self.use_ai_healing:
                break

            print(f"⚠️ {key} が見つかりません。AI修復を実行します... ({retries + 1}/{max_retries})")
            html_snippet = self._clean_html_for_ai(str(soup))
            new_sel = self._ask_ai_for_selector(html_snippet, "Item container element (li or div) in the search result grid", candidates)
            
            if new_sel:
                self.selector_manager.add_prioritized(key, new_sel)
            else:
                break
                
            retries += 1
            
        return []

    def extract_text(self, item_soup, key, description):
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
            
            if not self.use_ai_healing:
                break

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

# --- 画像ダウンロード関数 ---
def download_image_fast(url, save_path):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            with open(save_path, "wb") as f: f.write(r.content)
            return True
    except: pass
    return False

# --- 並列ワーカー (Playwright -> BS4) ---
def worker_process(worker_id, keyword, category_id, status_param, price_min, price_max, sort_val, order_val, start_page, shared_counter, total_limit, num_workers, api_key, base_url, model, download_images, use_ai_healing, headless_mode, csv_filename,sleep_time=30):
    print(f"🚀 Worker {worker_id}: 開始 (担当ページ: {start_page}, {start_page + num_workers}, ...)")
    
    logic = FastScraperLogic(api_key, base_url, model, use_ai_healing)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless_mode)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        current_page_idx = start_page
        skip_counter = 0
        max_skips = 3
        
        while shared_counter.value < total_limit:
            page_token = f"v1%3A{current_page_idx}"
            url = f"https://jp.mercari.com/search?keyword={quote(keyword)}&status={status_param}&sort={sort_val}&order={order_val}&page_token={page_token}"
            if category_id: url += f"&category_id={category_id}"
            if price_min: url += f"&price_min={price_min}"
            if price_max: url += f"&price_max={price_max}"

            # print(f"🌍 Worker {worker_id}: Accessing Page {current_page_idx}...")
            
            try:
                page.goto(url, timeout=TIMEOUT_MS, wait_until="domcontentloaded")
                page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                time.sleep(sleep_time) # 少し短縮
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(sleep_time) # 少し短縮
                
            except Exception as e:
                print(f"⚠️ Worker {worker_id}: 読み込みエラー (HTML解析は続行) - {e}")
                try: page.evaluate("window.stop()")
                except: pass

            html = page.content()
            if not html:
                current_page_idx += num_workers
                skip_counter += 1
                if skip_counter >= max_skips: break
                continue
                
            soup = logic.parse_page(html)
            items = logic.find_items(soup)
            
            if not items:
                current_page_idx += num_workers
                skip_counter += 1
                if skip_counter >= max_skips: break
                continue

            skip_counter = 0
            
            page_results_to_write = []
            for item in items:
                if shared_counter.value >= total_limit: break
                
                try:
                    title = logic.extract_text(item, "title", "商品名")
                    if title: title = title.replace("のサムネイル", "").strip()
                    price = logic.extract_text(item, "price", "価格")
                    img_src = logic.extract_image_url(item)
                    product_url = logic.extract_product_url(item)
                    title = title or "取得失敗"
                    price = price or "0"
                    
                    img_filename = "SKIP"
                    is_valid = not download_images

                    if download_images and img_src:
                        safe_name = f"{worker_id}_{current_page_idx}_{len(page_results_to_write)}_{int(time.time())}.jpg"
                        save_path = os.path.join(IMAGE_DIR, safe_name)
                        if download_image_fast(img_src, save_path):
                            img_filename = safe_name
                            is_valid = True
                    
                    if is_valid:
                        row = {"商品名": title, "価格": price, "画像パス": img_filename, "URL": product_url}
                        page_results_to_write.append(row)

                except Exception as e:
                    continue
            
            if page_results_to_write:
                pd.DataFrame(page_results_to_write).to_csv(csv_filename, mode='a', header=False, index=False, encoding="utf-8-sig")
                shared_counter.increment(len(page_results_to_write))
                # print(f"📦 Worker {worker_id}: {len(page_results_to_write)}件追加")

            if shared_counter.value >= total_limit:
                break
            
            current_page_idx += num_workers
            
        browser.close()
    return

# --- 共有カウンター ---
class SharedCounter:
    def __init__(self, initial_value=0):
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self, value=1):
        with self._lock:
            self._value += value
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value

# --- スクレイパークラス (コールバック対応版) ---
class MercariFastScraper:
    def __init__(self, api_key, base_url, model_name):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def run(self, keyword, category_id, category_name, status, price_min, price_max, sort_order, total_limit, num_workers, download_images, use_ai_healing, headless_mode, progress_callback=None,sleep_time=30):
        status_param = "on_sale%7Csold_out" if status == "すべて" else ("sold_out" if status == "売り切れ" else "on_sale")
        sort_map = {
            "おすすめ順": ("score", "desc"), "新しい順": ("created_time", "desc"),
            "価格の安い順": ("price", "asc"), "価格の高い順": ("price", "desc"), "いいね！順": ("num_likes", "desc")
        }
        sort_val, order_val = sort_map.get(sort_order, ("score", "desc"))

        # ファイル名生成
        filename_parts = []
        if keyword: filename_parts.append("".join([c for c in keyword if c.isalnum()]))
        if category_name:
            sanitized_cat = category_name.replace(">", "_").replace(" ", "").replace("/", "_")
            filename_parts.append(sanitized_cat)
        if status: filename_parts.append(status)
        if price_min or price_max:
            price_part = ""
            if price_min: price_part += f"min{price_min}"
            if price_max: price_part += f"max{price_max}"
            filename_parts.append(price_part)
        if sort_order: filename_parts.append(sort_order)
        filename_parts.append(f"{total_limit}件")
        
        safe_filename = "_".join(filter(None, filename_parts)) + ".csv"
        csv_filename = os.path.join(BASE_DIR, safe_filename)
        
        pd.DataFrame(columns=["商品名", "価格", "画像パス", "URL"]).to_csv(csv_filename, index=False, encoding="utf-8-sig")
        
        print(f"🔥 スクレイピング開始: {keyword} / {category_name}")
        
        futures = []
        shared_counter = SharedCounter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i in range(num_workers):
                futures.append(executor.submit(
                    worker_process, i, keyword, category_id, status_param, price_min, price_max, sort_val, order_val, 
                    i, shared_counter, total_limit, num_workers, self.api_key, self.base_url, self.model_name, 
                    download_images, use_ai_healing, headless_mode, csv_filename, sleep_time
                ))
            
            # 監視ループ
            while any(f.running() for f in futures):
                if progress_callback:
                    progress_callback(shared_counter.value, total_limit)
                time.sleep(1)
            
            # 完了待機
            for future in futures:
                try: future.result()
                except: pass

        if progress_callback:
            progress_callback(shared_counter.value, total_limit)

        try:
            df = pd.read_csv(csv_filename)
            df = df.drop_duplicates(subset=["URL"], keep='first')
            if len(df) > total_limit: df = df.head(total_limit)
            df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
            return len(df), csv_filename
        except:
            return 0, csv_filename

# --- ジョブキュー管理システム ---
class JobQueueManager:
    def __init__(self):
        self.queue = [] # List of dicts
        self.lock = threading.Lock()
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def add_job(self, params):
        with self.lock:
            job_id = str(uuid.uuid4())[:8]
            job = {
                "id": job_id,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "keyword": params.get("keyword", "-"),
                "category": params.get("category_name", "-"),
                "status": "待機中",
                "progress": "0/0",
                "result_file": "",
                "params": params
            }
            self.queue.append(job)
        return job_id

    def update_job(self, job_id, key, value):
        with self.lock:
            for job in self.queue:
                if job["id"] == job_id:
                    job[key] = value
                    break

    def get_status_df(self):
        with self.lock:
            if not self.queue:
                return pd.DataFrame(columns=["ID", "登録時刻", "KW", "カテゴリ", "状態", "進捗", "ファイル"])
            
            data = []
            for job in self.queue:
                data.append([
                    job["id"], job["timestamp"], job["keyword"], job["category"], 
                    job["status"], job["progress"], job["result_file"]
                ])
            return pd.DataFrame(data, columns=["ID", "登録時刻", "KW", "カテゴリ", "状態", "進捗", "ファイル"])

    def _process_queue(self):
        while self.is_running:
            job_to_run = None
            with self.lock:
                for job in self.queue:
                    if job["status"] == "待機中":
                        job_to_run = job
                        job["status"] = "実行中"
                        break
            
            if job_to_run:
                self._execute_job(job_to_run)
            else:
                time.sleep(1)

    def _execute_job(self, job):
        try:
            p = job["params"]
            
            def progress_callback(current, total):
                self.update_job(job["id"], "progress", f"{current}/{total}")
            
            api_key = p.get("api_key") or DEFAULT_API_KEY
            scraper = MercariFastScraper(api_key, DEFAULT_BASE_URL, DEFAULT_MODEL)
            cat_id = CATEGORY_MAP.get(p.get("category_name"))
            
            count, file_path = scraper.run(
                keyword=p["keyword"],
                category_id=cat_id,
                category_name=p["category_name"],
                status=p["status"],
                price_min=p["price_min"],
                price_max=p["price_max"],
                sort_order=p["sort_order"],
                total_limit=int(p["limit"]),
                num_workers=int(p["workers"]),
                download_images=p["download_images"],
                use_ai_healing=p["use_ai_healing"],
                headless_mode=p["headless_mode"],
                progress_callback=progress_callback,
                sleep_time=p["sleep_time"]
            )
            
            with self.lock:
                job["status"] = "完了"
                job["result_file"] = os.path.basename(file_path) if file_path else "Error"
                
        except Exception as e:
            print(f"Job Error: {e}")
            with self.lock:
                job["status"] = "エラー"
                job["progress"] = str(e)

# グローバルキューインスタンス
job_manager = JobQueueManager()

# --- UIイベントハンドラ ---
def add_to_queue(api_key, keyword, category_name, limit, status, price_min, price_max, sort_order, workers, download_images, use_ai_healing, headless_mode,sleep_time=30):
    if not keyword and not category_name:
        return job_manager.get_status_df(), "エラー: キーワードかカテゴリを指定してください。"
        
    params = {
        "api_key": api_key, "keyword": keyword, "category_name": category_name,
        "limit": limit, "status": status, "price_min": price_min,
        "price_max": price_max, "sort_order": sort_order, "workers": workers,
        "download_images": download_images, "use_ai_healing": use_ai_healing,
        "headless_mode": headless_mode,
        "sleep_time": sleep_time
    }
    
    job_id = job_manager.add_job(params)
    return job_manager.get_status_df(), f"キューに追加しました (ID: {job_id})"

def refresh_table():
    return job_manager.get_status_df()

# --- UI構築 ---
with gr.Blocks(title="メルカリScraper Queue") as demo:
    gr.Markdown("## メルカリAIスクレイピング (予約実行・キュー機能付き)")
    gr.Markdown("条件を設定して「キューに追加」を押すと、バックグラウンドで順番に処理されます。")
    
    with gr.Accordion("API設定", open=False):
        api_key_input = gr.Textbox(label="API Key", type="password")
    
    with gr.Row():
        keyword_input = gr.Textbox(label="検索キーワード", placeholder="例: iPhone 12")
        limit_input = gr.Number(label="目標合計取得件数", value=50, precision=0)
    
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
        ai_healing_input = gr.Checkbox(label="AI修復を有効にする", value=True)
        headless_input = gr.Checkbox(label="ヘッドレスモード", value=True)
        sleep_time = gr.Slider(label="ページ読み込み後の待機時間 (秒)", value=100, minimum=0, step=1,value=30)

    add_btn = gr.Button("キューに追加 (予約)", variant="primary")
    message_box = gr.Markdown("")
    
    gr.Markdown("---")
    gr.Markdown("### 実行キュー / ステータス")
    
    # データフレーム表示（インタラクティブな更新のため）
    status_table = gr.Dataframe(
        headers=["ID", "登録時刻", "KW", "カテゴリ", "状態", "進捗", "ファイル"],
        datatype=["str", "str", "str", "str", "str", "str", "str"],
        interactive=False
    )
    
    # 自動更新タイマー (2秒ごとに更新)
    timer = gr.Timer(2)
    timer.tick(refresh_table, outputs=status_table)
    
    add_btn.click(
        add_to_queue,
        inputs=[
            api_key_input, keyword_input, category_input, limit_input, 
            status_input, price_min_input, price_max_input, sort_input, 
            workers_input, image_dl_input, ai_healing_input, headless_input,sleep_time
        ],
        outputs=[status_table, message_box]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)