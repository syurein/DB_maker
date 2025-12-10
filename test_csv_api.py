import requests
from typing import List, Dict, Any

class MarketDataManager:
    def __init__(self, api_base_url: str = "https://shinssss-db.hf.space"):
        """
        Args:
            api_base_url (str): APIのベースURL
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.session = requests.Session()

    def fetch_market_data(self, search_queries: List[str]) -> List[Dict[str, Any]]:
        """
        API (/search-name) を使用して商品を検索し、リストを返す。
        
        Args:
            search_queries (list): 検索したいキーワードのリスト (元コードのregex_patternsに相当)
        """
        all_records = []
        # ドキュメントに基づき /search-name エンドポイントを使用
        endpoint = f"{self.api_base_url}/search-name"

        for query in search_queries:
            try:
                # API仕様: query (必須), limit (任意, default 10)
                params = {
                    "query": str(query),
                    "limit": 20  # 必要に応じて取得件数を調整
                }
                
                response = self.session.get(endpoint, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    # レスポンス形式: {"message": "...", "results": [{"name":..., "price":...}]}
                    results = data.get("results", [])
                    
                    for item in results:
                        record = {
                            # APIのレスポンスフィールドを元の形式にマッピング
                            "product_name": item.get("name"),
                            "price": item.get("price"),
                            # APIレスポンスに商品ページURLが含まれないため空文字を設定
                            "item_url": "", 
                            "image_url": item.get("image_url"),
                            "source": "DB_API"
                        }
                        all_records.append(record)
                else:
                    print(f"API Search Failed for '{query}': Status {response.status_code}")
                    print(response.text)

            except Exception as e:
                print(f"API Request Error for '{query}': {e}")
                continue

        return all_records

# --- 使用例 ---
if __name__ == "__main__":
    manager = MarketDataManager()
    
    # "regex_patterns" の代わりに検索キーワードのリストを渡す
    # APIは「部分一致」で検索します
    patterns = ["アクリル", "Camera"]
    
    data = manager.fetch_market_data(patterns)
    
    print(f"検索結果: {len(data)} 件")
    for d in data:
        print(d)