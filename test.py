import requests
import os

# APIのベースURL（環境に合わせて変更してください）
# ローカルで動かしている場合は http://localhost:8000 など
API_BASE_URL = "https://46e192aa6b9f.ngrok-free.app"
ENDPOINT = "/upload-zip"

def upload_zip_for_seeding(zip_file_path):
    """
    ZIPファイルをアップロードしてデータベースとストレージに登録する関数
    """
    url = f"{API_BASE_URL}{ENDPOINT}"
    
    # ファイルが存在するか確認
    if not os.path.exists(zip_file_path):
        print(f"エラー: ファイルが見つかりません -> {zip_file_path}")
        return

    try:
        # バイナリモード('rb')でファイルを開く
        with open(zip_file_path, "rb") as f:
            # multipart/form-data形式で送信するための準備
            # キー名 'file' はAPI定義の parameters -> name: file と一致させる必要があります
            files = {
                "file": (os.path.basename(zip_file_path), f, "application/zip")
            }
            
            print(f"アップロード中...: {zip_file_path}")
            response = requests.post(url, files=files)

        # レスポンスの確認
        if response.status_code == 200:
            print("✅ アップロード成功！")
            print("レスポンス:", response.json())
        elif response.status_code == 422:
            print("❌ バリデーションエラー（ファイル形式や中身を確認してください）")
            print("詳細:", response.json())
        else:
            print(f"❌ エラーが発生しました (Status: {response.status_code})")
            print("内容:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"通信エラー: {e}")

if __name__ == "__main__":
    # アップロードしたいZIPファイルのパスを指定
    target_zip = r"C:\Users\hikar\Downloads\DBscrayper\MercariScraper\all_meraged.zip" 
    
    upload_zip_for_seeding(target_zip)