
from playwright.sync_api import sync_playwright
import re
def janpara_price(product_name):
    with sync_playwright() as p:
        browser =p.chromium.launch(headless=True)
        page=browser.new_page()
        page.goto(f'https://buy.janpara.co.jp/buy/search?keyword={product_name}', wait_until='networkidle')
        page.screenshot(path='janpara.png')
        if page.locator('text=該当商品は見つかりませんでした').is_visible():
            print('該当商品は見つかりませんでした')
        else:
            price=page.locator('text=円').all_inner_texts()
            price=[int(re.sub(r'\D', '', p)) for p in price if re.sub(r'\D', '', p) != '']
            print(price)
        

if __name__=='__main__':
    janpara_price('WF1000xm')