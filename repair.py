import csv

input_file = r'C:\Users\hikar\Downloads\DBscrayper\test1\merged_data_total_6542.csv'   # 実際のファイル名に合わせてください
output_file = 'output.csv'

with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    
    writer.writeheader()
    
    for row in reader:
        raw_price = row['price']
        
        # Noneや空文字の場合の対策
        if not raw_price:
            continue

        # カンマとダブルクォートを除去
        clean_price = raw_price.replace(',', '').replace('"', '')

        try:
            # 数値変換を試みる
            row['price'] = int(clean_price)
            writer.writerow(row)
        except ValueError:
            # エラーが出た場合（'price'などの文字だった場合）はスキップしてログを出す
            print(f"スキップしました（数値ではありません）: {raw_price}")

print("変換が完了しました。")