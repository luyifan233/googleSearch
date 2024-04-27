import concurrent.futures
import requests
import time

cx = "d36bac06df74046a9"
api_key = "AIzaSyA1Y5Ou7k1mZ8QVtMf9N_PsqGhPtXmDnKc"

def google_search(keyword):
    url = f"https://www.googleapis.com/customsearch/v1?q={keyword}&key={api_key}&cx={cx}"
    try:
        time.sleep(1.1)
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'searchInformation' in data and 'totalResults' in data['searchInformation']:
                return int(data['searchInformation']['totalResults'])
            else:
                return -1
        else:
            handle_errors(response)
            return -1
    except Exception as e:
        print(f"发生异常：{e}")
        return -1

def handle_errors(response):
    if response.status_code == 429:
        print("请求过多：已达到 API 速率限制。稍后再试。")
    elif response.status_code == 403:
        print("禁止访问：可能是 API 密钥问题或服务禁用。")
    elif response.status_code == 401:
        print("未授权：检查 API 密钥是否正确。")
    elif response.status_code == 400:
        print("错误的请求：检查请求参数。")
    else:
        print(f"HTTP 错误 {response.status_code}: {response.text}")

def search_and_save(keyword, output_file):
    result_count = google_search(keyword[1])
    print(f"关键词 '{keyword[1]}' 的精确查询结果数量为: {result_count}")
    with open(output_file, 'a') as file:
        file.write(f'{str(keyword[0])},{str(result_count)}\n')

def read_from_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [(int(line.strip().split(',')[0]), line.strip().split(',')[1]) for line in f]

if __name__ == "__main__":
    titles = read_from_txt('titles.txt')
    usernames = read_from_txt('usernames.txt')

    for keywords, output_file in [(titles, 'title_result_counts.txt'), (usernames, 'username_result_counts.txt')]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.map(lambda keyword: search_and_save(keyword, output_file), keywords)
