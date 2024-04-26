import requests

def google_search(keyword, api_key, cx):
    url = f"https://www.googleapis.com/customsearch/v1?q={keyword}&key={api_key}&cx={cx}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'searchInformation' in data and 'totalResults' in data['searchInformation']:
                return int(data['searchInformation']['totalResults'])
            else:
                return -1  # 无法获取搜索结果数量
        else:
            # 处理不同的HTTP错误
            if response.status_code == 429:
                print("请求过多：已达到 API 速率限制。稍后再试。")
            elif response.status_code == 403:
                print("禁止访问：可能是 API 密钥问题或服务禁用。")
            elif response.status_code == 401:
                print("未授权：检查 API 密钥是否正确。")
            else:
                print(f"HTTP 错误 {response.status_code}: {response.text}")
            return -2
    except Exception as e:
        print(f"发生异常：{e}")
        return -1

# 使用 API 密钥和 CSE ID 进行搜索
api_key = "AIzaSyA1Y5Ou7k1mZ8QVtMf9N_PsqGhPtXmDnKc"
cx = "d36bac06df74046a9"
keyword = "KK红楼"
result_count = google_search(keyword, api_key, cx)

if result_count >= 0:
    print(f"关键词 '{keyword}' 的精确查询结果数量为: {result_count}")
else:
    print("无法获取搜索结果数量")