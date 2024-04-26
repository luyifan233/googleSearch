import concurrent.futures
import requests
import time

cx = "d36bac06df74046a9"
api_key = "AIzaSyA1Y5Ou7k1mZ8QVtMf9N_PsqGhPtXmDnKc"

def google_search(keyword):
    url = f"https://www.googleapis.com/customsearch/v1?q={keyword}&key={api_key}&cx={cx}"
    try:
        time.sleep(1)
        response = requests.get(url)
        data = response.json()
        if 'searchInformation' in data and 'totalResults' in data['searchInformation']:
            return int(data['searchInformation']['totalResults'])
        else:
            return -1  # 无法获取搜索结果数量
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1


def save_to_txt(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(str(item[0])+','+str(item[1]) + '\n')

def read_from_txt(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split(',')
            data.append((int(items[0]), items[1]))
    return data


def search_keyword(keyword):
    print(f"Searching for keyword '{keyword[1]}'")
    result_count = google_search(keyword[1])
    print(f"关键词 '{keyword[1]}' 的精确查询结果数量为: {result_count}")
    return keyword[0], result_count

def search(keyword):
    print(keyword)
    result_count = google_search(keyword[1])
    # result_counts.append((keyword[0], result_count))
    print("关键词 '{}' 的精确查询结果数量为: {}".format(keyword[1], result_count))
    return (keyword[0], result_count)


if __name__ == "__main__":

    # 读取保存的文本文件
    titles = read_from_txt('titles.txt')
    usernames = read_from_txt('usernames.txt')

    # 打印读取的数据，检查是否正确
    print("Titles from file:", titles)
    print("Usernames from file:", usernames)



    keywords = titles
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # 处理 titles
        keywords = titles
        # 使用 executor.map 函数来并行执行搜索任务
        title_results = list(executor.map(search, keywords))


    with open('title_result_counts.txt', 'a') as file:
        # 对每个结果数量进行遍历并将其写入文件中
        for count in title_results:
            file.write(f'{str(count[0])},{str(count[1])}\n')

    keywords = usernames
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        username_results = list(executor.map(search, keywords))


    with open('username_result_counts.txt', 'a') as file:
        # 对每个结果数量进行遍历并将其写入文件中
        for count in username_results:
            file.write(f'{str(count[0])},{str(count[1])}\n')

