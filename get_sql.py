from flask import Flask, jsonify
from flask_cors import CORS
import pymysql.cursors

app = Flask(__name__)
CORS(app)  # 启用 CORS 中间件

# host = "172.16.10.36"
# port = 3306
# user = "tg"
# password = "#G5rsPX@"
# db = "telegram"

# MySQL数据库连接配置
connection = pymysql.connect(
    host="127.0.0.1",
    user="root",  # 替换为你的MySQL用户名
    password="lyf-990803",  # 替换为你的MySQL密码
    database='telegram',  # 替换为你的数据库名
    port=3306,
    cursorclass=pymysql.cursors.DictCursor

    # host="172.16.10.36",
    # port=3306,
    # user="tg",
    # password="#G5rsPX@",
    # database="telegram",
    # cursorclass=pymysql.cursors.DictCursor
)

@app.route('/fetch_groups')
def fetch_groups():
    try:
        with connection.cursor() as cursor:
            # 查询群组数据
            sql = "SELECT * FROM tg_channel LIMIT 100"
            cursor.execute(sql)
            groups = cursor.fetchall()
            # print(1)
            print(groups[0])
            return jsonify(groups)
    except Exception as e:
        print({'error': str(e)})
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
