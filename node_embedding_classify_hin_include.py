import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from read_dialog_utils import get_dialog_relation, read_channel_ids
from transformers import BertTokenizer, BertModel
import pymysql
import json
import pandas as pd
import torch
import re
import random
from tqdm import tqdm
import itertools

host = "172.16.10.36"
port = 3306
user = "tg"
password = "#G5rsPX@"
db = "telegram"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 指定本地路径
local_path = "bert-base-chinese/"
vocab_file = 'bert-base-chinese/vocab.txt'
# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(vocab_file)
# tokenizer = BertTokenizer(vocab_file)
model = BertModel.from_pretrained(local_path)
model.to(device)

def contains_chinese(text):
    pattern = re.compile('[\u4e00-\u9fa5]')
    return bool(pattern.search(text))

def get_db():
    try:
        return pymysql.connect(host=host, port=port, user=user, password=password, db=db, use_unicode=True,
                               charset="utf8mb4")
    except Exception as e:
            print("数据库连接失败")
            print(e)

def get_channel_messages_tag(table_name, channel_id, messages_num):
    conn = get_db()
    tags_str = get_tags_from_channel_id(channel_id, conn)
    if not tags_str:
        return []

    tags = tags_str.split(',')
    results = []

    for tag in tags:
        table_name_tag = f"{table_name}{tag}"
        cursor = conn.cursor()
        query = f'SELECT * FROM {table_name_tag} WHERE channel_id=%s AND message IS NOT NULL ORDER BY date DESC LIMIT %s'
        cursor.execute(query, (channel_id, messages_num))
        results.extend(cursor.fetchall())
        cursor.close()

    conn.close()
    return results

def get_tags_from_channel_id(channel_id,conn):
    cursor = conn.cursor()
    cursor.execute(f'SELECT tags FROM channel_id_tag WHERE channel_id={channel_id}')
    tag = cursor.fetchone()
    cursor.close()

    return tag[0] if tag else None

# 读群组特征相关内容
def get_feature_sql(connection,table_name,channel_list):
    # SQL查询语句，选择 title、entity_info 和 messages 列

    # 使用参数化查询
    sql_query = 'SELECT title, entity_info, channel_id FROM {} WHERE channel_id IN %s'.format(table_name)

    # 使用pandas的read_sql函数执行查询并将结果保存为DataFrame
    df = pd.read_sql(sql_query, connection, params=(channel_list,))

    # 获取 messages 列
    df['messages'] = df['channel_id'].apply(lambda channel_id: get_channel_messages_tag('tg_messages_', channel_id, 200))


    # 从 messages 列中提取每一行中第二列的内容并组成一个长字符串
    df['messages'] = df['messages'].apply(
        lambda messages_list: ' '.join([str(tup[2]) for tup in messages_list]) if messages_list else None)
    # 限制字符串长度
    max_length = 200
    df['messages'] = df['messages'].apply(lambda x: str(x)[:max_length])


    # 从 JSON 字段 entity_info 中提取 description
    df['description'] = df['entity_info'].apply(
        lambda x: json.loads(x)['description'] if 'description' in json.loads(x) else None)

    # 在DataFrame中进行处理，将 title、description 和 messages 合并为一个新列 combined_text
    df['combined_text'] = df['title'] + ' ' + df['description'].astype(str) + ' ' + df['messages'].astype(str)

    return df

def get_text_embedding(text):
    max_length = 128  # 设定最大长度
    truncated_text = text[:max_length]  # 对文本进行前向截断
    inputs = tokenizer(truncated_text, return_tensors="pt", padding=True, truncation=True)
    inputs.to(device)
    # 获取文本嵌入
    with torch.no_grad():
        outputs = model(**inputs.to(device))
        text_embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu()  # 获取最后一层的平均特征向量
    return text_embedding

def messages_feature(messages):
    text_embeddings = []
    for message in messages:
        if message[2]:
            text_embeddings.append(get_text_embedding(message[2]))

    # user_representation = torch.mean(torch.stack(text_embeddings), dim=0)
    # average_embedding_list = user_representation.squeeze().tolist()

    embedding_dim = 768
    default_tensor = torch.zeros(embedding_dim)

    if len(text_embeddings) > 0:
        # 如果 text_embeddings 列表不为空，则执行 torch.stack() 函数
        user_representation = torch.mean(torch.stack(text_embeddings), dim=0)
    else:
        # 如果 text_embeddings 列表为空，则执行适当的处理操作，例如添加默认值或跳过此步骤
        user_representation = torch.zeros_like(default_tensor)  # 使用默认张量作为替代

    # print(len(user_representation[0][0]))
    # print(user_representation)

    # if not len(user_representation):
    #     print('text_embeddings:',text_embeddings)
    #     print(user_representation)

    return user_representation

def channel_feature(conn, channel_id):
    table_name = 'tg_channel'

    cursor = conn.cursor()
    query = f'SELECT title, entity_info, channel_id FROM {table_name} WHERE channel_id=%s'
    cursor.execute(query, (channel_id,))
    result = cursor.fetchall()[0]
    cursor.close()

    # print('result[1]:',result)
    description = json.loads(result[1])['description']
    # if description:
    #     print('description:',description)

    title = result[0]

    profile = title + description
    if not contains_chinese(profile):
        return None

    max_length = 128  # 设定最大长度
    profile = profile.replace(" ", "").replace("\n", "").replace("\r","")
    truncated_text = profile[:max_length]  # 对文本进行前向截断
    inputs = tokenizer(truncated_text, return_tensors="pt", padding=True, truncation=True)
    inputs.to(device)
    # 获取文本嵌入
    with torch.no_grad():
        outputs = model(**inputs.to(device))
        profile_embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu()  # 获取最后一层的平均特征向量

    messages = get_channel_messages_tag('tg_messages_', channel_id, 200)
    msg_feature = messages_feature(messages)

    profile_embeddings = [profile_embedding,msg_feature]
    group_representation = torch.mean(torch.stack(profile_embeddings), dim=0)

    all_channel_messages = get_channel_messages_tag('tg_messages_', channel_id, 10000)

    return group_representation, all_channel_messages


def get_user_message_dict(all_messages):
    # 设置随机数种子，这里的 42 是一个任意选择的数值，可以根据需要修改
    random.seed(42)

    # 创建一个空字典，用于存储每个用户的消息列表
    user_messages = {}

    # 遍历所有的消息
    for message in all_messages:
        user_id = message[5]  # 假设消息中索引为 5 的位置存储了用户 ID

        # 如果字典中已经有了该用户的消息列表，则将消息添加到列表中
        if user_id in user_messages:
            # 如果该用户的消息数量未达到 200 条，则直接添加消息到列表中
            if len(user_messages[user_id]) < 200:
                user_messages[user_id].append(message)
            # 否则，以一定概率替换已有消息
            elif random.random() < 200 / len(user_messages[user_id]):
                replace_index = random.randint(0, 199)
                user_messages[user_id][replace_index] = message
        # 否则，创建一个新的消息列表，并将消息添加到其中
        else:
            user_messages[user_id] = [message]

    return user_messages

if __name__ == "__main__":


    folder_path = 'user_reply_relations_10000_copy/friend_relationship'
    channel_ids = read_channel_ids(folder_path)
    dialog_relations_list = []
    all_users = set()

    channel_relation_ids = []

    # 群组节点特征
    conn = get_db()
    channel_features = {}
    all_messages = []

    channel_ids = channel_ids[:3]

    for channel_id in tqdm(channel_ids):
        # 调用 channel_feature 函数获取对应 channel_id 的 feature
        tmp = channel_feature(conn, channel_id)
        if tmp:
            feature, tmp_messages = tmp
            all_messages.extend(tmp_messages)
            # 将 feature 存储到字典中，键为 channel_id
            channel_features[channel_id] = feature
            channel_relation_ids.append(channel_id)
    conn.close()

    for channel_id in tqdm(channel_relation_ids):
        dialog_relations, all_users = get_dialog_relation(folder_path, channel_id)
        dialog_relations_list.extend(dialog_relations)

    user_channel_relations = []
    # 用户节点特征
    print("all_messages:",len(all_messages))
    user_features = {}
    user_messages = get_user_message_dict(all_messages)
    print("Number of users:", len(user_messages))
    for user_id, messages in tqdm(user_messages.items()):
        unique_message_0 = set()  # 用于跟踪已经出现过的 message[0]
        extracted_messages = []  # 用于存储提取的不重复的 message[0]
        # 遍历 messages，提取不重复的消息的 message[0]
        for message in messages:
            if message[0] not in unique_message_0:
                unique_message_0.add(message[0])
                extracted_messages.append(message[0])

        # 打印提取的消息
        for channel_id in extracted_messages:
            user_channel_relations.append((user_id,channel_id))

        # 对当前用户的消息列表应用 messages_feature 函数
        features = messages_feature(messages)
        user_features[user_id] = features


    # 构建异质图
    G = nx.Graph()

    # 添加节点类型和特征向量
    for user_id, feature in user_features.items():
        # print(feature)
        G.add_node(user_id, type='user', features=feature)

    for channel_id, feature in channel_features.items():
        G.add_node(channel_id, type='group', features=feature)

    # 初始化计数器
    user_count = 0
    group_count = 0

    # 遍历图中的节点，并根据节点类型进行计数
    for node, data in G.nodes(data=True):
        if data['type'] == 'user':
            user_count += 1
        elif data['type'] == 'group':
            group_count += 1

    # 输出节点数量
    print("用户节点数量:", user_count)
    print("群组节点数量:", group_count)


    # 添加边
    # for user1_id, user2_id in dialog_relations_list:
    #     G.add_edge(user1_id, user2_id, relation='dialog')
    for user1_id, user2_id in dialog_relations_list:
        user1_id = int(user1_id)
        user2_id = int(user2_id)
        if G.has_node(user1_id) and G.has_node(user2_id):  # 确保 user1_id 和 user2_id 都在图中
            if G.has_edge(user1_id, user2_id):
                # 如果边已经存在，则更新边的权重
                current_weight = G[user1_id][user2_id].get('weight', 0)
                G[user1_id][user2_id]['weight'] = current_weight + 1
            else:
                # 如果边不存在，则添加新的边，并设置权重为1
                G.add_edge(user1_id, user2_id, relation='dialog', weight=1)
        else:
            # 如果 user1_id 或 user2_id 不在图中，则跳过这个节点对
            print(f"Warning: Node {user1_id} or {user2_id} not found in graph. Skipping...")

    # 添加额外的边，例如用户与其所属的q之间的关系
    for user_id, channel_id in user_channel_relations:
        user_id = int(user_id)
        channel_id = int(channel_id)
        G.add_edge(user_id, channel_id, relation='belongs_to', weight=1)


    # Node2Vec 方法
    node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200, workers=4)
    model_node2vec = node2vec.fit(window=10, min_count=1, batch_words=4)
    # user_embeddings_node2vec = {
    #     node: np.concatenate((model_node2vec.wv.get_vector(node), G.nodes[node]['features'])) for node in G.nodes() if 'user' in node
    # }

    # print('model_node2vec.wv')
    # print(model_node2vec.wv)
    # print("键的数据类型:", type(list(model_node2vec.wv.key_to_index.keys())[0]))

    # user_embeddings_node2vec = {}
    # for node, data in G.nodes(data=True):
    #     # print('node:',type(node))
    #     # if type(node) == str:
    #     #     print(node)
    #     if data != {}:
    #         if 'user' in data['type']:
    #             if str(node) in model_node2vec.wv:
    #                 # print(type(node))
    #                 # print(node)
    #                 embedding = model_node2vec.wv.get_vector(str(node))
    #                 features = data['features']
    #                 # print('feature:',features.shape)
    #                 # 确保 features 是一个二维数组
    #                 feature_array = feature.numpy()
    #                 features = np.expand_dims(features, axis=0)
    #                 # print('embedding:',embedding.shape)
    #                 user_embeddings_node2vec[node] = np.concatenate((feature_array, embedding.reshape(1, -1)), axis=1)
    #                 # print(f"Success: Node {node} found in Node2Vec model.")
    #             else:
    #                 print(f"Warning: Node {node} not found in Node2Vec model. Skipping...")
    #         # embedding = model_node2vec.wv.get_vector(node)
    #         # features = data['features']
    #         # user_embeddings_node2vec[node] = np.concatenate((embedding, features))
    #
    #
    # # Metapath2Vec 方法
    # metapaths = [['group', 'user', 'group'], ['user', 'group', 'user']]
    # sentences = []
    # for metapath in metapaths:
    #     sentences.extend(node2vec.walks)
    # model_metapath2vec = Word2Vec(sentences, vector_size=64, window=10, min_count=1, sg=1, workers=4)
    # user_embeddings_metapath2vec = {
    #     node: np.concatenate((model_metapath2vec.wv.get_vector(str(node)), G.nodes[node]['features'])) for node in G.nodes() if 'user' in str(node)
    # }
    #
    # # HERec 方法
    # doc2vec_corpus = [TaggedDocument(words=metapath, tags=[str(i)]) for i, metapath in enumerate(sentences)]
    # model_herec = Doc2Vec(documents=doc2vec_corpus, vector_size=64, window=10, min_count=1, workers=4)
    # user_embeddings_herec = {
    #     node: np.concatenate((model_herec.dv[str(node)], G.nodes[node]['features'])) for node in G.nodes() if 'user' in str(node)
    #     # node: np.concatenate((model_herec.dv[str(i)], G.nodes[node]['features'])) for i, node in enumerate(G.nodes()) if 'user' in node
    # }


    # HIN2Vec 方法
    sentences_hin2vec = []
    for node in G.nodes():
        if 'user' in str(node):
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                weight = G.edges[node, neighbor]['weight']
                sentences_hin2vec.append([node, neighbor, weight])
    model_hin2vec = Word2Vec(sentences_hin2vec, vector_size=64, window=10, min_count=1, sg=1, workers=4)
    user_embeddings_hin2vec = {
        node: np.concatenate((model_hin2vec.wv.get_vector(node), G.nodes[node]['features'])) for node in G.nodes() if 'user' in node
    }


    # 进行测试

    # 获取所有用户节点
    user_nodes = [node for node, node_type in G.nodes(data='type') if node_type == 'user']

    # 生成用户组合
    user_combinations = list(itertools.combinations(user_nodes, 2))

    # 生成特征向量 X 和标签 y
    X_list = []
    y_list = []

    # 遍历用户组合，生成特征向量和标签
    for user1, user2 in user_combinations:
        X_node2vec = np.concatenate((user_embeddings_node2vec[user1], user_embeddings_node2vec[user2]))
        X_metapath2vec = np.concatenate((user_embeddings_metapath2vec[user1], user_embeddings_metapath2vec[user2]))
        X_herec = np.concatenate((user_embeddings_herec[user1], user_embeddings_herec[user2]))
        X_hin2vec = np.concatenate((user_embeddings_hin2vec[user1], user_embeddings_hin2vec[user2]))

        X = np.array([X_node2vec, X_metapath2vec, X_herec, X_hin2vec])
        X_list.append(X)

        # 随机生成标签
        y = np.random.choice([0, 1])
        y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练分类器
    classifiers = []
    for i in range(4):
        classifier = LogisticRegression()
        classifier.fit(X_train[:, i, :], y_train)
        classifiers.append(classifier)

    # 在测试集上进行预测并评估分类器性能
    accuracies = []
    for i in range(4):
        classifier = classifiers[i]
        accuracy = classifier.score(X_test[:, i, :], y_test)
        accuracies.append(accuracy)

    print("Accuracy for Node2Vec:", accuracies[0])
    print("Accuracy for Metapath2Vec:", accuracies[1])
    print("Accuracy for HERec:", accuracies[2])
    print("Accuracy for HIN2Vec:", accuracies[3])



    # 训练分类器
    X = np.array([np.concatenate((user_embeddings_node2vec['user1'], user_embeddings_node2vec['user2'])),
                  np.concatenate((user_embeddings_metapath2vec['user1'], user_embeddings_metapath2vec['user2'])),
                  np.concatenate((user_embeddings_herec['user1'], user_embeddings_herec['user2'])),
                  np.concatenate((user_embeddings_hin2vec['user1'], user_embeddings_hin2vec['user2']))])

    # X = np.array([np.concatenate((user_embeddings_node2vec['user1'], user_embeddings_node2vec['user2'])),
    #               np.concatenate((user_embeddings_node2vec['user2'], user_embeddings_node2vec['user3'])),
    #               np.concatenate((user_embeddings_node2vec['user1'], user_embeddings_node2vec['user3']))])
    y = np.array([1,1,1,0])  # 1表示存在好友关系，0表示不存在好友关系

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练逻辑回归分类器
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = classifier.predict(X_test)

    # 打印预测结果
    print("Predictions:", y_pred)

