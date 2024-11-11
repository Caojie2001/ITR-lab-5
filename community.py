import re
import jieba
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import HashingTF, MinHashLSH
from collections import defaultdict

spark = SparkSession.builder \
    .appName("GirvanNewmanCommunityDetection") \
    .getOrCreate()

from graphframes import GraphFrame

def load_data(file_path, stopwords_path):
    df = spark.read.csv(file_path, header=True)
    stopwords = set()
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        for line in file:
            stopwords.add(line.strip())
    broadcast_stopwords = spark.sparkContext.broadcast(stopwords)
    return df, broadcast_stopwords

def preprocess_text(df, broadcast_stopwords, threshold=3):
    df_filtered = df.filter(df['content'].contains('女'))
    user_counts = df_filtered.groupBy('user_id').count()
    valid_users = user_counts.filter(user_counts['count'] >= threshold).select('user_id')
    df_filtered = df_filtered.join(valid_users, on='user_id', how='inner')
    def tokenize(content):
        text = ''.join(re.findall(r'[\u4e00-\u9fa5]', content))
        words = jieba.lcut(text)
        tokens = [word for word in words if word not in broadcast_stopwords.value and len(word) > 1]
        return tokens
    tokenize_udf = F.udf(tokenize, ArrayType(StringType()))
    df_tokenized = df_filtered.withColumn('tokens', tokenize_udf('content'))
    df_tokenized = df_tokenized.filter(F.size('tokens') > 0)
    return df_tokenized.select('user_id', 'tokens')

def compute_jaccard_similarity(df, gamma_low=0.4):
    hashingTF = HashingTF(inputCol="tokens", outputCol="features", numFeatures=10000)
    df_features = hashingTF.transform(df)
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(df_features)
    similarity_df = model.approxSimilarityJoin(df_features, df_features, 1.0 - gamma_low, distCol="JaccardDistance")
    similarity_df = similarity_df.filter(similarity_df['datasetA.user_id'] != similarity_df['datasetB.user_id']) \
                                 .selectExpr("datasetA.user_id as src", "datasetB.user_id as dst", "1.0 - JaccardDistance as JaccardSimilarity")
    similarity_df = similarity_df.filter(similarity_df['JaccardSimilarity'] >= gamma_low)
    
    exact_matches = similarity_df.filter(similarity_df['JaccardSimilarity'] == 1.0)
    users_to_remove = exact_matches.select('dst').distinct()
    df = df.join(users_to_remove, on=df['user_id'] == users_to_remove['dst'], how='left_anti')
    similarity_df = similarity_df.join(users_to_remove, on=similarity_df['dst'] == users_to_remove['dst'], how='left_anti')
    return similarity_df, df

def build_graphframe(similarity_df, df):
    vertices = df.select('user_id').distinct().withColumnRenamed('user_id', 'id')
    edges = similarity_df.select('src', 'dst')
    g = GraphFrame(vertices, edges)
    return g

def compute_edge_betweenness(graph):
    edge_betweenness = defaultdict(float)
    vertices = graph.vertices.select("id").rdd.flatMap(lambda x: x).collect()
    for vertex in vertices:
        shortest_paths = graph.shortestPaths(landmarks=[vertex])
        paths = shortest_paths.select("id", "distances").collect()
        for row in paths:
            start_node = row['id']
            distances = row['distances']
            for target_node, _ in distances.items():
                if start_node != target_node:
                    path = [start_node, target_node]
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        edge_betweenness[edge] += 1.0  
    return edge_betweenness

def girvan_newman(graph, num_communities):
    current_graph = graph
    while True:
        edge_betweenness = compute_edge_betweenness(current_graph)
        max_betweenness_edge = max(edge_betweenness, key=edge_betweenness.get)
        current_graph = GraphFrame(
            current_graph.vertices,
            current_graph.edges.filter(
                ~((current_graph.edges.src == max_betweenness_edge[0]) &
                  (current_graph.edges.dst == max_betweenness_edge[1]))
            )
        )
        communities = current_graph.connectedComponents().select("id", "component").distinct()
        num_current_communities = communities.select("component").distinct().count()
        if num_current_communities >= num_communities:
            break
    return communities

def main_pipeline(data_file, stopwords_file, output_file, num_communities=10, gamma_low=0.4):
    df, broadcast_stopwords = load_data(data_file, stopwords_file)
    df_preprocessed = preprocess_text(df, broadcast_stopwords)
    similarity_df, df_nodes = compute_jaccard_similarity(df_preprocessed, gamma_low)
    graph = build_graphframe(similarity_df, df_nodes)
    if graph.edges.count() == 0:
        print("No edge.")
        return
    result_communities = girvan_newman(graph, num_communities)
    result_communities.write.csv(output_file, header=True, mode="overwrite")
    print(f"Results saved.")

data_file = '/home/caojie2001/Data Mining/week 5/2020-01.csv'
stopwords_file = '/home/caojie2001/Data Mining/week 5/baidu_stopwords.txt'
output_file = "/home/caojie2001/Data Mining/week 5/communities.csv"
main_pipeline(data_file, stopwords_file, output_file)
