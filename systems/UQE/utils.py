import os
import csv

from PIL import Image
import config_uqe
import base64

# 定义一个函数，从文件路径读取 CSV 内容并拼接为字符串
def read_and_concatenate(data_schema, filepath):
    try:
        # 检查文件是否存在
        filepath = os.path.join(data_schema.parent_dir, data_schema.dataset_name, filepath)
        if os.path.exists(filepath):
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as file:
                        # 读取 CSV 文件
                        reader = csv.reader(file)
                        # 将每行拼接成字符串
                        result = "\n".join([",".join(row) for row in reader])
                    return result
                except UnicodeDecodeError:
                    continue
            # 如果所有编码都失败，使用latin-1作为最后的尝试
            with open(filepath, 'r', encoding='latin-1') as file:
                reader = csv.reader(file)
                result = "\n".join([",".join(row) for row in reader])
            return result
        else:
            return f"File not found: {filepath}"
    except Exception as e:
        return f"Error reading file {filepath}: {str(e)}"
    
def read_image(data_schema, filepath):
    try:
        # 检查文件是否存在
        # filepath = os.path.join(data_schema.parent_dir, data_schema.dataset_name, filepath)
        if os.path.exists(filepath):
            # 读取图片文件
            # image = Image.open(filepath)
            # return image

            with open(filepath, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            return f"File not found: {filepath}"
    except Exception as e:
        return f"Error reading file {filepath}: {str(e)}"

def read_image_PIL(data_schema, filepath):
    try:
        # 检查文件是否存在
        # filepath = os.path.join(data_schema.parent_dir, data_schema.dataset_name, filepath)
        if os.path.exists(filepath):
            # 读取图片文件
            image = Image.open(filepath)
            return image
        else:
            return f"File not found: {filepath}"
    except Exception as e:
        return f"Error reading file {filepath}: {str(e)}"

def read_query_list(query_dir, query_type):
    query_dict = {}
    query_dir = os.path.join(query_dir, query_type)
    for file in os.listdir(query_dir):
        if file.endswith('.sql'):
            file_path = os.path.join(query_dir, file)
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            content = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                # 如果所有编码都失败，使用latin-1
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()

            # 将content中的COUNT SUM MAX MIN AVG 替换为小写
            content = content.replace("COUNT", "count").replace("SUM", "sum").replace("MAX", "max").replace("MIN", "min").replace("AVG", "avg")
            query_dict[file.split('/')[-1].split('.')[0]] = content
    # 按数字排序
    query_dict = dict(sorted(query_dict.items(), key=lambda x: int(x[0][1:])))
    return query_dict

def print_config_to_file(file_path):
    with open(file_path, 'w') as f:
        f.write(f"BATCH_SIZE: {config_uqe.BATCH_SIZE}\n")
        f.write(f"BUDGET: {config_uqe.BUDGET}\n")
        f.write(f"USE_BART: {config_uqe.USE_BART}\n")
        f.write(f"AGGR_STRATEGY: {config_uqe.AGGR_STRATEGY}\n")
        f.write(f"N_CENTROIDS: {config_uqe.N_CENTROIDS}\n")
        f.write(f"N_ITER: {config_uqe.N_ITER}\n")
        f.write(f"GROUP_EXTRACT_SAMPLE_RATIO: {config_uqe.GROUP_EXTRACT_SAMPLE_RATIO}\n")
        f.write(f"AGGR_CLUSTER_SAMPLE_RATIO: {config_uqe.AGGR_CLUSTER_SAMPLE_RATIO}\n")