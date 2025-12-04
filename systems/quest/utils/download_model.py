from modelscope import snapshot_download

def download_model(model_id, cache_dir=None):
    """
    使用 ModelScope SDK 下载指定模型
    
    参数:
        model_id (str): 模型ID，格式为"namespace/model-name"
        cache_dir (str, optional): 模型缓存目录，默认使用ModelScope的默认缓存目录
    """
    try:
        print(f"开始下载模型: {model_id}")
        model_dir = snapshot_download(model_id, cache_dir=cache_dir)
        print(f"模型下载完成，保存路径: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"下载模型时出错: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 指定要下载的模型ID，例如达摩院的"damo/nlp_structbert_sentence-similarity_chinese-base"
    model_id = "BAAI/bge-m3"
    
    # 可选: 指定自定义缓存目录
    cache_dir = "/data/QUEST/jzshe/project/quest/model/temp_download"
    
    # 下载模型
    # model_path = download_model(model_id)
    # 如果指定自定义缓存目录:
    model_path = download_model(model_id, cache_dir=cache_dir)
    
    if model_path:
        print("模型下载成功！")
    else:
        print("模型下载失败。")