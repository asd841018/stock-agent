import os
from dotenv import load_dotenv
from serpapi import SerpApiClient

load_dotenv()

def search(query: str) -> str:
    try:
        serpapi_api_key = os.getenv("SERPAPI_API_KEY")
        if not serpapi_api_key:
            raise ValueError("SERPAPI_API_KEY environment variable is not set.")
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_api_key,
            "gl": "tw",  # 國家代碼
            "hl": "zh-tw", # 語言代碼
        }
        client = SerpApiClient(params_dict=params)
        results = client.get_dict()
        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"