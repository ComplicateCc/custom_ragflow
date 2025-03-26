#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 测试脚本
用于循环测试RAG接口并收集检索结果
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# 导入RAGFlow SDK
from ragflow_sdk import RAGFlow

# 配置参数
API_KEY = "ragflow-E3ZDc1ZWUwMGEwZDExZjBhODkzMDI0Mm"  # 替换为您的API密钥
HOST_ADDRESS = "http://localhost:9380"  # 替换为您的RAGFlow服务地址
TEST_DATASET_NAME = "客服中台-bge_embedding"  # 替换为您要测试的现有数据集名称

# 测试问题列表
TEST_QUESTIONS = [
    "能告诉我重置密码在哪里吗?",
    "我账号被盗了怎么版?",
    "小号无法上下物品怎么办?",
    "我账号登录不上啦",
    "游戏蓝屏了怎么办?",
    "怎么解决游戏闪退问题?",
]

# 创建结果目录
RESULTS_DIR = "rag_test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_dataset_documents(dataset):
    """获取数据集中的所有文档"""
    print(f"正在获取数据集 '{dataset.name}' 中的文档...")
    
    # 获取第一页文档
    page = 1
    page_size = 100
    all_documents = []
    
    while True:
        documents_page = dataset.list_documents(page=page, page_size=page_size)
        if not documents_page:
            break
            
        all_documents.extend(documents_page)
        print(f"已获取 {len(all_documents)} 个文档")
        
        if len(documents_page) < page_size:
            break
            
        page += 1
    
    print(f"数据集中共有 {len(all_documents)} 个文档")
    return all_documents

def test_rag_queries(rag_instance, dataset, questions, documents=None):
    """测试RAG查询并收集结果"""
    results = []
    dataset_ids = [dataset.id]
    document_ids = [doc.id for doc in documents] if documents else None
    
    print(f"开始测试 {len(questions)} 个问题...")
    for i, question in enumerate(questions):
        print(f"测试问题 {i+1}/{len(questions)}: {question}")
        
        # 调用RAG接口获取检索结果
        start_time = time.time()
        chunks = rag_instance.retrieve(
            dataset_ids=dataset_ids,
            document_ids=document_ids,
            question=question,
            page=1,
            page_size=10,
            similarity_threshold=0.2,
            vector_similarity_weight=0.7,
            top_k=1024
        )
        end_time = time.time()
        
        # 收集结果
        query_result = {
            "question": question,
            "retrieval_time": end_time - start_time,
            "retrieval_count": len(chunks),
            "chunks": []
        }
        
        # 收集检索到的chunks详情
        for chunk in chunks:
            chunk_data = {
                "content": chunk.content,
                "document_id": chunk.document_id,
                "document_name": chunk.document_keyword,
                "similarity": chunk.similarity,
                "vector_similarity": chunk.vector_similarity,
                "term_similarity": chunk.term_similarity,
            }
            query_result["chunks"].append(chunk_data)
        
        results.append(query_result)
    
    return results

def save_results(results, output_dir):
    """保存测试结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"rag_test_results_{timestamp}.json")
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"测试结果已保存到: {result_file}")
    
    # 生成简单的统计报告
    report_file = os.path.join(output_dir, f"rag_test_report_{timestamp}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("RAG测试报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 总体统计
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"测试问题数: {len(results)}\n\n")
        
        # 每个问题的统计
        for i, result in enumerate(results):
            f.write(f"问题 {i+1}: {result['question']}\n")
            f.write(f"检索时间: {result['retrieval_time']:.4f} 秒\n")
            f.write(f"检索结果数: {result['retrieval_count']}\n")
            
            # 最高相似度的前3个结果
            if result['chunks']:
                f.write("前3个最相关的结果:\n")
                sorted_chunks = sorted(result['chunks'], key=lambda x: x['similarity'], reverse=True)[:3]
                for j, chunk in enumerate(sorted_chunks):
                    f.write(f"  {j+1}. 相似度: {chunk['similarity']:.4f}, 文档: {chunk['document_name']}\n")
                    f.write(f"     内容: {chunk['content'][:100]}...\n")
            
            f.write("\n" + "-" * 40 + "\n\n")
    
    print(f"测试报告已生成: {report_file}")
    return result_file, report_file

def analyze_results(result_file):
    """分析测试结果"""
    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 提取关键指标
    metrics = {
        "avg_retrieval_time": sum(r["retrieval_time"] for r in results) / len(results),
        "avg_chunk_count": sum(r["retrieval_count"] for r in results) / len(results),
        "questions_with_no_results": sum(1 for r in results if r["retrieval_count"] == 0),
        "similarity_scores": []
    }
    
    # 收集所有相似度分数
    for result in results:
        for chunk in result["chunks"]:
            metrics["similarity_scores"].append(chunk["similarity"])
    
    if metrics["similarity_scores"]:
        metrics["avg_similarity"] = sum(metrics["similarity_scores"]) / len(metrics["similarity_scores"])
        metrics["max_similarity"] = max(metrics["similarity_scores"])
        metrics["min_similarity"] = min(metrics["similarity_scores"])
    else:
        metrics["avg_similarity"] = 0
        metrics["max_similarity"] = 0
        metrics["min_similarity"] = 0
    
    # 打印分析结果
    print("\n============ RAG测试分析 ============")
    print(f"平均检索时间: {metrics['avg_retrieval_time']:.4f} 秒")
    print(f"平均检索结果数: {metrics['avg_chunk_count']:.2f}")
    print(f"没有检索到结果的问题数: {metrics['questions_with_no_results']}")
    if metrics["similarity_scores"]:
        print(f"平均相似度分数: {metrics['avg_similarity']:.4f}")
        print(f"最高相似度分数: {metrics['max_similarity']:.4f}")
        print(f"最低相似度分数: {metrics['min_similarity']:.4f}")
    else:
        print("没有检索到结果，无法计算相似度统计信息")
    print("====================================\n")
    
    return metrics

def main():
    try:
        print("启动RAG测试...")
        
        # 初始化RAGFlow客户端
        rag = RAGFlow(API_KEY, HOST_ADDRESS)
        print(f"成功连接到RAGFlow服务: {HOST_ADDRESS}")
        
        # 获取现有数据集
        dataset = rag.get_dataset(name=TEST_DATASET_NAME)
        if not dataset:
            raise Exception(f"未找到数据集: {TEST_DATASET_NAME}")
        print(f"成功获取数据集: {dataset.name} (ID: {dataset.id})")
        
        # 获取数据集中的文档
        documents = get_dataset_documents(dataset)
        if not documents:
            print("警告: 数据集中没有文档，将在所有数据集范围内测试")
        else:
            print(f"将使用数据集中的 {len(documents)} 个文档进行测试")
        
        # 运行测试查询
        results = test_rag_queries(rag, dataset, TEST_QUESTIONS, documents)
        
        # 保存测试结果
        result_file, report_file = save_results(results, RESULTS_DIR)
        
        # 分析结果
        analyze_results(result_file)
        
        print(f"测试完成! 结果保存在: {RESULTS_DIR}")
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())