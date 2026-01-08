import os
from pathlib import Path

import logging


from config import DEFAULT_CONFIG, RAGConfig
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.generation_integration import GenerationIntegrationModule
from rag_modules.index_construction import IndexConstructionModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径和API密钥
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("未设置DEEPSEEK_API_KEY环境变量")

    def initialize_system(self):
        """初始化所有模块"""
        # 1. 初始化数据准备模块
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
        )

        # 3. 初始化生成集成模块
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def build_knowledge_base(self):
        """构建知识库"""
        # 1. 尝试加载已保存的索引
        vector_store = self.index_module.load_index()

        if vector_store is not None:
            # 加载已有索引,但仍需文档和分块用于检索模块
            self.data_module.load_documents()
            chunks = self.data_module.chunk_documents()
        else:
            # 构建新索引的完整流程
            self.data_module.load_documents()
            chunks = self.data_module.chunk_documents()
            vector_store = self.index_module.build_vector_index(chunks)
            self.index_module.save_index()

        # 初始化索引优化模块
        self.retrieval_module = RetrievalOptimizationModule(
            vector_store=vector_store, chunks=chunks
        )

    def ask_question(self, question: str, stream: bool = False):
        """回答用户问题"""
        # 1. 查询路由
        route_type = self.generation_module.query_router(question)

        # 2. 智能查询重写(根据路由类型)
        if route_type == "list":
            rewritten_query = question  # 列表查询不重写
        else:
            rewritten_query = self.generation_module.query_rewrite(question)

        # 3. 检索相关子块
        relevant_chunks = self.retrieval_module.hybrid_search(
            rewritten_query, top_k=self.config.top_k
        )

        # 4. 根据路由类型选择回答方案
        if route_type == "list":
            # 列表查询: 返回推荐菜品名称列表
            relevant_docs = self.data_module.get_parent_document(relevant_chunks)
            return self.generation_module.generate_list_answer(
                rewritten_query, relevant_docs
            )
        else:
            # 详细查询: 获取完整文档并生成详细答案
            relevant_docs = self.data_module.get_parent_document(relevant_chunks)

            if route_type == "detail":
                # 详细查询使用分步指导模式
                return self.generation_module.generate_step_by_step_answer(
                    rewritten_query, relevant_docs
                )
            else:
                # 一般查询使用基础回答模式
                return self.generation_module.generate_basic_answer(
                    rewritten_query, relevant_docs
                )
                
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        
        # 初始化系统和构建知识库
        self.initialize_system()
        self.build_knowledge_base()
        
        while True:
            user_input = input("\n您的问题: ").strip()
            if user_input.lower() in ["退出", "exit", "quit"]:
                break
            
            # 询问是否使用流式输出
            # 询问是否使用流式输出
            stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
            use_stream = stream_choice != 'n'

            if use_stream:
                # 流式输出，实时显示生成过程
                for chunk in self.ask_question(user_input, stream=True):
                    print(chunk, end="", flush=True)
            else:
                # 普通输出
                answer = self.ask_question(user_input, stream=False)
                print(answer)


def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
