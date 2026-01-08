from pathlib import Path
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document



class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        index_save_path: str = "./vector_index",
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vector_store = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """初始化嵌入模型"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """构建向量索引"""
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        # 提取文本内容
        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        
        # 创建FAISS向量存储
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadata
        )
        
        return self.vector_store
    
    def save_index(self):
        """保存向量索引到配置的路径"""
        if not self.vector_store:
            raise ValueError("请先构建向量索引")
        
        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        
        self.vector_store.save_local(self.index_save_path)
    
    def load_index(self):
        """从配置的路径加载向量索引"""
        if not self.embeddings:
            self.setup_embeddings()
        
        if not Path(self.index_save_path).exists():
            return None
        
        self.vector_store = FAISS.load_local(
            self.index_save_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vector_store

