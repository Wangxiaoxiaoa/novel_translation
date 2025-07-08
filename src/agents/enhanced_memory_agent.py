"""
增强版记忆智能体 - 高级记忆管理和知识图谱
Enhanced Memory Agent - Advanced memory management and knowledge graph
"""

import asyncio
import json
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
from collections import defaultdict, deque
import networkx as nx
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

from .base_agent import BaseAgent
from ..models.base import (
    AgentMessage, AgentType, Chapter, Novel, Character, Location, 
    Item, Terminology, ProcessingStatus
)


class MemoryType(Enum):
    """记忆类型"""
    CHARACTER = "character"
    LOCATION = "location" 
    ITEM = "item"
    TERMINOLOGY = "terminology"
    PLOT_EVENT = "plot_event"
    RELATIONSHIP = "relationship"
    DIALOGUE = "dialogue"
    SCENE = "scene"
    EMOTION = "emotion"
    THEME = "theme"


@dataclass
class MemoryNode:
    """记忆节点"""
    id: str
    type: MemoryType
    name: str
    description: str
    attributes: Dict[str, Any]
    embeddings: Optional[List[float]]
    created_at: datetime
    updated_at: datetime
    access_count: int
    importance_score: float
    context_chapters: List[str]
    related_nodes: List[str]


@dataclass
class MemoryQuery:
    """记忆查询"""
    query_text: str
    query_type: MemoryType
    context_filter: Optional[Dict[str, Any]]
    similarity_threshold: float
    max_results: int


class EnhancedMemoryAgent(BaseAgent):
    """增强版记忆智能体"""
    
    def __init__(self, config, openai_client):
        super().__init__(config, openai_client)
        
        # 核心存储组件
        self.knowledge_graph = nx.MultiDiGraph()  # 知识图谱
        self.vector_store = VectorMemoryStore()   # 向量存储
        self.temporal_memory = TemporalMemoryManager()  # 时间记忆
        self.semantic_memory = SemanticMemoryManager()  # 语义记忆
        self.episodic_memory = EpisodicMemoryManager()  # 情节记忆
        
        # 高级功能模块
        self.relationship_analyzer = RelationshipAnalyzer()
        self.context_manager = ContextualMemoryManager()
        self.importance_calculator = ImportanceCalculator()
        self.memory_consolidator = MemoryConsolidator()
        self.retrieval_optimizer = RetrievalOptimizer()
        
        # 记忆索引
        self.memory_indices = {
            "by_type": defaultdict(list),
            "by_chapter": defaultdict(list), 
            "by_importance": defaultdict(list),
            "by_recency": deque(maxlen=1000)
        }
        
        # 记忆统计
        self.memory_stats = {
            "total_nodes": 0,
            "total_relationships": 0,
            "access_patterns": defaultdict(int),
            "query_patterns": defaultdict(int)
        }
        
        # 缓存系统
        self.retrieval_cache = {}
        self.embedding_cache = {}
        
    async def initialize(self) -> bool:
        """初始化增强版记忆智能体"""
        try:
            logger.info("初始化增强版记忆智能体...")
            
            # 初始化存储组件
            await self.vector_store.initialize()
            await self.temporal_memory.initialize()
            await self.semantic_memory.initialize()
            await self.episodic_memory.initialize()
            
            # 初始化功能模块
            await self.relationship_analyzer.initialize()
            await self.context_manager.initialize()
            await self.importance_calculator.initialize()
            await self.memory_consolidator.initialize()
            await self.retrieval_optimizer.initialize()
            
            # 加载持久化记忆
            await self.load_persistent_memory()
            
            # 健康检查
            health_ok = await self.health_check()
            if not health_ok:
                logger.error("增强版记忆智能体健康检查失败")
                return False
                
            self.status = ProcessingStatus.COMPLETED
            logger.info("增强版记忆智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"增强版记忆智能体初始化失败: {e}")
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理消息"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type == "store_memory":
                result = await self.store_comprehensive_memory(content)
            elif message_type == "retrieve_memory":
                result = await self.retrieve_contextual_memory(content)
            elif message_type == "update_memory":
                result = await self.update_memory_node(content)
            elif message_type == "delete_memory":
                result = await self.delete_memory_node(content)
            elif message_type == "build_knowledge_graph":
                result = await self.build_comprehensive_knowledge_graph(content)
            elif message_type == "analyze_relationships":
                result = await self.analyze_memory_relationships(content)
            elif message_type == "consolidate_memories":
                result = await self.consolidate_memories(content)
            elif message_type == "query_semantic_memory":
                result = await self.query_semantic_memory(content)
            elif message_type == "get_memory_stats":
                result = await self.get_memory_statistics(content)
            elif message_type == "optimize_memory":
                result = await self.optimize_memory_system(content)
            else:
                return await super().process_message(message)
            
            return AgentMessage(
                sender=self.agent_type,
                receiver=message.sender,
                message_type=f"{message_type}_result",
                content=result
            )
                
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            return AgentMessage(
                sender=self.agent_type,
                receiver=message.sender,
                message_type="error",
                content={"error": str(e)}
            )
    
    async def store_comprehensive_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """存储综合记忆"""
        try:
            self.current_task = "存储综合记忆"
            
            memory_data = data.get("memory_data", {})
            context = data.get("context", {})
            
            logger.info("开始存储综合记忆...")
            
            stored_nodes = []
            
            # 并行存储不同类型的记忆
            storage_tasks = []
            
            # 角色记忆
            if "characters" in memory_data:
                for char_data in memory_data["characters"]:
                    storage_tasks.append(self.store_character_memory(char_data, context))
            
            # 地点记忆
            if "locations" in memory_data:
                for loc_data in memory_data["locations"]:
                    storage_tasks.append(self.store_location_memory(loc_data, context))
            
            # 物品记忆
            if "items" in memory_data:
                for item_data in memory_data["items"]:
                    storage_tasks.append(self.store_item_memory(item_data, context))
            
            # 术语记忆
            if "terminologies" in memory_data:
                for term_data in memory_data["terminologies"]:
                    storage_tasks.append(self.store_terminology_memory(term_data, context))
            
            # 情节事件记忆
            if "plot_events" in memory_data:
                for event_data in memory_data["plot_events"]:
                    storage_tasks.append(self.store_plot_event_memory(event_data, context))
            
            # 关系记忆
            if "relationships" in memory_data:
                for rel_data in memory_data["relationships"]:
                    storage_tasks.append(self.store_relationship_memory(rel_data, context))
            
            # 执行存储任务
            storage_results = await asyncio.gather(*storage_tasks, return_exceptions=True)
            
            # 处理存储结果
            for result in storage_results:
                if isinstance(result, Exception):
                    logger.error(f"存储记忆失败: {result}")
                else:
                    stored_nodes.extend(result if isinstance(result, list) else [result])
            
            # 更新知识图谱
            await self.update_knowledge_graph(stored_nodes)
            
            # 建立关系连接
            await self.establish_memory_relationships(stored_nodes, context)
            
            # 计算重要性分数
            await self.calculate_importance_scores(stored_nodes)
            
            # 更新索引
            await self.update_memory_indices(stored_nodes)
            
            # 记忆整合
            await self.memory_consolidator.consolidate_new_memories(stored_nodes)
            
            result = {
                "stored_nodes": len(stored_nodes),
                "node_types": self.get_node_type_distribution(stored_nodes),
                "relationships_created": await self.count_new_relationships(),
                "knowledge_graph_stats": await self.get_knowledge_graph_stats(),
                "storage_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "context_chapter": context.get("chapter_id"),
                    "storage_duration": 0,  # 实际实现中计算
                    "memory_agent": self.agent_type
                }
            }
            
            logger.info(f"综合记忆存储完成，存储节点: {len(stored_nodes)}")
            return result
            
        except Exception as e:
            logger.error(f"存储综合记忆失败: {e}")
            raise
    
    async def store_character_memory(self, char_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """存储角色记忆"""
        try:
            character = Character(**char_data)
            
            # 创建记忆节点
            node_id = self.generate_node_id(MemoryType.CHARACTER, character.name)
            
            # 生成嵌入向量
            embeddings = await self.generate_embeddings(f"{character.name} {character.description}")
            
            # 创建记忆节点
            memory_node = MemoryNode(
                id=node_id,
                type=MemoryType.CHARACTER,
                name=character.name,
                description=character.description,
                attributes=asdict(character),
                embeddings=embeddings,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                access_count=0,
                importance_score=0.0,
                context_chapters=[context.get("chapter_id", "")],
                related_nodes=[]
            )
            
            # 存储到各个记忆系统
            await self.vector_store.store_node(memory_node)
            await self.semantic_memory.store_character(memory_node)
            await self.episodic_memory.add_character_appearance(memory_node, context)
            
            # 添加到知识图谱
            self.knowledge_graph.add_node(node_id, **asdict(memory_node))
            
            return [node_id]
            
        except Exception as e:
            logger.error(f"存储角色记忆失败: {e}")
            return []
    
    async def store_location_memory(self, loc_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """存储地点记忆"""
        try:
            location = Location(**loc_data)
            
            node_id = self.generate_node_id(MemoryType.LOCATION, location.name)
            embeddings = await self.generate_embeddings(f"{location.name} {location.description}")
            
            memory_node = MemoryNode(
                id=node_id,
                type=MemoryType.LOCATION,
                name=location.name,
                description=location.description,
                attributes=asdict(location),
                embeddings=embeddings,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                access_count=0,
                importance_score=0.0,
                context_chapters=[context.get("chapter_id", "")],
                related_nodes=[]
            )
            
            await self.vector_store.store_node(memory_node)
            await self.semantic_memory.store_location(memory_node)
            
            self.knowledge_graph.add_node(node_id, **asdict(memory_node))
            
            return [node_id]
            
        except Exception as e:
            logger.error(f"存储地点记忆失败: {e}")
            return []
    
    async def store_item_memory(self, item_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """存储物品记忆"""
        try:
            item = Item(**item_data)
            
            node_id = self.generate_node_id(MemoryType.ITEM, item.name)
            embeddings = await self.generate_embeddings(f"{item.name} {item.description}")
            
            memory_node = MemoryNode(
                id=node_id,
                type=MemoryType.ITEM,
                name=item.name,
                description=item.description,
                attributes=asdict(item),
                embeddings=embeddings,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                access_count=0,
                importance_score=0.0,
                context_chapters=[context.get("chapter_id", "")],
                related_nodes=[]
            )
            
            await self.vector_store.store_node(memory_node)
            await self.semantic_memory.store_item(memory_node)
            
            self.knowledge_graph.add_node(node_id, **asdict(memory_node))
            
            return [node_id]
            
        except Exception as e:
            logger.error(f"存储物品记忆失败: {e}")
            return []
    
    async def store_terminology_memory(self, term_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """存储术语记忆"""
        try:
            terminology = Terminology(**term_data)
            
            node_id = self.generate_node_id(MemoryType.TERMINOLOGY, terminology.term)
            embeddings = await self.generate_embeddings(f"{terminology.term} {terminology.definition}")
            
            memory_node = MemoryNode(
                id=node_id,
                type=MemoryType.TERMINOLOGY,
                name=terminology.term,
                description=terminology.definition,
                attributes=asdict(terminology),
                embeddings=embeddings,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                access_count=0,
                importance_score=0.0,
                context_chapters=[context.get("chapter_id", "")],
                related_nodes=[]
            )
            
            await self.vector_store.store_node(memory_node)
            await self.semantic_memory.store_terminology(memory_node)
            
            self.knowledge_graph.add_node(node_id, **asdict(memory_node))
            
            return [node_id]
            
        except Exception as e:
            logger.error(f"存储术语记忆失败: {e}")
            return []
    
    async def store_plot_event_memory(self, event_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """存储情节事件记忆"""
        try:
            event_name = event_data.get("name", "未命名事件")
            event_description = event_data.get("description", "")
            
            node_id = self.generate_node_id(MemoryType.PLOT_EVENT, event_name)
            embeddings = await self.generate_embeddings(f"{event_name} {event_description}")
            
            memory_node = MemoryNode(
                id=node_id,
                type=MemoryType.PLOT_EVENT,
                name=event_name,
                description=event_description,
                attributes=event_data,
                embeddings=embeddings,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                access_count=0,
                importance_score=0.0,
                context_chapters=[context.get("chapter_id", "")],
                related_nodes=[]
            )
            
            await self.vector_store.store_node(memory_node)
            await self.episodic_memory.store_plot_event(memory_node, context)
            
            self.knowledge_graph.add_node(node_id, **asdict(memory_node))
            
            return [node_id]
            
        except Exception as e:
            logger.error(f"存储情节事件记忆失败: {e}")
            return []
    
    async def store_relationship_memory(self, rel_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """存储关系记忆"""
        try:
            source = rel_data.get("source", "")
            target = rel_data.get("target", "")
            relation_type = rel_data.get("type", "")
            
            relationship_name = f"{source}_{relation_type}_{target}"
            node_id = self.generate_node_id(MemoryType.RELATIONSHIP, relationship_name)
            
            embeddings = await self.generate_embeddings(f"{source} {relation_type} {target}")
            
            memory_node = MemoryNode(
                id=node_id,
                type=MemoryType.RELATIONSHIP,
                name=relationship_name,
                description=rel_data.get("description", ""),
                attributes=rel_data,
                embeddings=embeddings,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                access_count=0,
                importance_score=0.0,
                context_chapters=[context.get("chapter_id", "")],
                related_nodes=[]
            )
            
            await self.vector_store.store_node(memory_node)
            self.knowledge_graph.add_node(node_id, **asdict(memory_node))
            
            # 在知识图谱中建立连接
            source_id = self.find_node_by_name(source)
            target_id = self.find_node_by_name(target)
            
            if source_id and target_id:
                self.knowledge_graph.add_edge(source_id, target_id, 
                                            relationship=relation_type, **rel_data)
            
            return [node_id]
            
        except Exception as e:
            logger.error(f"存储关系记忆失败: {e}")
            return []
    
    async def retrieve_contextual_memory(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """检索上下文记忆"""
        try:
            self.current_task = "检索上下文记忆"
            
            query = MemoryQuery(**query_data)
            
            # 多模式检索
            retrieval_results = await asyncio.gather(
                self.vector_store.similarity_search(query),
                self.semantic_memory.semantic_search(query),
                self.episodic_memory.episodic_search(query),
                self.temporal_memory.temporal_search(query)
            )
            
            # 合并和排序结果
            combined_results = await self.retrieval_optimizer.combine_results(
                retrieval_results, query
            )
            
            # 上下文增强
            enhanced_results = await self.context_manager.enhance_retrieval_context(
                combined_results, query
            )
            
            # 更新访问统计
            await self.update_access_statistics(enhanced_results)
            
            result = {
                "query": asdict(query),
                "results": enhanced_results,
                "result_count": len(enhanced_results),
                "retrieval_methods": ["vector", "semantic", "episodic", "temporal"],
                "retrieval_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "retrieval_duration": 0,
                    "cache_hit": query.query_text in self.retrieval_cache
                }
            }
            
            # 缓存结果
            self.retrieval_cache[query.query_text] = enhanced_results
            
            return result
            
        except Exception as e:
            logger.error(f"检索上下文记忆失败: {e}")
            raise
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """生成嵌入向量"""
        try:
            # 检查缓存
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
            
            # 使用OpenAI生成嵌入
            response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            embeddings = response.data[0].embedding
            
            # 缓存结果
            self.embedding_cache[text_hash] = embeddings
            
            return embeddings
            
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            return []
    
    def generate_node_id(self, memory_type: MemoryType, name: str) -> str:
        """生成节点ID"""
        return f"{memory_type.value}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
    
    def find_node_by_name(self, name: str) -> Optional[str]:
        """根据名称查找节点"""
        for node_id, node_data in self.knowledge_graph.nodes(data=True):
            if node_data.get("name") == name:
                return node_id
        return None
    
    def get_node_type_distribution(self, nodes: List[str]) -> Dict[str, int]:
        """获取节点类型分布"""
        distribution = defaultdict(int)
        for node_id in nodes:
            if self.knowledge_graph.has_node(node_id):
                node_type = self.knowledge_graph.nodes[node_id].get("type", "unknown")
                distribution[str(node_type)] += 1
        return dict(distribution)
    
    async def count_new_relationships(self) -> int:
        """计算新建关系数量"""
        return self.knowledge_graph.number_of_edges()
    
    async def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """获取知识图谱统计"""
        return {
            "nodes": self.knowledge_graph.number_of_nodes(),
            "edges": self.knowledge_graph.number_of_edges(),
            "density": nx.density(self.knowledge_graph),
            "components": nx.number_weakly_connected_components(self.knowledge_graph)
        }
    
    async def update_knowledge_graph(self, node_ids: List[str]):
        """更新知识图谱"""
        # 实现知识图谱更新逻辑
        pass
    
    async def establish_memory_relationships(self, node_ids: List[str], context: Dict[str, Any]):
        """建立记忆关系"""
        # 实现记忆关系建立逻辑
        pass
    
    async def calculate_importance_scores(self, node_ids: List[str]):
        """计算重要性分数"""
        # 实现重要性分数计算逻辑
        pass
    
    async def update_memory_indices(self, node_ids: List[str]):
        """更新记忆索引"""
        # 实现记忆索引更新逻辑
        pass
    
    async def update_access_statistics(self, results: List[Dict[str, Any]]):
        """更新访问统计"""
        # 实现访问统计更新逻辑
        pass
    
    async def load_persistent_memory(self):
        """加载持久化记忆"""
        try:
            # 从文件或数据库加载记忆
            pass
        except Exception as e:
            logger.warning(f"加载持久化记忆失败: {e}")
    
    async def save_persistent_memory(self):
        """保存持久化记忆"""
        try:
            # 保存记忆到文件或数据库
            pass
        except Exception as e:
            logger.error(f"保存持久化记忆失败: {e}")
    
    async def cleanup(self) -> bool:
        """清理资源"""
        try:
            logger.info("清理增强版记忆智能体资源...")
            
            # 保存持久化记忆
            await self.save_persistent_memory()
            
            # 清理各个组件
            await self.vector_store.cleanup()
            await self.temporal_memory.cleanup()
            await self.semantic_memory.cleanup()
            await self.episodic_memory.cleanup()
            
            await self.relationship_analyzer.cleanup()
            await self.context_manager.cleanup()
            await self.importance_calculator.cleanup()
            await self.memory_consolidator.cleanup()
            await self.retrieval_optimizer.cleanup()
            
            # 清理缓存
            self.retrieval_cache.clear()
            self.embedding_cache.clear()
            
            return True
        except Exception as e:
            logger.error(f"清理增强版记忆智能体资源失败: {e}")
            return False


# 支持类实现
class VectorMemoryStore:
    """向量记忆存储"""
    
    async def initialize(self):
        pass
    
    async def store_node(self, node: MemoryNode):
        pass
    
    async def similarity_search(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        return []
    
    async def cleanup(self):
        pass


class TemporalMemoryManager:
    """时间记忆管理器"""
    
    async def initialize(self):
        pass
    
    async def temporal_search(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        return []
    
    async def cleanup(self):
        pass


class SemanticMemoryManager:
    """语义记忆管理器"""
    
    async def initialize(self):
        pass
    
    async def store_character(self, node: MemoryNode):
        pass
    
    async def store_location(self, node: MemoryNode):
        pass
    
    async def store_item(self, node: MemoryNode):
        pass
    
    async def store_terminology(self, node: MemoryNode):
        pass
    
    async def semantic_search(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        return []
    
    async def cleanup(self):
        pass


class EpisodicMemoryManager:
    """情节记忆管理器"""
    
    async def initialize(self):
        pass
    
    async def add_character_appearance(self, node: MemoryNode, context: Dict[str, Any]):
        pass
    
    async def store_plot_event(self, node: MemoryNode, context: Dict[str, Any]):
        pass
    
    async def episodic_search(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        return []
    
    async def cleanup(self):
        pass


class RelationshipAnalyzer:
    """关系分析器"""
    
    async def initialize(self):
        pass
    
    async def cleanup(self):
        pass


class ContextualMemoryManager:
    """上下文记忆管理器"""
    
    async def initialize(self):
        pass
    
    async def enhance_retrieval_context(self, results: List[Dict[str, Any]], query: MemoryQuery) -> List[Dict[str, Any]]:
        return results
    
    async def cleanup(self):
        pass


class ImportanceCalculator:
    """重要性计算器"""
    
    async def initialize(self):
        pass
    
    async def cleanup(self):
        pass


class MemoryConsolidator:
    """记忆整合器"""
    
    async def initialize(self):
        pass
    
    async def consolidate_new_memories(self, nodes: List[str]):
        pass
    
    async def cleanup(self):
        pass


class RetrievalOptimizer:
    """检索优化器"""
    
    async def initialize(self):
        pass
    
    async def combine_results(self, results: List[List[Dict[str, Any]]], query: MemoryQuery) -> List[Dict[str, Any]]:
        # 合并多个检索结果
        combined = []
        for result_list in results:
            combined.extend(result_list)
        return combined
    
    async def cleanup(self):
        pass 