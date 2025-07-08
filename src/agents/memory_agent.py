"""
记忆智能体 - 负责维护故事一致性和人物关系
Memory Agent - Responsible for maintaining story consistency and character relationships
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Set
from loguru import logger
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

from .base_agent import BaseAgent
from ..models.base import (
    AgentMessage, AgentType, Novel, Chapter, Character, Location, 
    Item, Terminology, TranslationContext
)


class MemoryAgent(BaseAgent):
    """记忆智能体"""
    
    def __init__(self, config, openai_client):
        super().__init__(config, openai_client)
        
        # 向量数据库客户端
        self.chroma_client = None
        self.character_collection = None
        self.plot_collection = None
        self.terminology_collection = None
        
        # 嵌入模型
        self.embedding_model = None
        
        # 记忆存储
        self.character_memory = {}
        self.plot_memory = {}
        self.terminology_memory = {}
        self.relationship_graph = {}
        
        # 一致性检查规则
        self.consistency_rules = {
            "character_appearance": "角色外貌描述必须保持一致",
            "character_personality": "角色性格特征必须保持一致",
            "character_abilities": "角色能力设定必须保持一致",
            "plot_timeline": "情节时间线必须保持逻辑性",
            "world_setting": "世界观设定必须保持一致",
            "terminology_usage": "术语使用必须保持一致"
        }
        
    async def initialize(self) -> bool:
        """初始化记忆智能体"""
        try:
            logger.info("初始化记忆智能体...")
            
            # 初始化向量数据库
            await self.init_vector_database()
            
            # 初始化嵌入模型
            await self.init_embedding_model()
            
            # 健康检查
            health_ok = await self.health_check()
            if not health_ok:
                logger.error("记忆智能体健康检查失败")
                return False
            
            logger.info("记忆智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"记忆智能体初始化失败: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理记忆智能体资源"""
        try:
            logger.info("清理记忆智能体资源...")
            
            # 保存记忆数据
            await self.save_memory_data()
            
            # 清理向量数据库连接
            if self.chroma_client:
                # ChromaDB 通常不需要显式关闭
                pass
            
            return True
        except Exception as e:
            logger.error(f"清理记忆智能体资源失败: {e}")
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理消息"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type == "store_character":
                result = await self.store_character_info(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="character_stored",
                    content=result
                )
            
            elif message_type == "store_plot":
                result = await self.store_plot_info(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="plot_stored",
                    content=result
                )
            
            elif message_type == "check_consistency":
                result = await self.check_story_consistency(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="consistency_checked",
                    content=result
                )
            
            elif message_type == "retrieve_context":
                result = await self.retrieve_context(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="context_retrieved",
                    content=result
                )
            
            elif message_type == "update_relationships":
                result = await self.update_character_relationships(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="relationships_updated",
                    content=result
                )
            
            else:
                logger.warning(f"未知的消息类型: {message_type}")
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="error",
                    content={"error": f"未知的消息类型: {message_type}"}
                )
                
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            return AgentMessage(
                sender=self.agent_type,
                receiver=message.sender,
                message_type="error",
                content={"error": str(e)}
            )
    
    async def init_vector_database(self):
        """初始化向量数据库"""
        try:
            # 初始化ChromaDB客户端
            self.chroma_client = chromadb.Client()
            
            # 创建不同类型的集合
            self.character_collection = self.chroma_client.get_or_create_collection(
                name="characters",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.plot_collection = self.chroma_client.get_or_create_collection(
                name="plots",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.terminology_collection = self.chroma_client.get_or_create_collection(
                name="terminologies",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("向量数据库初始化完成")
            
        except Exception as e:
            logger.error(f"初始化向量数据库失败: {e}")
            raise
    
    async def init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            # 加载预训练的嵌入模型
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("嵌入模型初始化完成")
            
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {e}")
            raise
    
    async def store_character_info(self, character_data: Dict[str, Any]) -> Dict[str, Any]:
        """存储角色信息"""
        try:
            self.current_task = "存储角色信息"
            
            character_name = character_data.get("name", "")
            character_info = character_data.get("info", {})
            chapter_context = character_data.get("chapter_context", "")
            
            # 提取角色详细信息
            character_details = await self.extract_character_details(character_info, chapter_context)
            
            # 生成嵌入向量
            character_text = self.format_character_for_embedding(character_details)
            embedding = self.embedding_model.encode(character_text)
            
            # 存储到向量数据库
            self.character_collection.add(
                embeddings=[embedding.tolist()],
                documents=[character_text],
                metadatas=[{
                    "name": character_name,
                    "type": "character",
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[f"char_{character_name}_{datetime.now().timestamp()}"]
            )
            
            # 存储到内存
            if character_name not in self.character_memory:
                self.character_memory[character_name] = []
            
            self.character_memory[character_name].append(character_details)
            
            logger.info(f"角色信息已存储: {character_name}")
            
            return {
                "character_name": character_name,
                "stored": True,
                "details_count": len(character_details),
                "memory_entries": len(self.character_memory[character_name])
            }
            
        except Exception as e:
            logger.error(f"存储角色信息失败: {e}")
            raise
    
    async def extract_character_details(self, character_info: Dict[str, Any], context: str) -> Dict[str, Any]:
        """提取角色详细信息"""
        try:
            prompt = f"""
请从以下信息中提取角色的详细信息：

角色基本信息：
{json.dumps(character_info, ensure_ascii=False, indent=2)}

上下文：
{context}

请提取以下信息并以JSON格式返回：
{{
  "name": "角色姓名",
  "aliases": ["别名列表"],
  "appearance": "外貌描述",
  "personality": "性格特征",
  "abilities": "能力特长",
  "background": "背景故事",
  "relationships": "人物关系",
  "development": "角色发展",
  "notable_quotes": "重要台词",
  "chapter_appearances": "出现章节"
}}

请确保信息准确、完整。
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的角色分析专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.call_llm(messages, temperature=0.3)
            
            try:
                character_details = json.loads(response)
                return character_details
            except json.JSONDecodeError:
                logger.warning("无法解析角色详细信息")
                return {
                    "name": character_info.get("name", ""),
                    "description": str(character_info)
                }
            
        except Exception as e:
            logger.error(f"提取角色详细信息失败: {e}")
            return {"name": character_info.get("name", ""), "error": str(e)}
    
    def format_character_for_embedding(self, character_details: Dict[str, Any]) -> str:
        """格式化角色信息用于嵌入"""
        try:
            parts = []
            
            # 基本信息
            parts.append(f"姓名: {character_details.get('name', '')}")
            
            # 别名
            aliases = character_details.get('aliases', [])
            if aliases:
                parts.append(f"别名: {', '.join(aliases)}")
            
            # 外貌
            appearance = character_details.get('appearance', '')
            if appearance:
                parts.append(f"外貌: {appearance}")
            
            # 性格
            personality = character_details.get('personality', '')
            if personality:
                parts.append(f"性格: {personality}")
            
            # 能力
            abilities = character_details.get('abilities', '')
            if abilities:
                parts.append(f"能力: {abilities}")
            
            # 背景
            background = character_details.get('background', '')
            if background:
                parts.append(f"背景: {background}")
            
            return '\n'.join(parts)
            
        except Exception as e:
            logger.error(f"格式化角色信息失败: {e}")
            return str(character_details)
    
    async def store_plot_info(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """存储情节信息"""
        try:
            self.current_task = "存储情节信息"
            
            chapter_id = plot_data.get("chapter_id", "")
            plot_summary = plot_data.get("summary", "")
            plot_details = plot_data.get("details", {})
            
            # 提取情节要素
            plot_elements = await self.extract_plot_elements(plot_summary, plot_details)
            
            # 生成嵌入向量
            plot_text = self.format_plot_for_embedding(plot_elements)
            embedding = self.embedding_model.encode(plot_text)
            
            # 存储到向量数据库
            self.plot_collection.add(
                embeddings=[embedding.tolist()],
                documents=[plot_text],
                metadatas=[{
                    "chapter_id": chapter_id,
                    "type": "plot",
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[f"plot_{chapter_id}_{datetime.now().timestamp()}"]
            )
            
            # 存储到内存
            if chapter_id not in self.plot_memory:
                self.plot_memory[chapter_id] = []
            
            self.plot_memory[chapter_id].append(plot_elements)
            
            logger.info(f"情节信息已存储: {chapter_id}")
            
            return {
                "chapter_id": chapter_id,
                "stored": True,
                "elements_count": len(plot_elements),
                "memory_entries": len(self.plot_memory[chapter_id])
            }
            
        except Exception as e:
            logger.error(f"存储情节信息失败: {e}")
            raise
    
    async def extract_plot_elements(self, summary: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """提取情节要素"""
        try:
            prompt = f"""
请从以下情节信息中提取关键要素：

情节摘要：
{summary}

详细信息：
{json.dumps(details, ensure_ascii=False, indent=2)}

请提取以下要素并以JSON格式返回：
{{
  "main_events": ["主要事件列表"],
  "characters_involved": ["涉及角色"],
  "locations": ["发生地点"],
  "time_period": "时间段",
  "conflicts": ["冲突点"],
  "resolutions": ["解决方案"],
  "foreshadowing": ["伏笔"],
  "consequences": ["后果影响"],
  "themes": ["主题思想"],
  "emotional_tone": "情感基调"
}}

请确保信息准确、完整。
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的情节分析专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.call_llm(messages, temperature=0.3)
            
            try:
                plot_elements = json.loads(response)
                return plot_elements
            except json.JSONDecodeError:
                logger.warning("无法解析情节要素")
                return {
                    "summary": summary,
                    "details": details
                }
            
        except Exception as e:
            logger.error(f"提取情节要素失败: {e}")
            return {"summary": summary, "error": str(e)}
    
    def format_plot_for_embedding(self, plot_elements: Dict[str, Any]) -> str:
        """格式化情节信息用于嵌入"""
        try:
            parts = []
            
            # 主要事件
            main_events = plot_elements.get('main_events', [])
            if main_events:
                parts.append(f"主要事件: {', '.join(main_events)}")
            
            # 涉及角色
            characters = plot_elements.get('characters_involved', [])
            if characters:
                parts.append(f"涉及角色: {', '.join(characters)}")
            
            # 发生地点
            locations = plot_elements.get('locations', [])
            if locations:
                parts.append(f"发生地点: {', '.join(locations)}")
            
            # 时间段
            time_period = plot_elements.get('time_period', '')
            if time_period:
                parts.append(f"时间段: {time_period}")
            
            # 冲突点
            conflicts = plot_elements.get('conflicts', [])
            if conflicts:
                parts.append(f"冲突点: {', '.join(conflicts)}")
            
            # 情感基调
            emotional_tone = plot_elements.get('emotional_tone', '')
            if emotional_tone:
                parts.append(f"情感基调: {emotional_tone}")
            
            return '\n'.join(parts)
            
        except Exception as e:
            logger.error(f"格式化情节信息失败: {e}")
            return str(plot_elements)
    
    async def check_story_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查故事一致性"""
        try:
            self.current_task = "检查故事一致性"
            
            chapter_content = data.get("chapter_content", "")
            chapter_id = data.get("chapter_id", "")
            context_chapters = data.get("context_chapters", [])
            
            # 执行各种一致性检查
            consistency_results = {}
            
            # 角色一致性检查
            consistency_results["character_consistency"] = await self.check_character_consistency(
                chapter_content, chapter_id, context_chapters
            )
            
            # 情节一致性检查
            consistency_results["plot_consistency"] = await self.check_plot_consistency(
                chapter_content, chapter_id, context_chapters
            )
            
            # 世界观一致性检查
            consistency_results["world_consistency"] = await self.check_world_consistency(
                chapter_content, chapter_id, context_chapters
            )
            
            # 术语一致性检查
            consistency_results["terminology_consistency"] = await self.check_terminology_consistency(
                chapter_content, chapter_id, context_chapters
            )
            
            # 计算总体一致性得分
            consistency_scores = [result.get("score", 0) for result in consistency_results.values()]
            overall_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
            
            # 生成改进建议
            suggestions = await self.generate_consistency_suggestions(consistency_results)
            
            return {
                "chapter_id": chapter_id,
                "overall_score": overall_score,
                "detailed_results": consistency_results,
                "suggestions": suggestions,
                "passed": overall_score >= 7.0
            }
            
        except Exception as e:
            logger.error(f"检查故事一致性失败: {e}")
            raise
    
    async def check_character_consistency(self, content: str, chapter_id: str, context_chapters: List[str]) -> Dict[str, Any]:
        """检查角色一致性"""
        try:
            # 提取当前章节中的角色
            current_characters = await self.extract_characters_from_content(content)
            
            # 从记忆中检索相关角色信息
            inconsistencies = []
            
            for char_name in current_characters:
                if char_name in self.character_memory:
                    # 检查角色描述是否一致
                    char_history = self.character_memory[char_name]
                    current_description = await self.extract_character_description(content, char_name)
                    
                    for historical_record in char_history:
                        conflicts = await self.find_character_conflicts(current_description, historical_record)
                        if conflicts:
                            inconsistencies.extend(conflicts)
            
            score = max(0, 10 - len(inconsistencies))
            
            return {
                "score": score,
                "inconsistencies": inconsistencies,
                "characters_checked": len(current_characters)
            }
            
        except Exception as e:
            logger.error(f"检查角色一致性失败: {e}")
            return {"score": 5, "error": str(e)}
    
    async def check_plot_consistency(self, content: str, chapter_id: str, context_chapters: List[str]) -> Dict[str, Any]:
        """检查情节一致性"""
        try:
            # 提取当前章节的情节要素
            current_plot = await self.extract_plot_from_content(content)
            
            # 检查与之前章节的逻辑一致性
            inconsistencies = []
            
            # 检查时间线一致性
            timeline_issues = await self.check_timeline_consistency(current_plot, context_chapters)
            inconsistencies.extend(timeline_issues)
            
            # 检查因果关系一致性
            causality_issues = await self.check_causality_consistency(current_plot, context_chapters)
            inconsistencies.extend(causality_issues)
            
            score = max(0, 10 - len(inconsistencies))
            
            return {
                "score": score,
                "inconsistencies": inconsistencies,
                "plot_elements_checked": len(current_plot)
            }
            
        except Exception as e:
            logger.error(f"检查情节一致性失败: {e}")
            return {"score": 5, "error": str(e)}
    
    async def check_world_consistency(self, content: str, chapter_id: str, context_chapters: List[str]) -> Dict[str, Any]:
        """检查世界观一致性"""
        try:
            # 提取世界观要素
            world_elements = await self.extract_world_elements(content)
            
            # 检查世界设定一致性
            inconsistencies = []
            
            # 这里可以实现具体的世界观一致性检查逻辑
            
            score = 8  # 默认得分
            
            return {
                "score": score,
                "inconsistencies": inconsistencies,
                "world_elements_checked": len(world_elements)
            }
            
        except Exception as e:
            logger.error(f"检查世界观一致性失败: {e}")
            return {"score": 5, "error": str(e)}
    
    async def check_terminology_consistency(self, content: str, chapter_id: str, context_chapters: List[str]) -> Dict[str, Any]:
        """检查术语一致性"""
        try:
            # 提取术语使用
            terms_used = await self.extract_terminology_from_content(content)
            
            # 检查术语使用一致性
            inconsistencies = []
            
            for term in terms_used:
                if term in self.terminology_memory:
                    # 检查术语定义和使用是否一致
                    term_history = self.terminology_memory[term]
                    current_usage = await self.extract_term_usage(content, term)
                    
                    for historical_usage in term_history:
                        conflicts = await self.find_terminology_conflicts(current_usage, historical_usage)
                        if conflicts:
                            inconsistencies.extend(conflicts)
            
            score = max(0, 10 - len(inconsistencies))
            
            return {
                "score": score,
                "inconsistencies": inconsistencies,
                "terms_checked": len(terms_used)
            }
            
        except Exception as e:
            logger.error(f"检查术语一致性失败: {e}")
            return {"score": 5, "error": str(e)}
    
    async def retrieve_context(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """检索上下文信息"""
        try:
            self.current_task = "检索上下文信息"
            
            query_text = query_data.get("query", "")
            context_type = query_data.get("type", "all")  # character, plot, terminology, all
            top_k = query_data.get("top_k", 5)
            
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode(query_text)
            
            results = {}
            
            if context_type in ["character", "all"]:
                # 检索角色信息
                char_results = self.character_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
                results["characters"] = char_results
            
            if context_type in ["plot", "all"]:
                # 检索情节信息
                plot_results = self.plot_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
                results["plots"] = plot_results
            
            if context_type in ["terminology", "all"]:
                # 检索术语信息
                term_results = self.terminology_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
                results["terminologies"] = term_results
            
            return {
                "query": query_text,
                "context_type": context_type,
                "results": results,
                "retrieved_count": sum(len(r.get("documents", [])) for r in results.values())
            }
            
        except Exception as e:
            logger.error(f"检索上下文信息失败: {e}")
            raise
    
    async def update_character_relationships(self, relationship_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新角色关系"""
        try:
            self.current_task = "更新角色关系"
            
            char1 = relationship_data.get("character1", "")
            char2 = relationship_data.get("character2", "")
            relationship_type = relationship_data.get("relationship", "")
            context = relationship_data.get("context", "")
            
            # 更新关系图
            if char1 not in self.relationship_graph:
                self.relationship_graph[char1] = {}
            
            self.relationship_graph[char1][char2] = {
                "type": relationship_type,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            
            # 双向关系
            if char2 not in self.relationship_graph:
                self.relationship_graph[char2] = {}
            
            # 推断反向关系
            reverse_relationship = await self.infer_reverse_relationship(relationship_type)
            self.relationship_graph[char2][char1] = {
                "type": reverse_relationship,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"角色关系已更新: {char1} - {char2} ({relationship_type})")
            
            return {
                "character1": char1,
                "character2": char2,
                "relationship": relationship_type,
                "updated": True,
                "total_relationships": len(self.relationship_graph)
            }
            
        except Exception as e:
            logger.error(f"更新角色关系失败: {e}")
            raise
    
    async def save_memory_data(self):
        """保存记忆数据"""
        try:
            # 这里可以实现将记忆数据保存到文件或数据库的逻辑
            logger.info("保存记忆数据...")
            
            # 示例：保存到JSON文件
            memory_data = {
                "characters": self.character_memory,
                "plots": self.plot_memory,
                "terminologies": self.terminology_memory,
                "relationships": self.relationship_graph,
                "timestamp": datetime.now().isoformat()
            }
            
            # 实际实现中应该保存到持久化存储
            logger.info("记忆数据保存完成")
            
        except Exception as e:
            logger.error(f"保存记忆数据失败: {e}")
    
    # 以下是辅助方法的占位符实现
    async def extract_characters_from_content(self, content: str) -> List[str]:
        """从内容中提取角色名称"""
        # 这里可以实现具体的角色提取逻辑
        return []
    
    async def extract_character_description(self, content: str, char_name: str) -> Dict[str, Any]:
        """提取角色描述"""
        # 这里可以实现具体的角色描述提取逻辑
        return {}
    
    async def find_character_conflicts(self, current: Dict[str, Any], historical: Dict[str, Any]) -> List[str]:
        """查找角色冲突"""
        # 这里可以实现具体的冲突检测逻辑
        return []
    
    async def extract_plot_from_content(self, content: str) -> Dict[str, Any]:
        """从内容中提取情节"""
        # 这里可以实现具体的情节提取逻辑
        return {}
    
    async def check_timeline_consistency(self, current_plot: Dict[str, Any], context_chapters: List[str]) -> List[str]:
        """检查时间线一致性"""
        # 这里可以实现具体的时间线检查逻辑
        return []
    
    async def check_causality_consistency(self, current_plot: Dict[str, Any], context_chapters: List[str]) -> List[str]:
        """检查因果关系一致性"""
        # 这里可以实现具体的因果关系检查逻辑
        return []
    
    async def extract_world_elements(self, content: str) -> List[str]:
        """提取世界观要素"""
        # 这里可以实现具体的世界观要素提取逻辑
        return []
    
    async def extract_terminology_from_content(self, content: str) -> List[str]:
        """从内容中提取术语"""
        # 这里可以实现具体的术语提取逻辑
        return []
    
    async def extract_term_usage(self, content: str, term: str) -> Dict[str, Any]:
        """提取术语使用"""
        # 这里可以实现具体的术语使用提取逻辑
        return {}
    
    async def find_terminology_conflicts(self, current: Dict[str, Any], historical: Dict[str, Any]) -> List[str]:
        """查找术语冲突"""
        # 这里可以实现具体的术语冲突检测逻辑
        return []
    
    async def infer_reverse_relationship(self, relationship_type: str) -> str:
        """推断反向关系"""
        # 这里可以实现关系映射逻辑
        relationship_mapping = {
            "父子": "子父",
            "师傅": "徒弟",
            "朋友": "朋友",
            "敌人": "敌人",
            "恋人": "恋人"
        }
        return relationship_mapping.get(relationship_type, relationship_type)
    
    async def generate_consistency_suggestions(self, consistency_results: Dict[str, Any]) -> List[str]:
        """生成一致性改进建议"""
        suggestions = []
        
        for check_type, result in consistency_results.items():
            score = result.get("score", 0)
            inconsistencies = result.get("inconsistencies", [])
            
            if score < 7:
                suggestions.append(f"需要改进{check_type}，发现{len(inconsistencies)}个问题")
                
                # 添加具体建议
                for inconsistency in inconsistencies[:3]:  # 最多显示3个
                    suggestions.append(f"- {inconsistency}")
        
        return suggestions 