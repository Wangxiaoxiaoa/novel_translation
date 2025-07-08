"""
增强版解析智能体 - 深度内容理解和结构分析
Enhanced Parser Agent - Advanced content understanding and structural analysis
"""

import asyncio
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from loguru import logger
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import jieba
import jieba.analyse
from datetime import datetime

from .base_agent import BaseAgent
from ..models.base import (
    AgentMessage, AgentType, Novel, Chapter, Character, Location, 
    Item, Terminology, ProcessingStatus
)


class EnhancedParserAgent(BaseAgent):
    """增强版解析智能体"""
    
    def __init__(self, config, openai_client):
        super().__init__(config, openai_client)
        
        # 高级分析模块
        self.plot_analyzer = PlotStructureAnalyzer()
        self.character_analyzer = CharacterNetworkAnalyzer()
        self.style_analyzer = WritingStyleAnalyzer()
        self.timeline_manager = TimelineManager()
        self.emotion_analyzer = EmotionAnalyzer()
        
        # 知识图谱
        self.knowledge_graph = nx.DiGraph()
        
        # 高级章节分割算法
        self.chapter_splitter = IntelligentChapterSplitter()
        
        # 内容理解模型
        self.content_understanding = ContentUnderstanding()
        
        # 缓存系统
        self.analysis_cache = {}
        
    async def initialize(self) -> bool:
        """初始化增强版解析智能体"""
        try:
            logger.info("初始化增强版解析智能体...")
            
            # 初始化jieba分词
            jieba.initialize()
            
            # 初始化各个分析模块
            await self.plot_analyzer.initialize()
            await self.character_analyzer.initialize()
            await self.style_analyzer.initialize()
            await self.timeline_manager.initialize()
            await self.emotion_analyzer.initialize()
            
            # 初始化章节分割器
            await self.chapter_splitter.initialize()
            
            # 初始化内容理解模块
            await self.content_understanding.initialize()
            
            # 健康检查
            health_ok = await self.health_check()
            if not health_ok:
                logger.error("增强版解析智能体健康检查失败")
                return False
                
            self.status = ProcessingStatus.COMPLETED
            logger.info("增强版解析智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"增强版解析智能体初始化失败: {e}")
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理消息"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type == "deep_parse_document":
                result = await self.deep_parse_document(content)
            elif message_type == "intelligent_chapter_split":
                result = await self.intelligent_chapter_split(content)
            elif message_type == "analyze_plot_structure":
                result = await self.analyze_plot_structure(content)
            elif message_type == "build_character_network":
                result = await self.build_character_network(content)
            elif message_type == "analyze_writing_style":
                result = await self.analyze_writing_style(content)
            elif message_type == "extract_timeline":
                result = await self.extract_timeline(content)
            elif message_type == "analyze_emotions":
                result = await self.analyze_emotions(content)
            elif message_type == "build_knowledge_graph":
                result = await self.build_knowledge_graph(content)
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
    
    async def deep_parse_document(self, file_path: str) -> Dict[str, Any]:
        """深度解析文档"""
        try:
            self.current_task = f"深度解析文档: {file_path}"
            logger.info(f"开始深度解析文档: {file_path}")
            
            # 基础解析
            basic_result = await self.parse_document(file_path)
            content = basic_result["content"]
            
            # 深度分析
            analysis_results = await asyncio.gather(
                self.content_understanding.analyze_content(content),
                self.plot_analyzer.analyze_structure(content),
                self.character_analyzer.extract_characters(content),
                self.style_analyzer.analyze_style(content),
                self.timeline_manager.extract_timeline(content),
                self.emotion_analyzer.analyze_emotions(content)
            )
            
            content_analysis, plot_structure, characters, style_analysis, timeline, emotions = analysis_results
            
            # 构建知识图谱
            knowledge_graph = await self.build_comprehensive_knowledge_graph(
                content, characters, plot_structure, timeline
            )
            
            # 整合结果
            enhanced_result = {
                **basic_result,
                "deep_analysis": {
                    "content_understanding": content_analysis,
                    "plot_structure": plot_structure,
                    "character_network": characters,
                    "writing_style": style_analysis,
                    "timeline": timeline,
                    "emotion_analysis": emotions,
                    "knowledge_graph": knowledge_graph,
                    "complexity_score": await self.calculate_complexity_score(content),
                    "readability_score": await self.calculate_readability_score(content),
                    "genre_classification": await self.classify_genre(content),
                    "themes": await self.extract_themes(content)
                }
            }
            
            logger.info(f"深度解析完成: {file_path}")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"深度解析文档失败: {e}")
            raise
    
    async def intelligent_chapter_split(self, data: Dict[str, Any]) -> List[Chapter]:
        """智能章节切分"""
        try:
            self.current_task = "智能章节切分"
            content = data.get("content", "")
            
            # 使用多种算法进行章节切分
            split_results = await asyncio.gather(
                self.chapter_splitter.semantic_split(content),
                self.chapter_splitter.structural_split(content),
                self.chapter_splitter.topic_split(content),
                self.chapter_splitter.emotional_split(content)
            )
            
            # 融合多种切分结果
            optimal_splits = await self.chapter_splitter.fuse_splits(split_results, content)
            
            # 生成章节对象
            chapters = []
            for i, chapter_content in enumerate(optimal_splits):
                # 深度分析每个章节
                chapter_analysis = await self.analyze_single_chapter(chapter_content, i + 1)
                
                chapter = Chapter(
                    id=f"chapter_{i+1:03d}",
                    title=chapter_analysis["title"],
                    content=chapter_content.strip(),
                    original_content=chapter_content.strip(),
                    chapter_number=i + 1,
                    word_count=len(chapter_content),
                    characters=chapter_analysis["characters"],
                    locations=chapter_analysis["locations"],
                    items=chapter_analysis["items"],
                    terminologies=chapter_analysis["terminologies"],
                    summary=chapter_analysis["summary"],
                    metadata={
                        "emotion_arc": chapter_analysis["emotion_arc"],
                        "plot_points": chapter_analysis["plot_points"],
                        "themes": chapter_analysis["themes"],
                        "complexity": chapter_analysis["complexity"],
                        "pacing": chapter_analysis["pacing"],
                        "conflicts": chapter_analysis["conflicts"],
                        "resolutions": chapter_analysis["resolutions"]
                    }
                )
                chapters.append(chapter)
            
            logger.info(f"智能章节切分完成，共 {len(chapters)} 章")
            return chapters
            
        except Exception as e:
            logger.error(f"智能章节切分失败: {e}")
            raise
    
    async def analyze_single_chapter(self, content: str, chapter_num: int) -> Dict[str, Any]:
        """深度分析单个章节"""
        try:
            # 并行执行多种分析
            analysis_tasks = [
                self.extract_chapter_title(content, chapter_num),
                self.extract_advanced_characters(content),
                self.extract_advanced_locations(content),
                self.extract_advanced_items(content),
                self.extract_advanced_terminologies(content),
                self.generate_advanced_summary(content),
                self.analyze_emotion_arc(content),
                self.extract_plot_points(content),
                self.extract_chapter_themes(content),
                self.calculate_chapter_complexity(content),
                self.analyze_pacing(content),
                self.extract_conflicts(content),
                self.extract_resolutions(content)
            ]
            
            results = await asyncio.gather(*analysis_tasks)
            
            return {
                "title": results[0],
                "characters": results[1],
                "locations": results[2],
                "items": results[3],
                "terminologies": results[4],
                "summary": results[5],
                "emotion_arc": results[6],
                "plot_points": results[7],
                "themes": results[8],
                "complexity": results[9],
                "pacing": results[10],
                "conflicts": results[11],
                "resolutions": results[12]
            }
            
        except Exception as e:
            logger.error(f"分析章节失败: {e}")
            return {"title": f"第{chapter_num}章", "error": str(e)}
    
    async def extract_advanced_characters(self, content: str) -> List[Dict[str, Any]]:
        """高级角色提取"""
        try:
            # 使用NER和语义分析提取角色
            characters = []
            
            # 基础角色提取
            basic_chars = await self.extract_characters(content)
            
            # 使用LLM进行深度角色分析
            prompt = f"""
            请从以下文本中提取所有角色信息，包括：
            1. 角色姓名（包括别名、称号）
            2. 角色描述（外貌、性格、能力）
            3. 角色关系（与其他角色的关系）
            4. 角色作用（主角、配角、反派等）
            5. 角色发展（在本章中的变化）
            
            文本内容：
            {content[:2000]}...
            
            请以JSON格式返回，格式如下：
            {{
                "characters": [
                    {{
                        "name": "角色姓名",
                        "aliases": ["别名1", "别名2"],
                        "description": "角色描述",
                        "relationships": ["关系描述"],
                        "role": "角色作用",
                        "development": "角色发展",
                        "appearance_type": "登场方式",
                        "emotional_state": "情感状态"
                    }}
                ]
            }}
            """
            
            messages = [
                {"role": "system", "content": "你是一个专业的文学分析专家，擅长角色分析。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.call_llm(messages, temperature=0.3)
            
            try:
                char_data = json.loads(response)
                characters = char_data.get("characters", [])
            except json.JSONDecodeError:
                logger.warning("角色分析结果解析失败，使用基础提取")
                characters = [{"name": name, "description": ""} for name in basic_chars]
            
            return characters
            
        except Exception as e:
            logger.error(f"高级角色提取失败: {e}")
            return []
    
    async def extract_advanced_locations(self, content: str) -> List[Dict[str, Any]]:
        """高级地点提取"""
        try:
            prompt = f"""
            请从以下文本中提取所有地点信息，包括：
            1. 地点名称
            2. 地点类型（城市、建筑、自然景观等）
            3. 地点描述
            4. 地点在故事中的重要性
            5. 与其他地点的关系
            
            文本内容：
            {content[:2000]}...
            
            请以JSON格式返回。
            """
            
            messages = [
                {"role": "system", "content": "你是一个专业的文学分析专家，擅长场景分析。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.call_llm(messages, temperature=0.3)
            
            try:
                location_data = json.loads(response)
                return location_data.get("locations", [])
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            logger.error(f"高级地点提取失败: {e}")
            return []
    
    async def analyze_emotion_arc(self, content: str) -> Dict[str, Any]:
        """分析情感弧线"""
        try:
            return await self.emotion_analyzer.analyze_emotion_arc(content)
        except Exception as e:
            logger.error(f"情感弧线分析失败: {e}")
            return {}
    
    async def extract_plot_points(self, content: str) -> List[Dict[str, Any]]:
        """提取情节要点"""
        try:
            prompt = f"""
            请分析以下文本的情节要点，包括：
            1. 起始情况
            2. 触发事件
            3. 发展过程
            4. 高潮部分
            5. 结局状态
            6. 转折点
            7. 伏笔
            
            文本内容：
            {content}
            
            请以JSON格式返回情节分析结果。
            """
            
            messages = [
                {"role": "system", "content": "你是一个专业的情节分析专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.call_llm(messages, temperature=0.4)
            
            try:
                plot_data = json.loads(response)
                return plot_data.get("plot_points", [])
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            logger.error(f"提取情节要点失败: {e}")
            return []
    
    async def build_comprehensive_knowledge_graph(self, content: str, characters: List, 
                                                 plot_structure: Dict, timeline: List) -> Dict[str, Any]:
        """构建综合知识图谱"""
        try:
            # 创建知识图谱
            kg = nx.DiGraph()
            
            # 添加角色节点
            for char in characters:
                kg.add_node(char["name"], type="character", **char)
            
            # 添加地点节点
            locations = await self.extract_advanced_locations(content)
            for loc in locations:
                kg.add_node(loc["name"], type="location", **loc)
            
            # 添加事件节点
            for event in timeline:
                kg.add_node(event["event"], type="event", **event)
            
            # 添加关系边
            await self.add_relationships_to_graph(kg, content)
            
            # 计算图谱统计信息
            graph_stats = {
                "nodes": kg.number_of_nodes(),
                "edges": kg.number_of_edges(),
                "density": nx.density(kg),
                "clustering": nx.average_clustering(kg.to_undirected()),
                "centrality": dict(nx.degree_centrality(kg))
            }
            
            # 导出图谱数据
            graph_data = {
                "nodes": [{"id": node, **data} for node, data in kg.nodes(data=True)],
                "edges": [{"source": u, "target": v, **data} for u, v, data in kg.edges(data=True)],
                "statistics": graph_stats
            }
            
            return graph_data
            
        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            return {}
    
    async def add_relationships_to_graph(self, graph: nx.DiGraph, content: str):
        """添加关系到知识图谱"""
        try:
            # 使用LLM提取关系
            prompt = f"""
            请从以下文本中提取实体之间的关系，包括：
            1. 人物之间的关系（朋友、敌人、师徒、恋人等）
            2. 人物与地点的关系（居住、工作、出生等）
            3. 人物与事件的关系（参与、导致、受影响等）
            4. 地点之间的关系（包含、邻近、对立等）
            
            文本内容：
            {content[:3000]}...
            
            请以JSON格式返回关系列表。
            """
            
            messages = [
                {"role": "system", "content": "你是一个专业的关系提取专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.call_llm(messages, temperature=0.3)
            
            try:
                relationship_data = json.loads(response)
                relationships = relationship_data.get("relationships", [])
                
                for rel in relationships:
                    source = rel.get("source")
                    target = rel.get("target")
                    relation_type = rel.get("type")
                    
                    if source and target and graph.has_node(source) and graph.has_node(target):
                        graph.add_edge(source, target, relation=relation_type, **rel)
                        
            except json.JSONDecodeError:
                logger.warning("关系提取结果解析失败")
                
        except Exception as e:
            logger.error(f"添加关系失败: {e}")
    
    async def calculate_complexity_score(self, content: str) -> float:
        """计算文本复杂度评分"""
        try:
            # 词汇复杂度
            words = list(jieba.cut(content))
            unique_words = len(set(words))
            total_words = len(words)
            vocab_complexity = unique_words / total_words if total_words > 0 else 0
            
            # 句子复杂度
            sentences = re.split(r'[。！？]', content)
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
            sentence_complexity = min(avg_sentence_length / 50, 1.0)  # 标准化到0-1
            
            # 语法复杂度（使用从句、修饰语等的频率）
            complex_patterns = [
                r'[，、；：]',  # 标点符号复杂度
                r'[的地得]',   # 修饰语
                r'虽然.*但是', # 转折句
                r'不仅.*而且', # 递进句
                r'如果.*就',   # 条件句
            ]
            
            grammar_complexity = 0
            for pattern in complex_patterns:
                matches = len(re.findall(pattern, content))
                grammar_complexity += matches / len(content) * 1000
            
            grammar_complexity = min(grammar_complexity, 1.0)
            
            # 综合复杂度
            overall_complexity = (vocab_complexity * 0.4 + sentence_complexity * 0.4 + grammar_complexity * 0.2)
            
            return round(overall_complexity, 3)
            
        except Exception as e:
            logger.error(f"计算复杂度失败: {e}")
            return 0.5
    
    async def cleanup(self) -> bool:
        """清理资源"""
        try:
            logger.info("清理增强版解析智能体资源...")
            
            # 清理各个分析模块
            await self.plot_analyzer.cleanup()
            await self.character_analyzer.cleanup()
            await self.style_analyzer.cleanup()
            await self.timeline_manager.cleanup()
            await self.emotion_analyzer.cleanup()
            
            # 清理缓存
            self.analysis_cache.clear()
            
            return True
        except Exception as e:
            logger.error(f"清理增强版解析智能体资源失败: {e}")
            return False


class PlotStructureAnalyzer:
    """情节结构分析器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def analyze_structure(self, content: str) -> Dict[str, Any]:
        """分析情节结构"""
        # 这里实现情节结构分析逻辑
        return {
            "three_act_structure": await self.analyze_three_acts(content),
            "dramatic_arc": await self.analyze_dramatic_arc(content),
            "plot_devices": await self.identify_plot_devices(content),
            "narrative_techniques": await self.analyze_narrative_techniques(content)
        }
    
    async def analyze_three_acts(self, content: str) -> Dict[str, Any]:
        """三幕结构分析"""
        # 实现三幕结构分析
        return {"act1": 0.25, "act2": 0.5, "act3": 0.25}
    
    async def analyze_dramatic_arc(self, content: str) -> Dict[str, Any]:
        """戏剧弧线分析"""
        # 实现戏剧弧线分析
        return {"exposition": 0.2, "rising_action": 0.4, "climax": 0.1, "falling_action": 0.2, "resolution": 0.1}
    
    async def identify_plot_devices(self, content: str) -> List[str]:
        """识别情节手法"""
        return ["foreshadowing", "flashback", "suspense"]
    
    async def analyze_narrative_techniques(self, content: str) -> List[str]:
        """分析叙事技巧"""
        return ["first_person", "past_tense", "linear_narrative"]
    
    async def cleanup(self):
        """清理资源"""
        pass


class CharacterNetworkAnalyzer:
    """角色网络分析器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def extract_characters(self, content: str) -> Dict[str, Any]:
        """提取角色网络"""
        return {
            "main_characters": [],
            "supporting_characters": [],
            "relationships": [],
            "character_development": {}
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class WritingStyleAnalyzer:
    """写作风格分析器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def analyze_style(self, content: str) -> Dict[str, Any]:
        """分析写作风格"""
        return {
            "tone": "neutral",
            "mood": "balanced",
            "pace": "moderate",
            "perspective": "third_person",
            "tense": "past",
            "style_complexity": 0.5
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class TimelineManager:
    """时间线管理器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def extract_timeline(self, content: str) -> List[Dict[str, Any]]:
        """提取时间线"""
        return []
    
    async def cleanup(self):
        """清理资源"""
        pass


class EmotionAnalyzer:
    """情感分析器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def analyze_emotions(self, content: str) -> Dict[str, Any]:
        """分析情感"""
        return {
            "dominant_emotion": "neutral",
            "emotion_intensity": 0.5,
            "emotion_progression": []
        }
    
    async def analyze_emotion_arc(self, content: str) -> Dict[str, Any]:
        """分析情感弧线"""
        return {
            "start_emotion": "neutral",
            "peak_emotion": "excitement",
            "end_emotion": "satisfaction",
            "emotional_tension": 0.6
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class IntelligentChapterSplitter:
    """智能章节分割器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def semantic_split(self, content: str) -> List[str]:
        """语义分割"""
        # 基于语义相似度的分割
        return [content]
    
    async def structural_split(self, content: str) -> List[str]:
        """结构分割"""
        # 基于文本结构的分割
        return [content]
    
    async def topic_split(self, content: str) -> List[str]:
        """主题分割"""
        # 基于主题变化的分割
        return [content]
    
    async def emotional_split(self, content: str) -> List[str]:
        """情感分割"""
        # 基于情感变化的分割
        return [content]
    
    async def fuse_splits(self, split_results: List[List[str]], content: str) -> List[str]:
        """融合多种分割结果"""
        # 融合多种分割算法的结果
        return [content]


class ContentUnderstanding:
    """内容理解模块"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def analyze_content(self, content: str) -> Dict[str, Any]:
        """分析内容"""
        return {
            "main_theme": "unknown",
            "sub_themes": [],
            "genre_elements": [],
            "narrative_structure": "linear",
            "content_quality": 0.7
        } 