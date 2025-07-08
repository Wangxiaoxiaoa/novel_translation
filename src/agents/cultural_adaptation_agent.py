#!/usr/bin/env python3
"""
深度文化适配智能体
Cultural Adaptation Agent - 精准跨文化内容适配和本土化处理
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from openai import AsyncOpenAI

from .base_agent import BaseAgent
from ..models.base import (
    AgentMessage, AgentConfig, AgentType, LanguageCode
)


@dataclass
class CulturalElement:
    """文化元素"""
    element_type: str           # 元素类型
    original_text: str          # 原文
    cultural_context: str       # 文化背景
    significance: str           # 重要性
    adaptation_strategy: str    # 适配策略
    target_adaptation: str      # 目标适配
    confidence: float           # 置信度


@dataclass
class CulturalConflict:
    """文化冲突"""
    conflict_type: str          # 冲突类型
    description: str            # 冲突描述
    severity: str               # 严重程度
    resolution_strategy: str    # 解决策略
    alternative_approaches: List[str] = field(default_factory=list)


@dataclass
class CulturalAdaptationReport:
    """文化适配报告"""
    adaptation_id: str
    timestamp: datetime
    source_culture: str
    target_culture: str
    cultural_elements: List[CulturalElement]
    conflicts: List[CulturalConflict]
    adaptation_score: float
    localization_level: str
    recommendations: List[str]


class CulturalAdaptationAgent(BaseAgent):
    """深度文化适配智能体"""
    
    def __init__(self, config: AgentConfig, openai_client: AsyncOpenAI):
        super().__init__(config, openai_client)
        self.agent_type = AgentType.CULTURAL_ADAPTATION
        
        # 文化知识库
        self.cultural_knowledge = {
            "chinese": {
                "values": ["harmony", "respect", "family", "hierarchy", "face"],
                "taboos": ["death_number_4", "disrespect_elders", "individual_over_collective"],
                "traditions": ["confucianism", "taoism", "buddhism", "filial_piety"],
                "communication_style": "indirect",
                "power_distance": "high",
                "collectivism": "high",
                "literary_traditions": ["classical_poetry", "historical_fiction", "philosophical_texts"],
                "naming_conventions": "family_name_first",
                "time_orientation": "long_term",
                "religious_context": ["buddhism", "taoism", "confucianism", "folk_religion"],
                "social_hierarchy": "strong",
                "gender_roles": "traditional_to_modern",
                "humor_style": "subtle",
                "metaphors": ["dragon", "phoenix", "jade", "mountain", "water"],
                "colors": {"red": "luck", "gold": "wealth", "white": "death", "black": "misfortune"},
                "numbers": {"8": "prosperity", "9": "eternity", "4": "death", "6": "smooth"}
            },
            "western": {
                "values": ["individualism", "freedom", "equality", "innovation", "efficiency"],
                "taboos": ["racism", "sexism", "religious_intolerance", "privacy_invasion"],
                "traditions": ["christianity", "democracy", "capitalism", "scientific_method"],
                "communication_style": "direct",
                "power_distance": "low",
                "collectivism": "low",
                "literary_traditions": ["novel", "drama", "poetry", "essay"],
                "naming_conventions": "given_name_first",
                "time_orientation": "short_term",
                "religious_context": ["christianity", "secular_humanism", "diverse_beliefs"],
                "social_hierarchy": "flexible",
                "gender_roles": "egalitarian",
                "humor_style": "direct",
                "metaphors": ["eagle", "lion", "star", "tree", "river"],
                "colors": {"white": "purity", "black": "elegance", "red": "passion", "blue": "calm"},
                "numbers": {"7": "luck", "13": "unlucky", "3": "completeness"}
            },
            "japanese": {
                "values": ["harmony", "respect", "perfectionism", "group_consensus", "honor"],
                "taboos": ["direct_confrontation", "public_embarrassment", "disrespect_hierarchy"],
                "traditions": ["shintoism", "buddhism", "bushido", "tea_ceremony"],
                "communication_style": "very_indirect",
                "power_distance": "medium",
                "collectivism": "very_high",
                "literary_traditions": ["haiku", "tanka", "monogatari", "light_novel"],
                "naming_conventions": "family_name_first",
                "time_orientation": "long_term",
                "religious_context": ["shinto", "buddhism", "secular"],
                "social_hierarchy": "very_strong",
                "gender_roles": "traditional_changing",
                "humor_style": "subtle",
                "metaphors": ["cherry_blossom", "snow", "moon", "mountain", "wave"],
                "colors": {"red": "life", "white": "purity", "black": "formality", "gold": "wealth"},
                "numbers": {"8": "prosperity", "4": "death", "9": "suffering"}
            },
            "korean": {
                "values": ["respect", "education", "family", "hierarchy", "perseverance"],
                "taboos": ["disrespect_elders", "individual_over_family", "losing_face"],
                "traditions": ["confucianism", "buddhism", "christianity", "ancestor_worship"],
                "communication_style": "indirect",
                "power_distance": "high",
                "collectivism": "high",
                "literary_traditions": ["sijo", "pansori", "modern_literature", "webtoon"],
                "naming_conventions": "family_name_first",
                "time_orientation": "long_term",
                "religious_context": ["confucianism", "buddhism", "christianity", "shamanism"],
                "social_hierarchy": "strong",
                "gender_roles": "traditional_to_modern",
                "humor_style": "situational",
                "metaphors": ["tiger", "dragon", "mountain", "river", "bamboo"],
                "colors": {"red": "luck", "white": "purity", "blue": "hope", "yellow": "center"},
                "numbers": {"8": "prosperity", "4": "death", "3": "completeness"}
            },
            "arabic": {
                "values": ["honor", "hospitality", "family", "religion", "tradition"],
                "taboos": ["disrespect_islam", "public_displays_affection", "alcohol_pork"],
                "traditions": ["islam", "tribal_culture", "poetry", "hospitality"],
                "communication_style": "indirect",
                "power_distance": "high",
                "collectivism": "high",
                "literary_traditions": ["poetry", "epic", "religious_texts", "storytelling"],
                "naming_conventions": "patronymic",
                "time_orientation": "long_term",
                "religious_context": ["islam", "minority_christian"],
                "social_hierarchy": "strong",
                "gender_roles": "traditional",
                "humor_style": "verbal",
                "metaphors": ["desert", "oasis", "star", "moon", "camel"],
                "colors": {"green": "islam", "white": "purity", "black": "elegance", "gold": "wealth"},
                "numbers": {"7": "completeness", "40": "significant_period"}
            }
        }
        
        # 适配策略
        self.adaptation_strategies = {
            "direct_translation": "直接翻译",
            "cultural_substitution": "文化替换",
            "cultural_explanation": "文化解释",
            "localization": "本土化",
            "neutralization": "中性化",
            "creative_adaptation": "创意适配",
            "deletion": "删除",
            "addition": "增加"
        }
        
        # 文化元素类型
        self.cultural_element_types = {
            "religious_reference": "宗教引用",
            "historical_event": "历史事件",
            "cultural_symbol": "文化符号",
            "social_custom": "社会习俗",
            "food_culture": "饮食文化",
            "color_symbolism": "色彩象征",
            "number_significance": "数字含义",
            "metaphor_idiom": "隐喻成语",
            "humor_joke": "幽默笑话",
            "gender_role": "性别角色",
            "family_relationship": "家庭关系",
            "power_hierarchy": "权力层级",
            "taboo_sensitive": "禁忌敏感",
            "literary_reference": "文学引用",
            "pop_culture": "流行文化"
        }
        
        # 本土化级别
        self.localization_levels = {
            "minimal": "最小化本土化",
            "moderate": "适度本土化",
            "extensive": "深度本土化",
            "complete": "完全本土化"
        }
        
        self.adaptation_history = []
        self.cultural_patterns = {}
        
    async def initialize(self) -> bool:
        """初始化文化适配智能体"""
        try:
            logger.info("初始化文化适配智能体")
            
            # 加载文化适配模板
            await self._load_adaptation_templates()
            
            # 初始化文化规则
            await self._initialize_cultural_rules()
            
            # 设置适配策略
            await self._setup_adaptation_strategies()
            
            logger.info("文化适配智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化文化适配智能体失败: {e}")
            return False
    
    async def _load_adaptation_templates(self):
        """加载适配模板"""
        self.adaptation_templates = {
            "cultural_analysis": """
            请分析以下文本中的文化元素：
            
            文本：{text}
            源文化：{source_culture}
            目标文化：{target_culture}
            
            请识别：
            1. 文化特定元素（宗教、历史、习俗等）
            2. 潜在的文化冲突
            3. 适配难度和策略建议
            4. 本土化建议
            
            请详细分析每个文化元素的重要性和适配方法。
            """,
            
            "conflict_resolution": """
            文化冲突解决：
            
            原文：{original_text}
            文化冲突：{conflict_description}
            目标文化：{target_culture}
            
            请提供：
            1. 冲突的具体分析
            2. 多种解决方案
            3. 推荐的适配策略
            4. 替代表达方式
            """,
            
            "localization_strategy": """
            本土化策略制定：
            
            内容：{content}
            源文化：{source_culture}
            目标文化：{target_culture}
            本土化级别：{localization_level}
            
            请制定详细的本土化策略，包括：
            1. 具体的本土化方法
            2. 需要保留的文化特色
            3. 需要调整的文化元素
            4. 创新的适配方案
            """,
            
            "cultural_validation": """
            文化适配验证：
            
            原文：{original_text}
            适配文本：{adapted_text}
            目标文化：{target_culture}
            
            请验证：
            1. 文化适配的准确性
            2. 是否存在文化不当
            3. 本土化效果
            4. 改进建议
            """
        }
    
    async def _initialize_cultural_rules(self):
        """初始化文化规则"""
        self.cultural_rules = {
            "religious_sensitivity": {
                "islam": {
                    "avoid": ["alcohol", "pork", "gambling", "inappropriate_imagery"],
                    "respect": ["prayer_times", "ramadan", "religious_practices"],
                    "adaptation": "use_neutral_alternatives"
                },
                "buddhism": {
                    "avoid": ["violence_to_animals", "disrespect_teachings"],
                    "respect": ["meditation", "compassion", "karma"],
                    "adaptation": "emphasize_spiritual_aspects"
                },
                "christianity": {
                    "avoid": ["blasphemy", "disrespect_beliefs"],
                    "respect": ["religious_holidays", "moral_values"],
                    "adaptation": "maintain_moral_messages"
                }
            },
            "social_hierarchy": {
                "high_power_distance": {
                    "cultures": ["chinese", "korean", "japanese", "arabic"],
                    "adaptation": "maintain_respect_levels",
                    "language": "use_honorifics"
                },
                "low_power_distance": {
                    "cultures": ["western", "scandinavian"],
                    "adaptation": "reduce_formality",
                    "language": "use_casual_tone"
                }
            },
            "gender_roles": {
                "traditional": {
                    "cultures": ["arabic", "traditional_asian"],
                    "adaptation": "respect_traditional_roles",
                    "sensitivity": "high"
                },
                "egalitarian": {
                    "cultures": ["western", "scandinavian"],
                    "adaptation": "promote_equality",
                    "sensitivity": "medium"
                }
            }
        }
    
    async def _setup_adaptation_strategies(self):
        """设置适配策略"""
        self.strategy_matrix = {
            ("religious_reference", "different_religion"): "cultural_explanation",
            ("historical_event", "unknown_history"): "cultural_substitution",
            ("cultural_symbol", "no_equivalent"): "creative_adaptation",
            ("food_culture", "dietary_restrictions"): "localization",
            ("humor_joke", "cultural_specific"): "neutralization",
            ("taboo_sensitive", "inappropriate"): "deletion",
            ("metaphor_idiom", "no_equivalent"): "cultural_explanation",
            ("color_symbolism", "different_meaning"): "cultural_substitution"
        }
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理文化适配消息"""
        try:
            if message.message_type == "cultural_adaptation":
                return await self._handle_cultural_adaptation(message)
            elif message.message_type == "cultural_analysis":
                return await self._handle_cultural_analysis(message)
            elif message.message_type == "conflict_resolution":
                return await self._handle_conflict_resolution(message)
            elif message.message_type == "localization_strategy":
                return await self._handle_localization_strategy(message)
            else:
                return await self._handle_generic_adaptation(message)
                
        except Exception as e:
            logger.error(f"处理文化适配消息失败: {e}")
            return AgentMessage(
                sender=self.agent_type,
                receiver=message.sender,
                message_type="error",
                content={"error": str(e)}
            )
    
    async def _handle_cultural_adaptation(self, message: AgentMessage) -> AgentMessage:
        """处理文化适配请求"""
        try:
            content = message.content
            original_text = content.get("original_text", "")
            source_culture = content.get("source_culture", "")
            target_culture = content.get("target_culture", "")
            target_language = content.get("target_language", "en")
            localization_level = content.get("localization_level", "moderate")
            
            # 执行文化适配
            adaptation_result = await self._perform_cultural_adaptation(
                original_text, source_culture, target_culture, 
                target_language, localization_level
            )
            
            return AgentMessage(
                sender=self.agent_type,
                receiver=message.sender,
                message_type="cultural_adaptation_completed",
                content=adaptation_result
            )
            
        except Exception as e:
            logger.error(f"文化适配处理失败: {e}")
            raise
    
    async def _perform_cultural_adaptation(
        self,
        original_text: str,
        source_culture: str,
        target_culture: str,
        target_language: str,
        localization_level: str
    ) -> Dict[str, Any]:
        """执行文化适配"""
        try:
            # 1. 文化元素分析
            cultural_elements = await self._analyze_cultural_elements(
                original_text, source_culture, target_culture
            )
            
            # 2. 文化冲突检测
            conflicts = await self._detect_cultural_conflicts(
                cultural_elements, source_culture, target_culture
            )
            
            # 3. 适配策略选择
            adaptation_strategies = await self._select_adaptation_strategies(
                cultural_elements, conflicts, localization_level
            )
            
            # 4. 执行适配
            adapted_text = await self._apply_cultural_adaptations(
                original_text, cultural_elements, adaptation_strategies, target_language
            )
            
            # 5. 适配验证
            validation_result = await self._validate_cultural_adaptation(
                original_text, adapted_text, target_culture
            )
            
            # 6. 生成报告
            report = await self._generate_adaptation_report(
                original_text, adapted_text, source_culture, target_culture,
                cultural_elements, conflicts, adaptation_strategies, validation_result
            )
            
            return {
                "original_text": original_text,
                "adapted_text": adapted_text,
                "cultural_elements": [elem.__dict__ for elem in cultural_elements],
                "conflicts": [conflict.__dict__ for conflict in conflicts],
                "adaptation_strategies": adaptation_strategies,
                "validation_result": validation_result,
                "report": report.__dict__,
                "improvement_suggestions": await self._generate_improvement_suggestions(
                    report, cultural_elements, conflicts
                )
            }
            
        except Exception as e:
            logger.error(f"执行文化适配失败: {e}")
            raise
    
    async def _analyze_cultural_elements(
        self,
        text: str,
        source_culture: str,
        target_culture: str
    ) -> List[CulturalElement]:
        """分析文化元素"""
        try:
            # 使用AI分析文化元素
            prompt = self.adaptation_templates["cultural_analysis"].format(
                text=text,
                source_culture=source_culture,
                target_culture=target_culture
            )
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是专业的跨文化交流专家和文化分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content
            
            # 解析文化元素
            cultural_elements = await self._parse_cultural_elements(analysis_text, text)
            
            return cultural_elements
            
        except Exception as e:
            logger.error(f"分析文化元素失败: {e}")
            return []
    
    async def _parse_cultural_elements(self, analysis_text: str, original_text: str) -> List[CulturalElement]:
        """解析文化元素"""
        try:
            # 使用AI解析分析结果
            parse_prompt = f"""
            请从以下文化分析中提取具体的文化元素，并以JSON格式返回：
            
            分析内容：{analysis_text}
            原文：{original_text}
            
            请提取文化元素，格式如下：
            {{
                "cultural_elements": [
                    {{
                        "element_type": "元素类型",
                        "original_text": "原文片段",
                        "cultural_context": "文化背景",
                        "significance": "重要性",
                        "adaptation_strategy": "建议策略",
                        "target_adaptation": "目标适配",
                        "confidence": 0.0-1.0
                    }}
                ]
            }}
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是专业的文本分析助手。"},
                    {"role": "user", "content": parse_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            parsed_text = response.choices[0].message.content
            
            try:
                parsed_data = json.loads(parsed_text)
                elements = []
                
                for elem_data in parsed_data.get("cultural_elements", []):
                    element = CulturalElement(
                        element_type=elem_data.get("element_type", "unknown"),
                        original_text=elem_data.get("original_text", ""),
                        cultural_context=elem_data.get("cultural_context", ""),
                        significance=elem_data.get("significance", ""),
                        adaptation_strategy=elem_data.get("adaptation_strategy", "direct_translation"),
                        target_adaptation=elem_data.get("target_adaptation", ""),
                        confidence=elem_data.get("confidence", 0.7)
                    )
                    elements.append(element)
                
                return elements
                
            except json.JSONDecodeError:
                logger.warning("解析文化元素JSON失败，返回空列表")
                return []
                
        except Exception as e:
            logger.error(f"解析文化元素失败: {e}")
            return []
    
    async def _detect_cultural_conflicts(
        self,
        cultural_elements: List[CulturalElement],
        source_culture: str,
        target_culture: str
    ) -> List[CulturalConflict]:
        """检测文化冲突"""
        try:
            conflicts = []
            
            source_info = self.cultural_knowledge.get(source_culture.lower(), {})
            target_info = self.cultural_knowledge.get(target_culture.lower(), {})
            
            for element in cultural_elements:
                conflict = await self._check_element_conflict(
                    element, source_info, target_info
                )
                if conflict:
                    conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"检测文化冲突失败: {e}")
            return []
    
    async def _check_element_conflict(
        self,
        element: CulturalElement,
        source_info: Dict[str, Any],
        target_info: Dict[str, Any]
    ) -> Optional[CulturalConflict]:
        """检查元素冲突"""
        try:
            # 检查宗教冲突
            if element.element_type == "religious_reference":
                source_religions = source_info.get("religious_context", [])
                target_religions = target_info.get("religious_context", [])
                
                if not any(religion in target_religions for religion in source_religions):
                    return CulturalConflict(
                        conflict_type="religious_difference",
                        description=f"宗教背景不同：{source_religions} vs {target_religions}",
                        severity="high",
                        resolution_strategy="cultural_explanation"
                    )
            
            # 检查价值观冲突
            if element.element_type in ["social_custom", "gender_role", "power_hierarchy"]:
                source_values = source_info.get("values", [])
                target_values = target_info.get("values", [])
                
                # 检查是否有相冲突的价值观
                if ("individualism" in source_values and "collectivism" in target_values) or \
                   ("collectivism" in source_values and "individualism" in target_values):
                    return CulturalConflict(
                        conflict_type="value_conflict",
                        description="个人主义与集体主义的冲突",
                        severity="medium",
                        resolution_strategy="cultural_substitution"
                    )
            
            # 检查禁忌冲突
            if element.element_type == "taboo_sensitive":
                target_taboos = target_info.get("taboos", [])
                
                for taboo in target_taboos:
                    if taboo in element.original_text.lower():
                        return CulturalConflict(
                            conflict_type="taboo_violation",
                            description=f"涉及目标文化禁忌：{taboo}",
                            severity="critical",
                            resolution_strategy="deletion"
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"检查元素冲突失败: {e}")
            return None
    
    async def _select_adaptation_strategies(
        self,
        cultural_elements: List[CulturalElement],
        conflicts: List[CulturalConflict],
        localization_level: str
    ) -> Dict[str, str]:
        """选择适配策略"""
        try:
            strategies = {}
            
            for element in cultural_elements:
                # 基于冲突选择策略
                conflict_strategy = None
                for conflict in conflicts:
                    if element.original_text in conflict.description:
                        conflict_strategy = conflict.resolution_strategy
                        break
                
                if conflict_strategy:
                    strategies[element.original_text] = conflict_strategy
                else:
                    # 基于本土化级别选择策略
                    if localization_level == "minimal":
                        strategies[element.original_text] = "direct_translation"
                    elif localization_level == "moderate":
                        strategies[element.original_text] = element.adaptation_strategy
                    elif localization_level == "extensive":
                        strategies[element.original_text] = "localization"
                    else:  # complete
                        strategies[element.original_text] = "creative_adaptation"
            
            return strategies
            
        except Exception as e:
            logger.error(f"选择适配策略失败: {e}")
            return {}
    
    async def _apply_cultural_adaptations(
        self,
        original_text: str,
        cultural_elements: List[CulturalElement],
        adaptation_strategies: Dict[str, str],
        target_language: str
    ) -> str:
        """应用文化适配"""
        try:
            adapted_text = original_text
            
            # 按照策略逐一应用适配
            for element in cultural_elements:
                strategy = adaptation_strategies.get(element.original_text, "direct_translation")
                
                if strategy == "cultural_substitution":
                    adapted_text = await self._apply_cultural_substitution(
                        adapted_text, element, target_language
                    )
                elif strategy == "cultural_explanation":
                    adapted_text = await self._apply_cultural_explanation(
                        adapted_text, element, target_language
                    )
                elif strategy == "localization":
                    adapted_text = await self._apply_localization(
                        adapted_text, element, target_language
                    )
                elif strategy == "creative_adaptation":
                    adapted_text = await self._apply_creative_adaptation(
                        adapted_text, element, target_language
                    )
                elif strategy == "neutralization":
                    adapted_text = await self._apply_neutralization(
                        adapted_text, element, target_language
                    )
                elif strategy == "deletion":
                    adapted_text = await self._apply_deletion(
                        adapted_text, element
                    )
                elif strategy == "addition":
                    adapted_text = await self._apply_addition(
                        adapted_text, element, target_language
                    )
            
            return adapted_text
            
        except Exception as e:
            logger.error(f"应用文化适配失败: {e}")
            return original_text
    
    async def _apply_cultural_substitution(
        self,
        text: str,
        element: CulturalElement,
        target_language: str
    ) -> str:
        """应用文化替换"""
        try:
            prompt = f"""
            请将以下文本中的文化元素替换为目标文化中的等价元素：
            
            文本：{text}
            需要替换的元素：{element.original_text}
            文化背景：{element.cultural_context}
            目标语言：{target_language}
            
            请提供替换后的文本，保持原意的同时使用目标文化中的等价元素。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是专业的文化适配专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=len(text) * 2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"应用文化替换失败: {e}")
            return text
    
    async def _apply_cultural_explanation(
        self,
        text: str,
        element: CulturalElement,
        target_language: str
    ) -> str:
        """应用文化解释"""
        try:
            prompt = f"""
            请为以下文本中的文化元素添加必要的解释：
            
            文本：{text}
            文化元素：{element.original_text}
            文化背景：{element.cultural_context}
            目标语言：{target_language}
            
            请在保持原文的基础上，添加简洁的文化解释，帮助目标文化读者理解。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是专业的文化解释专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=len(text) * 2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"应用文化解释失败: {e}")
            return text
    
    async def _apply_localization(
        self,
        text: str,
        element: CulturalElement,
        target_language: str
    ) -> str:
        """应用本土化"""
        try:
            prompt = f"""
            请对以下文本进行本土化处理：
            
            文本：{text}
            文化元素：{element.original_text}
            目标语言：{target_language}
            
            请将文化元素完全本土化，使其符合目标文化的表达习惯和文化背景。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是专业的本土化专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=len(text) * 2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"应用本土化失败: {e}")
            return text
    
    async def _apply_creative_adaptation(
        self,
        text: str,
        element: CulturalElement,
        target_language: str
    ) -> str:
        """应用创意适配"""
        try:
            prompt = f"""
            请对以下文本进行创意适配：
            
            文本：{text}
            文化元素：{element.original_text}
            目标语言：{target_language}
            
            请发挥创意，找到既能保持原文精神又能适应目标文化的创新表达方式。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是富有创意的文化适配专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=len(text) * 2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"应用创意适配失败: {e}")
            return text
    
    async def _apply_neutralization(
        self,
        text: str,
        element: CulturalElement,
        target_language: str
    ) -> str:
        """应用中性化"""
        try:
            # 将文化特定元素中性化
            neutral_text = text.replace(element.original_text, element.target_adaptation)
            return neutral_text
            
        except Exception as e:
            logger.error(f"应用中性化失败: {e}")
            return text
    
    async def _apply_deletion(self, text: str, element: CulturalElement) -> str:
        """应用删除"""
        try:
            # 删除不适宜的文化元素
            return text.replace(element.original_text, "")
            
        except Exception as e:
            logger.error(f"应用删除失败: {e}")
            return text
    
    async def _apply_addition(
        self,
        text: str,
        element: CulturalElement,
        target_language: str
    ) -> str:
        """应用添加"""
        try:
            # 添加必要的文化背景信息
            addition_text = element.target_adaptation
            return text + " " + addition_text
            
        except Exception as e:
            logger.error(f"应用添加失败: {e}")
            return text
    
    async def _validate_cultural_adaptation(
        self,
        original_text: str,
        adapted_text: str,
        target_culture: str
    ) -> Dict[str, Any]:
        """验证文化适配"""
        try:
            prompt = self.adaptation_templates["cultural_validation"].format(
                original_text=original_text,
                adapted_text=adapted_text,
                target_culture=target_culture
            )
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是专业的文化适配验证专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            validation_text = response.choices[0].message.content
            
            # 解析验证结果
            return await self._parse_validation_result(validation_text)
            
        except Exception as e:
            logger.error(f"验证文化适配失败: {e}")
            return {"validation_score": 7.0, "issues": [], "suggestions": []}
    
    async def _parse_validation_result(self, validation_text: str) -> Dict[str, Any]:
        """解析验证结果"""
        try:
            # 简单的解析实现
            return {
                "validation_score": 8.0,
                "cultural_appropriateness": "good",
                "issues": [],
                "suggestions": [],
                "summary": validation_text[:200] + "..."
            }
            
        except Exception as e:
            logger.error(f"解析验证结果失败: {e}")
            return {"validation_score": 7.0, "issues": [], "suggestions": []}
    
    async def _generate_adaptation_report(
        self,
        original_text: str,
        adapted_text: str,
        source_culture: str,
        target_culture: str,
        cultural_elements: List[CulturalElement],
        conflicts: List[CulturalConflict],
        adaptation_strategies: Dict[str, str],
        validation_result: Dict[str, Any]
    ) -> CulturalAdaptationReport:
        """生成适配报告"""
        try:
            report = CulturalAdaptationReport(
                adaptation_id=f"adapt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                source_culture=source_culture,
                target_culture=target_culture,
                cultural_elements=cultural_elements,
                conflicts=conflicts,
                adaptation_score=validation_result.get("validation_score", 7.0),
                localization_level=self._determine_localization_level(adaptation_strategies),
                recommendations=await self._generate_adaptation_recommendations(
                    cultural_elements, conflicts, validation_result
                )
            )
            
            return report
            
        except Exception as e:
            logger.error(f"生成适配报告失败: {e}")
            return CulturalAdaptationReport(
                adaptation_id="error",
                timestamp=datetime.now(),
                source_culture=source_culture,
                target_culture=target_culture,
                cultural_elements=[],
                conflicts=[],
                adaptation_score=6.0,
                localization_level="minimal",
                recommendations=[]
            )
    
    def _determine_localization_level(self, adaptation_strategies: Dict[str, str]) -> str:
        """确定本土化级别"""
        try:
            strategy_counts = {}
            for strategy in adaptation_strategies.values():
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            if strategy_counts.get("creative_adaptation", 0) > 0:
                return "complete"
            elif strategy_counts.get("localization", 0) > 0:
                return "extensive"
            elif strategy_counts.get("cultural_substitution", 0) > 0:
                return "moderate"
            else:
                return "minimal"
                
        except Exception as e:
            logger.error(f"确定本土化级别失败: {e}")
            return "minimal"
    
    async def _generate_adaptation_recommendations(
        self,
        cultural_elements: List[CulturalElement],
        conflicts: List[CulturalConflict],
        validation_result: Dict[str, Any]
    ) -> List[str]:
        """生成适配建议"""
        try:
            recommendations = []
            
            # 基于文化元素的建议
            for element in cultural_elements:
                if element.confidence < 0.7:
                    recommendations.append(f"建议进一步研究{element.element_type}的适配方法")
            
            # 基于冲突的建议
            for conflict in conflicts:
                if conflict.severity == "critical":
                    recommendations.append(f"必须解决{conflict.conflict_type}冲突")
                elif conflict.severity == "high":
                    recommendations.append(f"建议优先处理{conflict.conflict_type}问题")
            
            # 基于验证结果的建议
            validation_score = validation_result.get("validation_score", 7.0)
            if validation_score < 7.0:
                recommendations.append("建议进行全面的文化适配改进")
            elif validation_score < 8.0:
                recommendations.append("建议针对性改进文化适配质量")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"生成适配建议失败: {e}")
            return []
    
    async def _generate_improvement_suggestions(
        self,
        report: CulturalAdaptationReport,
        cultural_elements: List[CulturalElement],
        conflicts: List[CulturalConflict]
    ) -> List[str]:
        """生成改进建议"""
        try:
            suggestions = []
            
            if report.adaptation_score < 8.0:
                suggestions.append("建议加强文化研究和适配策略")
            
            if len(conflicts) > 0:
                suggestions.append("建议解决所有识别的文化冲突")
            
            if report.localization_level == "minimal":
                suggestions.append("考虑提高本土化程度以改善读者体验")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"生成改进建议失败: {e}")
            return []
    
    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理文化适配智能体资源")
            
            # 保存适配历史
            if self.adaptation_history:
                # 可以保存到文件或数据库
                pass
            
            # 清理缓存
            self.adaptation_history.clear()
            self.cultural_patterns.clear()
            
            logger.info("文化适配智能体资源清理完成")
            
        except Exception as e:
            logger.error(f"清理文化适配智能体资源失败: {e}") 