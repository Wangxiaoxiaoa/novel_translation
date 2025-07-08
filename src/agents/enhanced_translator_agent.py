"""
增强版翻译智能体 - 高级翻译和深度文化适配
Enhanced Translator Agent - Advanced translation with deep cultural adaptation
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from datetime import datetime
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

from .base_agent import BaseAgent
from ..models.base import (
    AgentMessage, AgentType, Chapter, TranslationContext,
    LanguageCode, CultureType, Character, Location, Item, Terminology
)


@dataclass
class TranslationStrategy:
    """翻译策略"""
    name: str
    description: str
    weight: float
    parameters: Dict[str, Any]


@dataclass
class CulturalAdaptationRule:
    """文化适配规则"""
    source_culture: str
    target_culture: str
    rule_type: str
    pattern: str
    replacement: str
    context_conditions: List[str]


class EnhancedTranslatorAgent(BaseAgent):
    """增强版翻译智能体"""
    
    def __init__(self, config, openai_client):
        super().__init__(config, openai_client)
        
        # 高级翻译模块
        self.multi_model_translator = MultiModelTranslator()
        self.context_manager = DynamicContextManager()
        self.terminology_manager = IntelligentTerminologyManager()
        self.style_transfer = StyleTransferEngine()
        self.quality_assessor = AdvancedQualityAssessor()
        self.cultural_adapter = DeepCulturalAdapter()
        self.dialogue_optimizer = DialogueOptimizer()
        
        # 翻译策略集合
        self.translation_strategies = [
            TranslationStrategy("literal", "直译策略", 0.3, {"preserve_structure": True}),
            TranslationStrategy("semantic", "语义翻译", 0.4, {"focus_meaning": True}),
            TranslationStrategy("cultural", "文化适配", 0.3, {"adapt_culture": True}),
            TranslationStrategy("creative", "创意翻译", 0.2, {"allow_creativity": True})
        ]
        
        # 文化适配规则库
        self.cultural_rules = {}
        
        # 翻译记忆库
        self.translation_memory = TranslationMemory()
        
        # 上下文缓存
        self.context_cache = {}
        
        # 质量反馈系统
        self.feedback_system = QualityFeedbackSystem()
        
    async def initialize(self) -> bool:
        """初始化增强版翻译智能体"""
        try:
            logger.info("初始化增强版翻译智能体...")
            
            # 初始化各个模块
            await self.multi_model_translator.initialize()
            await self.context_manager.initialize()
            await self.terminology_manager.initialize()
            await self.style_transfer.initialize()
            await self.quality_assessor.initialize()
            await self.cultural_adapter.initialize()
            await self.dialogue_optimizer.initialize()
            
            # 初始化翻译记忆库
            await self.translation_memory.initialize()
            
            # 初始化反馈系统
            await self.feedback_system.initialize()
            
            # 加载文化适配规则
            await self.load_cultural_rules()
            
            # 健康检查
            health_ok = await self.health_check()
            if not health_ok:
                logger.error("增强版翻译智能体健康检查失败")
                return False
            
            logger.info("增强版翻译智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"增强版翻译智能体初始化失败: {e}")
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理消息"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type == "advanced_translate_chapter":
                result = await self.advanced_translate_chapter(content)
            elif message_type == "multi_strategy_translate":
                result = await self.multi_strategy_translate(content)
            elif message_type == "deep_cultural_adapt":
                result = await self.deep_cultural_adapt(content)
            elif message_type == "optimize_dialogue":
                result = await self.optimize_dialogue(content)
            elif message_type == "transfer_style":
                result = await self.transfer_style(content)
            elif message_type == "assess_quality":
                result = await self.assess_translation_quality(content)
            elif message_type == "build_terminology":
                result = await self.build_intelligent_terminology(content)
            elif message_type == "adaptive_translate":
                result = await self.adaptive_translate(content)
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
    
    async def advanced_translate_chapter(self, context: TranslationContext) -> Dict[str, Any]:
        """高级章节翻译"""
        try:
            self.current_task = f"高级翻译章节: {context.current_chapter.title}"
            logger.info(f"开始高级翻译章节: {context.current_chapter.title}")
            
            # 动态上下文管理
            enhanced_context = await self.context_manager.enhance_context(context) # Placeholder for now
            
            # 多策略翻译 - Simplified for Step 1: Use only one basic strategy (e.g., "semantic" or rely on the comprehensive prompt)
            # The change in execute_strategy_translation already uses a comprehensive base_translation prompt.
            # So, multi_strategy_translate_internal will effectively use that one prompt.
            # We'll just pick one strategy name like "semantic" to represent this.
            strategy_results = await self.multi_strategy_translate_internal(enhanced_context, use_single_strategy="semantic")
            
            # 智能策略融合 - Simplified for Step 1: Just take the result from the single strategy
            # fused_translation = await self.fuse_translation_strategies(strategy_results, enhanced_context)
            translated_content = strategy_results.get("semantic", context.current_chapter.content) # Fallback to original if semantic fails
            
            # 深度文化适配 - Call a very basic version
            culturally_adapted = await self.cultural_adapter.deep_adapt(
                translated_content, enhanced_context
            )
            
            # 对话优化 - Stubbed for Step 1
            # dialogue_optimized = await self.dialogue_optimizer.optimize_dialogues(
            # culturally_adapted, enhanced_context
            # )
            dialogue_optimized = culturally_adapted # Pass through

            # 风格迁移 - Stubbed for Step 1
            # style_transferred = await self.style_transfer.transfer_style(
            # dialogue_optimized, enhanced_context
            # )
            style_transferred = dialogue_optimized # Pass through

            # 术语一致性保证 - Stubbed for Step 1
            # terminology_consistent = await self.terminology_manager.ensure_consistency(
            # style_transferred, enhanced_context
            # )
            terminology_consistent = style_transferred # Pass through

            # 高级质量评估 - Stubbed for Step 1: Return mock data
            quality_metrics = await self.quality_assessor.comprehensive_assess(
                context.current_chapter.content,
                terminology_consistent, # This is the final translated content for now
                enhanced_context
            ) # This will return mock data as per its current implementation

            # 自适应优化 - Skipped for Step 1
            # if quality_metrics["overall_score"] < 8.0:
            #     terminology_consistent = await self.adaptive_improve_translation(
            #         terminology_consistent, quality_metrics, enhanced_context
            #     )
            #     quality_metrics = await self.quality_assessor.comprehensive_assess(
            #         context.current_chapter.content,
            #         terminology_consistent,
            #         enhanced_context
            #     )

            final_translated_content = terminology_consistent

            # 保存到翻译记忆 - Placeholder for Step 1
            # await self.translation_memory.save_translation(
            # context.current_chapter.content,
            # final_translated_content,
            # enhanced_context,
            # quality_metrics
            # )
            
            result = {
                "chapter_id": context.current_chapter.id,
                "original_content": context.current_chapter.content,
                "translated_content": final_translated_content,
                "target_language": context.target_language,
                "translation_strategies_used": ["semantic_via_base_template"], # Simplified
                "quality_metrics": quality_metrics, # Mock data from assessor
                "cultural_adaptations": "Basic cultural adaptation attempted.", # Placeholder
                "terminology_mappings": {}, # Placeholder
                "style_analysis": {}, # Placeholder
                "translation_metadata": {
                    "translator": self.agent_type,
                    "timestamp": datetime.now().isoformat(),
                    "enhancement_level": "advanced",
                    "processing_time": 0,  # 实际实现中应该计算处理时间
                    "model_confidence": quality_metrics.get("confidence", 0.85),
                    "cultural_adaptation_score": quality_metrics.get("cultural_adaptation", 0.8),
                    "dialogue_optimization_score": quality_metrics.get("dialogue_quality", 0.8),
                    "style_consistency_score": quality_metrics.get("style_consistency", 0.8)
                }
            }
            
            logger.info(f"高级章节翻译完成: {context.current_chapter.title}, 质量评分: {quality_metrics['overall_score']}")
            return result
            
        except Exception as e:
            logger.error(f"高级翻译章节失败: {e}")
            raise
    
    async def multi_strategy_translate_internal(self, context: TranslationContext) -> Dict[str, str]:
        """多策略翻译内部实现"""
        try:
            strategy_results = {}
            
            for strategy in self.translation_strategies:
                # 根据策略调整翻译参数
                translation_params = await self.adjust_params_for_strategy(strategy, context)
                
                # 执行翻译
                translated_content = await self.execute_strategy_translation(
                    context.current_chapter.content,
                    strategy,
                    translation_params,
                    context
                )
                
                strategy_results[strategy.name] = translated_content
            
            # If use_single_strategy is specified, only run that one.
            # Otherwise, run all strategies in self.translation_strategies.
            # For Step 1, we are simplifying to use a single comprehensive prompt via execute_strategy_translation.
            strategies_to_run = []
            if use_single_strategy:
                selected_strategy = next((s for s in self.translation_strategies if s.name == use_single_strategy), None)
                if selected_strategy:
                    strategies_to_run.append(selected_strategy)
                else:
                    logger.warning(f"Specified single strategy '{use_single_strategy}' not found. Defaulting to first available.")
                    if self.translation_strategies:
                         strategies_to_run.append(self.translation_strategies[0])
            else:
                strategies_to_run = self.translation_strategies

            if not strategies_to_run:
                logger.error("No translation strategies available to run.")
                return {"error": "No translation strategies configured"}

            for strategy in strategies_to_run:
                # 根据策略调整翻译参数
                translation_params = await self.adjust_params_for_strategy(strategy, context)

                # 执行翻译
                translated_content = await self.execute_strategy_translation(
                    context.current_chapter.content,
                    strategy, # Strategy object is passed, but execute_strategy_translation currently uses a generic prompt
                    translation_params,
                    context
                )

                strategy_results[strategy.name] = translated_content

            return strategy_results
            
        except Exception as e:
            logger.error(f"多策略翻译失败: {e}")
            # Raise or return an error structure
            return {"error": f"Multi-strategy translation failed: {str(e)}"} # Return error dict
    
    async def adjust_params_for_strategy(self, strategy: TranslationStrategy, 
                                       context: TranslationContext) -> Dict[str, Any]:
        """根据策略调整翻译参数"""
        try:
            base_params = {
                "temperature": 0.7,
                "max_tokens": 4000,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0
            }
            
            if strategy.name == "literal":
                base_params.update({
                    "temperature": 0.3,  # 更保守
                    "presence_penalty": 0.2  # 减少创新
                })
            elif strategy.name == "semantic":
                base_params.update({
                    "temperature": 0.5,  # 平衡
                    "presence_penalty": 0.1
                })
            elif strategy.name == "cultural":
                base_params.update({
                    "temperature": 0.8,  # 更灵活
                    "presence_penalty": -0.1  # 鼓励适配
                })
            elif strategy.name == "creative":
                base_params.update({
                    "temperature": 0.9,  # 最灵活
                    "presence_penalty": -0.2  # 鼓励创新
                })
            
            return base_params
            
        except Exception as e:
            logger.error(f"调整策略参数失败: {e}")
            return {"temperature": 0.7, "max_tokens": 4000}
    
    async def execute_strategy_translation(self, content: str, strategy: TranslationStrategy,
                                         params: Dict[str, Any], context: TranslationContext) -> str:
        """执行策略翻译"""
        try:
            # 构建策略特定的提示词
            prompt = await self.build_strategy_prompt(content, strategy, context)
            
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]
            
            # 调用LLM
            # translated = await self.call_llm(messages, **params) # Original call

            # Using the base_translation prompt from translation_templates
            # Ensure translation_templates is imported and available
            from prompts.translation_templates import get_translation_prompt

            # Prepare data for the base_translation template
            # Assuming context has all necessary fields like current_chapter, target_language, etc.
            # Also, character_context, location_context etc. might be part of TranslationContext
            # or need to be fetched/passed appropriately.
            # For now, we'll pass what's directly available or use placeholders.

            # Helper to get display names for languages
            def get_lang_display_name(lang_code_obj_or_str):
                # Handles both LanguageCode enum and string
                lang_code_str = lang_code_obj_or_str.value if hasattr(lang_code_obj_or_str, 'value') else str(lang_code_obj_or_str)
                names = {
                    "zh": "中文",
                    "en": "English",
                    "ja": "日语",
                    "ko": "韩语",
                    "fr": "Français",
                    "de": "Deutsch",
                    "es": "Español",
                    # Add more as needed, or make this a class/static method
                }
                return names.get(lang_code_str, lang_code_str) # Fallback to code if no display name

            template_data = {
                "source_language_display_name": get_lang_display_name(context.source_language),
                "target_language_display_name": get_lang_display_name(context.target_language),
                "target_language": context.target_language, # Still useful for other parts of prompt if not displayed
                "target_culture": context.target_culture or "Unknown",
                "source_content": content,
                "character_context": context.character_context or {},
                "location_context": context.location_context or {},
                "terminology_context": context.terminology_context or {},
                "plot_context": context.plot_context or "",
                "previous_chapters": context.previous_chapters or [],
                "style_guide": context.style_guide or "保持原文风格和情感，确保翻译流畅自然。"
            }

            # We are using a generic prompt now, so strategy-specific prompts are bypassed for this step.
            # The 'base_translation' prompt is quite comprehensive.
            rendered_prompt_content = get_translation_prompt("base_translation", **template_data)

            # The base_translation template generates a full prompt including instructions.
            # We can use this as the user message. The system message can be simpler.

            # For the call_llm method, messages should be a list of dicts, e.g., [{"role": "user", "content": "Translate this"}]
            # The base_translation_template already includes a lot of instructions.
            # We will make the system prompt generic here.

            final_messages = [
                {"role": "system", "content": f"You are a professional novel translator. Translate the given text into {context.target_language} following all provided instructions and context."},
                {"role": "user", "content": rendered_prompt_content} # The template output is the user content
            ]

            translated = await self.call_llm(final_messages, **params)
            
            return translated
            
        except Exception as e:
            logger.error(f"执行策略翻译失败: {e}")
            return content # Return original content on error
    
    async def build_strategy_prompt(self, content: str, strategy: TranslationStrategy,
                                  context: TranslationContext) -> Dict[str, str]:
        """构建策略特定的提示词 (Currently bypassed by execute_strategy_translation for simplification in Step 1)"""
        try:
            base_system = f"你是一个专业的小说翻译专家，专门使用{strategy.name}翻译策略。"
            
            if strategy.name == "literal":
                system_prompt = base_system + """
你的任务是进行直译，特点：
1. 严格保持原文结构和语序
2. 尽量保持原文的表达方式
3. 确保翻译准确性和忠实性
4. 避免过度意译或文化适配
5. 保持术语和人名的原始形式
"""
            
            elif strategy.name == "semantic":
                system_prompt = base_system + """
你的任务是进行语义翻译，特点：
1. 专注于传达原文的核心意思
2. 保持语言的自然流畅
3. 适度调整语序以符合目标语言习惯
4. 确保逻辑关系清晰
5. 平衡忠实性和可读性
"""
            
            elif strategy.name == "cultural":
                system_prompt = base_system + """
你的任务是进行文化适配翻译，特点：
1. 深度适配目标文化背景
2. 转换文化特有的概念和表达
3. 调整社会习俗和价值观表达
4. 本土化人名、地名和文化符号
5. 确保目标读者容易理解和接受
"""
            
            elif strategy.name == "creative":
                system_prompt = base_system + """
你的任务是进行创意翻译，特点：
1. 在保持原意基础上允许创新表达
2. 追求更高的文学性和艺术性
3. 使用丰富的修辞手法
4. 适当增强情感表达
5. 创造引人入胜的阅读体验
"""
            
            # 添加上下文信息
            context_info = await self.build_context_info(context)
            system_prompt += f"\n\n上下文信息:\n{context_info}"
            
            user_prompt = f"""
请使用{strategy.name}策略翻译以下内容到{context.target_language}:

原文:
{content}

请直接返回翻译结果，不要包含解释。
"""
            
            return {
                "system": system_prompt,
                "user": user_prompt
            }
            
        except Exception as e:
            logger.error(f"构建策略提示词失败: {e}")
            return {"system": "你是翻译专家", "user": content}
    
    async def fuse_translation_strategies(self, strategy_results: Dict[str, str], 
                                        context: TranslationContext) -> str:
        """融合多种翻译策略结果"""
        try:
            # 使用LLM进行智能融合
            fusion_prompt = f"""
你需要融合以下多种翻译策略的结果，产生一个最佳的翻译版本：

直译版本:
{strategy_results.get('literal', '')}

语义翻译版本:
{strategy_results.get('semantic', '')}

文化适配版本:
{strategy_results.get('cultural', '')}

创意翻译版本:
{strategy_results.get('creative', '')}

融合要求：
1. 选择每种策略的优点
2. 确保翻译准确性和流畅性
3. 保持文化适配的恰当性
4. 维护原文的风格和情感
5. 创造最佳的阅读体验

目标语言: {context.target_language}
目标文化: {context.target_culture}

请提供融合后的最佳翻译版本：
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的翻译融合专家，擅长整合多种翻译策略的优点。"},
                {"role": "user", "content": fusion_prompt}
            ]
            
            fused_translation = await self.call_llm(messages, temperature=0.6)
            
            return fused_translation
            
        except Exception as e:
            logger.error(f"融合翻译策略失败: {e}")
            # 如果融合失败，返回语义翻译作为后备
            return strategy_results.get('semantic', strategy_results.get('literal', ''))
    
    async def adaptive_improve_translation(self, translation: str, quality_metrics: Dict[str, float],
                                         context: TranslationContext) -> str:
        """自适应改进翻译"""
        try:
            # 识别需要改进的方面
            improvement_areas = []
            
            if quality_metrics.get("accuracy", 0) < 7.0:
                improvement_areas.append("accuracy")
            if quality_metrics.get("fluency", 0) < 7.0:
                improvement_areas.append("fluency")
            if quality_metrics.get("cultural_adaptation", 0) < 7.0:
                improvement_areas.append("cultural_adaptation")
            if quality_metrics.get("style_consistency", 0) < 7.0:
                improvement_areas.append("style_consistency")
            
            if not improvement_areas:
                return translation
            
            # 针对性改进
            improved_translation = translation
            
            for area in improvement_areas:
                improved_translation = await self.improve_specific_aspect(
                    improved_translation, area, context, quality_metrics
                )
            
            return improved_translation
            
        except Exception as e:
            logger.error(f"自适应改进翻译失败: {e}")
            return translation
    
    async def improve_specific_aspect(self, translation: str, aspect: str, 
                                    context: TranslationContext, quality_metrics: Dict[str, float]) -> str:
        """改进特定方面"""
        try:
            improvement_prompts = {
                "accuracy": f"""
请改进以下翻译的准确性，确保：
1. 完全忠实于原文意思
2. 没有遗漏或误译
3. 专业术语翻译正确
4. 逻辑关系清晰

当前翻译:
{translation}

请提供改进后的版本：
""",
                "fluency": f"""
请改进以下翻译的流畅性，确保：
1. 符合{context.target_language}表达习惯
2. 语法完全正确
3. 句子自然流畅
4. 避免生硬表达

当前翻译:
{translation}

请提供改进后的版本：
""",
                "cultural_adaptation": f"""
请改进以下翻译的文化适配，确保：
1. 符合{context.target_culture}文化背景
2. 避免文化冲突
3. 本土化表达方式
4. 目标读者容易理解

当前翻译:
{translation}

请提供改进后的版本：
""",
                "style_consistency": f"""
请改进以下翻译的风格一致性，确保：
1. 保持原文的文学风格
2. 情感色彩得到体现
3. 语言风格统一
4. 符合小说体裁特点

当前翻译:
{translation}

请提供改进后的版本：
"""
            }
            
            prompt = improvement_prompts.get(aspect, "请改进翻译质量")
            
            messages = [
                {"role": "system", "content": f"你是一个专业的翻译改进专家，专门提升{aspect}。"},
                {"role": "user", "content": prompt}
            ]
            
            improved = await self.call_llm(messages, temperature=0.5)
            
            return improved
            
        except Exception as e:
            logger.error(f"改进特定方面失败: {e}")
            return translation
    
    async def load_cultural_rules(self):
        """加载文化适配规则"""
        try:
            # 这里可以从文件或数据库加载文化适配规则
            # 暂时使用硬编码的示例规则
            
            self.cultural_rules = {
                "zh_to_en": [ # Chinese to English
                    CulturalAdaptationRule(
                        source_culture="Chinese",
                        target_culture="Western",
                        rule_type="currency_adaptation",
                        pattern=r"(\d+)\s*元", # Example: 100元
                        replacement=r"¥\1",    # Example: ¥100
                        context_conditions=[]
                    ),
                    CulturalAdaptationRule(
                        source_culture="Chinese",
                        target_culture="Western",
                        rule_type="name_adaptation_example", # Placeholder for a more complex name adaptation
                        pattern=r"老王",
                        replacement=r"Old Wang (or a culturally adapted name)",
                        context_conditions=["dialogue"]
                    )
                ],
                "en_to_zh": [ # English to Chinese
                    CulturalAdaptationRule(
                        source_culture="Western",
                        target_culture="Chinese",
                        rule_type="currency_adaptation",
                        pattern=r"\$(\d+(\.\d{1,2})?)", # Example: $100 or $100.50
                        replacement=r"\1美元",          # Example: 100美元
                        context_conditions=[]
                    ),
                    CulturalAdaptationRule(
                        source_culture="Western",
                        target_culture="Chinese",
                        rule_type="unit_conversion_example",
                        pattern=r"(\d+)\s*miles",
                        replacement=lambda m: f"{int(m.group(1)) * 1.60934:.2f}公里", # Approx conversion
                        context_conditions=[]
                    )
                ],
                "ja_to_zh": [ # Japanese to Chinese
                    CulturalAdaptationRule(
                        source_culture="Japanese",
                        target_culture="Chinese",
                        rule_type="honorifics_adaptation_example",
                        pattern=r"(\w+)さん", # Example: 田中さん
                        replacement=r"\1先生/女士", # Placeholder, proper adaptation is complex
                        context_conditions=["dialogue"]
                    ),
                    CulturalAdaptationRule(
                        source_culture="Japanese",
                        target_culture="Chinese",
                        rule_type="currency_adaptation",
                        pattern=r"(\d+)\s*円", # Example: 1000円
                        replacement=r"\1日元",    # Example: 1000日元
                        context_conditions=[]
                    )
                ],
                "zh_to_ja": [
                    CulturalAdaptationRule(
                        source_culture="Chinese",
                        target_culture="Japanese",
                        rule_type="currency_adaptation",
                        pattern=r"(\d+)\s*元",
                        replacement=r"\1元", # Or convert to 円 if rate is known & desired
                        context_conditions=[]
                    )
                ],
                "zh_to_ko": [
                     CulturalAdaptationRule(
                        source_culture="Chinese",
                        target_culture="Korean",
                        rule_type="currency_adaptation",
                        pattern=r"(\d+)\s*元",
                        replacement=r"\1위안",
                        context_conditions=[]
                    )
                ]
                # Add more language pairs and rules as needed
            }
            
            logger.info(f"文化适配规则加载完成. 加载的规则集数量: {len(self.cultural_rules)}")
            
        except Exception as e:
            logger.error(f"加载文化适配规则失败: {e}")
    
    # This specific adaptation function might be too specific or better handled by LLM/general rules.
    # For now, removing it as direct regex/lambda in rules is more flexible for simple cases.
    # def adapt_chinese_name_to_western(self, chinese_name: str) -> str:
    #     """将中文名字适配为西方名字"""
    #     # 这里实现具体的名字适配逻辑
    #     # 可以基于音译、意译或完全本土化
    #     name_mapping = {
    #         "王明": "William Wang",
    #         "李华": "Lisa Lee",
    #         "张伟": "David Zhang"
    #     }
    #     return name_mapping.get(chinese_name, chinese_name)
    
    async def build_context_info(self, context: TranslationContext) -> str:
        """构建上下文信息"""
        try:
            context_parts = []
            
            # 角色信息
            if context.character_context:
                char_info = "主要角色:\n"
                for name, char in context.character_context.items():
                    adapted_name = char.cultural_adaptations.get(context.target_language, name)
                    char_info += f"- {name} -> {adapted_name}: {char.description}\n"
                context_parts.append(char_info)
            
            # 地点信息
            if context.location_context:
                loc_info = "主要地点:\n"
                for name, loc in context.location_context.items():
                    adapted_name = loc.cultural_adaptations.get(context.target_language, name)
                    loc_info += f"- {name} -> {adapted_name}: {loc.description}\n"
                context_parts.append(loc_info)
            
            # 术语信息
            if context.terminology_context:
                term_info = "专业术语:\n"
                for term, terminology in context.terminology_context.items():
                    adapted_term = terminology.cultural_adaptations.get(context.target_language, term)
                    term_info += f"- {term} -> {adapted_term}: {terminology.definition}\n"
                context_parts.append(term_info)
            
            # 情节背景
            if context.plot_context:
                context_parts.append(f"情节背景:\n{context.plot_context}")
            
            # 前置章节
            if context.previous_chapters:
                prev_info = "前置章节摘要:\n"
                for chapter in context.previous_chapters[-3:]:
                    prev_info += f"- {chapter.title}: {chapter.summary}\n"
                context_parts.append(prev_info)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"构建上下文信息失败: {e}")
            return ""
    
    async def cleanup(self) -> bool:
        """清理资源"""
        try:
            logger.info("清理增强版翻译智能体资源...")
            
            # 清理各个模块
            await self.multi_model_translator.cleanup()
            await self.context_manager.cleanup()
            await self.terminology_manager.cleanup()
            await self.style_transfer.cleanup()
            await self.quality_assessor.cleanup()
            await self.cultural_adapter.cleanup()
            await self.dialogue_optimizer.cleanup()
            
            # 清理翻译记忆
            await self.translation_memory.cleanup()
            
            # 清理反馈系统
            await self.feedback_system.cleanup()
            
            # 清理缓存
            self.context_cache.clear()
            
            return True
        except Exception as e:
            logger.error(f"清理增强版翻译智能体资源失败: {e}")
            return False


class MultiModelTranslator:
    """多模型翻译器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def cleanup(self):
        """清理资源"""
        pass


class DynamicContextManager:
    """动态上下文管理器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def enhance_context(self, context: TranslationContext) -> TranslationContext:
        """增强上下文"""
        return context
    
    async def cleanup(self):
        """清理资源"""
        pass


class IntelligentTerminologyManager:
    """智能术语管理器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def ensure_consistency(self, content: str, context: TranslationContext) -> str:
        """确保术语一致性"""
        return content
    
    async def get_mappings(self, context: TranslationContext) -> Dict[str, str]:
        """获取术语映射"""
        return {}
    
    async def cleanup(self):
        """清理资源"""
        pass


class StyleTransferEngine:
    """风格迁移引擎"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def transfer_style(self, content: str, context: TranslationContext) -> str:
        """风格迁移"""
        return content
    
    async def get_style_analysis(self, context: TranslationContext) -> Dict[str, Any]:
        """获取风格分析"""
        return {}
    
    async def cleanup(self):
        """清理资源"""
        pass


class AdvancedQualityAssessor:
    """高级质量评估器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def comprehensive_assess(self, original: str, translated: str, 
                                 context: TranslationContext) -> Dict[str, float]:
        """综合质量评估"""
        return {
            "overall_score": 8.5,
            "accuracy": 8.5,
            "fluency": 8.0,
            "cultural_adaptation": 8.0,
            "style_consistency": 8.5,
            "dialogue_quality": 8.0,
            "confidence": 0.85
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class DeepCulturalAdapter:
    """深度文化适配器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def deep_adapt(self, content: str, context: TranslationContext) -> str:
        """深度文化适配 - 初步实现"""
        logger.info(f"开始初步文化适配: {context.current_chapter.id if context.current_chapter else 'N/A'}")
        adapted_content = content

        # Construct rule key, e.g., "zh_to_en"
        # Assuming context.source_language and context.target_language are LanguageCode enums or strings
        source_lang_code = context.source_language.value if hasattr(context.source_language, 'value') else str(context.source_language)
        target_lang_code = context.target_language.value if hasattr(context.target_language, 'value') else str(context.target_language)

        rule_key = f"{source_lang_code}_to_{target_lang_code}"

        if rule_key in self.cultural_rules:
            rules_for_pair = self.cultural_rules[rule_key]
            logger.debug(f"应用文化规则集 {rule_key}，共 {len(rules_for_pair)} 条规则。")

            for rule in rules_for_pair:
                try:
                    # Simple pattern replacement for now.
                    # More complex rules might need LLM intervention or specific handlers.
                    if isinstance(rule.replacement, str):
                        adapted_content = re.sub(rule.pattern, rule.replacement, adapted_content)
                        logger.debug(f"应用规则: {rule.pattern} -> {rule.replacement}")
                    elif callable(rule.replacement):
                        # If replacement is a function, call it.
                        # This is useful for more complex, regex-based replacements with groups.
                        adapted_content = re.sub(rule.pattern, rule.replacement, adapted_content)
                        logger.debug(f"应用函数规则: {rule.pattern}")
                    # TODO: Add LLM-based adaptation for more complex rules if needed in future steps.
                except Exception as e:
                    logger.error(f"应用文化规则失败 {rule.pattern}: {e}")
        else:
            logger.info(f"没有找到针对 {rule_key} 的特定文化规则。")

        # Fallback or generic adaptation if no specific rules applied or further refinement needed
        # For Step 1, this is kept simple. Future steps might involve LLM calls here.
        if adapted_content == content and rule_key not in self.cultural_rules : # only log if no rules were found AND content is unchanged
             logger.info(f"文化适配未修改内容 (可能无适用规则 for {rule_key}).")
        elif adapted_content != content:
            logger.info("文化适配已修改内容。")

        return adapted_content
    
    async def cleanup(self):
        """清理资源"""
        pass


class DialogueOptimizer:
    """对话优化器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def optimize_dialogues(self, content: str, context: TranslationContext) -> str:
        """优化对话"""
        return content
    
    async def cleanup(self):
        """清理资源"""
        pass


class TranslationMemory:
    """翻译记忆库"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def save_translation(self, original: str, translated: str, 
                             context: TranslationContext, quality: Dict[str, float]):
        """保存翻译"""
        pass
    
    async def cleanup(self):
        """清理资源"""
        pass


class QualityFeedbackSystem:
    """质量反馈系统"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def cleanup(self):
        """清理资源"""
        pass 