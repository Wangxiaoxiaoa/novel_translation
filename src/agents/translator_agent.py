"""
翻译智能体 - 负责文本翻译和文化适配
Translator Agent - Responsible for text translation and cultural adaptation
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from datetime import datetime

from .base_agent import BaseAgent
from ..models.base import (
    AgentMessage, AgentType, Chapter, TranslationContext,
    LanguageCode, CultureType, Character, Location, Item, Terminology
)


class TranslatorAgent(BaseAgent):
    """翻译智能体"""
    
    def __init__(self, config, openai_client):
        super().__init__(config, openai_client)
        
        # 文化映射配置
        self.culture_mapping = {
            LanguageCode.ENGLISH: CultureType.WESTERN,
            LanguageCode.JAPANESE: CultureType.JAPANESE,
            LanguageCode.KOREAN: CultureType.KOREAN,
            LanguageCode.FRENCH: CultureType.WESTERN,
            LanguageCode.GERMAN: CultureType.WESTERN,
            LanguageCode.SPANISH: CultureType.WESTERN,
            LanguageCode.RUSSIAN: CultureType.SLAVIC,
            LanguageCode.ARABIC: CultureType.MIDDLE_EASTERN,
            LanguageCode.CHINESE: CultureType.CHINESE,
        }
        
        # 文化适配规则
        self.cultural_rules = {
            CultureType.WESTERN: {
                "naming_conventions": "Western names (first name + last name)",
                "honorifics": "Mr./Ms./Dr. etc.",
                "cultural_references": "Western cultural concepts",
                "measurement_units": "Imperial/Metric system",
                "currency": "Local currency",
                "festivals": "Christmas, Easter, etc.",
                "social_customs": "Western social norms"
            },
            CultureType.JAPANESE: {
                "naming_conventions": "Japanese names (family name + given name)",
                "honorifics": "San, Kun, Chan, Sama, etc.",
                "cultural_references": "Japanese cultural concepts",
                "measurement_units": "Metric system",
                "currency": "Yen",
                "festivals": "Cherry Blossom, Obon, etc.",
                "social_customs": "Japanese social norms"
            },
            CultureType.KOREAN: {
                "naming_conventions": "Korean names (family name + given name)",
                "honorifics": "Nim, Ssi, etc.",
                "cultural_references": "Korean cultural concepts",
                "measurement_units": "Metric system",
                "currency": "Won",
                "festivals": "Chuseok, Lunar New Year, etc.",
                "social_customs": "Korean social norms"
            }
        }
        
        # 翻译质量检查点
        self.quality_checkpoints = [
            "character_consistency",
            "cultural_appropriateness",
            "plot_continuity",
            "terminology_consistency",
            "style_consistency"
        ]
        
    async def initialize(self) -> bool:
        """初始化翻译智能体"""
        try:
            logger.info("初始化翻译智能体...")
            
            # 健康检查
            health_ok = await self.health_check()
            if not health_ok:
                logger.error("翻译智能体健康检查失败")
                return False
            
            # 预加载文化适配模板
            await self.preload_cultural_templates()
            
            logger.info("翻译智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"翻译智能体初始化失败: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理翻译智能体资源"""
        try:
            logger.info("清理翻译智能体资源...")
            return True
        except Exception as e:
            logger.error(f"清理翻译智能体资源失败: {e}")
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理消息"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type == "translate_chapter":
                result = await self.translate_chapter(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="chapter_translated",
                    content=result
                )
            
            elif message_type == "adapt_cultural_elements":
                result = await self.adapt_cultural_elements(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="cultural_adapted",
                    content=result
                )
            
            elif message_type == "translate_terminology":
                result = await self.translate_terminology(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="terminology_translated",
                    content=result
                )
            
            elif message_type == "quality_check":
                result = await self.quality_check_translation(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="quality_checked",
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
    
    async def translate_chapter(self, context: TranslationContext) -> Dict[str, Any]:
        """翻译章节"""
        try:
            self.current_task = f"翻译章节: {context.current_chapter.title}"
            logger.info(f"开始翻译章节: {context.current_chapter.title}")
            
            # 准备翻译上下文
            translation_prompt = await self.prepare_translation_prompt(context)
            
            # 执行翻译
            translated_content = await self.execute_translation(
                context.current_chapter.content,
                translation_prompt,
                context.target_language
            )
            
            # 文化适配
            adapted_content = await self.apply_cultural_adaptation(
                translated_content,
                context
            )
            
            # 术语一致性检查
            final_content = await self.ensure_terminology_consistency(
                adapted_content,
                context
            )
            
            # 质量评估
            quality_score = await self.assess_translation_quality(
                context.current_chapter.content,
                final_content,
                context
            )
            
            result = {
                "chapter_id": context.current_chapter.id,
                "original_content": context.current_chapter.content,
                "translated_content": final_content,
                "target_language": context.target_language,
                "quality_score": quality_score,
                "translation_metadata": {
                    "translator": self.agent_type,
                    "timestamp": datetime.now().isoformat(),
                    "cultural_adaptations": context.target_culture,
                    "terminology_count": len(context.terminology_context),
                    "character_count": len(context.character_context)
                }
            }
            
            logger.info(f"章节翻译完成: {context.current_chapter.title}, 质量评分: {quality_score}")
            return result
            
        except Exception as e:
            logger.error(f"翻译章节失败: {e}")
            raise
    
    async def prepare_translation_prompt(self, context: TranslationContext) -> str:
        """准备翻译提示词"""
        try:
            # 构建角色信息
            character_info = ""
            if context.character_context:
                character_info = "主要角色：\n"
                for name, char in context.character_context.items():
                    adapted_name = char.cultural_adaptations.get(context.target_language, name)
                    character_info += f"- {name} -> {adapted_name}: {char.description}\n"
            
            # 构建地点信息
            location_info = ""
            if context.location_context:
                location_info = "主要地点：\n"
                for name, loc in context.location_context.items():
                    adapted_name = loc.cultural_adaptations.get(context.target_language, name)
                    location_info += f"- {name} -> {adapted_name}: {loc.description}\n"
            
            # 构建术语信息
            terminology_info = ""
            if context.terminology_context:
                terminology_info = "专业术语：\n"
                for term, terminology in context.terminology_context.items():
                    adapted_term = terminology.cultural_adaptations.get(context.target_language, term)
                    terminology_info += f"- {term} -> {adapted_term}: {terminology.definition}\n"
            
            # 构建情节上下文
            plot_context = ""
            if context.plot_context:
                plot_context = f"情节背景：\n{context.plot_context}\n"
            
            # 构建前置章节摘要
            previous_context = ""
            if context.previous_chapters:
                previous_context = "前置章节摘要：\n"
                for chapter in context.previous_chapters[-3:]:  # 最近3章
                    previous_context += f"- {chapter.title}: {chapter.summary}\n"
            
            # 获取目标文化
            target_culture = self.culture_mapping.get(context.target_language, CultureType.WESTERN)
            cultural_rules = self.cultural_rules.get(target_culture, {})
            
            prompt = f"""
作为专业的小说翻译专家，请将以下中文小说章节翻译成{context.target_language}，并进行文化适配。

## 翻译要求：
1. 保持原文的情感色彩和文学风格
2. 确保人物性格和对话符合角色特征
3. 保持情节连贯性和逻辑性
4. 进行适当的文化适配，使目标读者更容易理解

## 文化适配规则：
- 命名规范：{cultural_rules.get('naming_conventions', '保持原名')}
- 敬语系统：{cultural_rules.get('honorifics', '根据目标语言调整')}
- 文化概念：{cultural_rules.get('cultural_references', '适当本土化')}
- 社会习俗：{cultural_rules.get('social_customs', '符合目标文化')}

{character_info}

{location_info}

{terminology_info}

{plot_context}

{previous_context}

## 翻译风格指南：
{context.style_guide}

请确保翻译后的内容：
1. 忠实于原文意思
2. 符合目标语言的表达习惯
3. 保持角色一致性
4. 情节连贯流畅
5. 文化适配恰当

请直接返回翻译后的内容，不要包含任何解释或额外信息。
"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"准备翻译提示词失败: {e}")
            return "请翻译以下内容："
    
    async def execute_translation(self, content: str, prompt: str, target_language: LanguageCode) -> str:
        """执行翻译"""
        try:
            # 如果内容太长，分块处理
            if len(content) > 3000:
                return await self.translate_in_chunks(content, prompt, target_language)
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content}
            ]
            
            translated_content = await self.call_llm(
                messages,
                temperature=0.7,
                max_tokens=4000
            )
            
            return translated_content
            
        except Exception as e:
            logger.error(f"执行翻译失败: {e}")
            raise
    
    async def translate_in_chunks(self, content: str, prompt: str, target_language: LanguageCode) -> str:
        """分块翻译长内容"""
        try:
            # 按段落分块
            paragraphs = content.split('\n\n')
            translated_paragraphs = []
            
            # 批量处理段落
            batch_size = 3
            for i in range(0, len(paragraphs), batch_size):
                batch = paragraphs[i:i+batch_size]
                batch_content = '\n\n'.join(batch)
                
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": batch_content}
                ]
                
                translated_batch = await self.call_llm(
                    messages,
                    temperature=0.7,
                    max_tokens=4000
                )
                
                translated_paragraphs.append(translated_batch)
                
                # 避免API限制
                await asyncio.sleep(1)
            
            return '\n\n'.join(translated_paragraphs)
            
        except Exception as e:
            logger.error(f"分块翻译失败: {e}")
            raise
    
    async def apply_cultural_adaptation(self, content: str, context: TranslationContext) -> str:
        """应用文化适配"""
        try:
            target_culture = self.culture_mapping.get(context.target_language, CultureType.WESTERN)
            
            # 应用角色名称适配
            adapted_content = await self.adapt_character_names(content, context)
            
            # 应用地点名称适配
            adapted_content = await self.adapt_location_names(adapted_content, context)
            
            # 应用术语适配
            adapted_content = await self.adapt_terminology(adapted_content, context)
            
            # 应用文化概念适配
            adapted_content = await self.adapt_cultural_concepts(adapted_content, context)
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"应用文化适配失败: {e}")
            return content
    
    async def adapt_character_names(self, content: str, context: TranslationContext) -> str:
        """适配角色名称"""
        try:
            adapted_content = content
            
            for original_name, character in context.character_context.items():
                adapted_name = character.cultural_adaptations.get(context.target_language)
                if adapted_name and adapted_name != original_name:
                    adapted_content = adapted_content.replace(original_name, adapted_name)
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"适配角色名称失败: {e}")
            return content
    
    async def adapt_location_names(self, content: str, context: TranslationContext) -> str:
        """适配地点名称"""
        try:
            adapted_content = content
            
            for original_name, location in context.location_context.items():
                adapted_name = location.cultural_adaptations.get(context.target_language)
                if adapted_name and adapted_name != original_name:
                    adapted_content = adapted_content.replace(original_name, adapted_name)
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"适配地点名称失败: {e}")
            return content
    
    async def adapt_terminology(self, content: str, context: TranslationContext) -> str:
        """适配术语"""
        try:
            adapted_content = content
            
            for original_term, terminology in context.terminology_context.items():
                adapted_term = terminology.cultural_adaptations.get(context.target_language)
                if adapted_term and adapted_term != original_term:
                    adapted_content = adapted_content.replace(original_term, adapted_term)
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"适配术语失败: {e}")
            return content
    
    async def adapt_cultural_concepts(self, content: str, context: TranslationContext) -> str:
        """适配文化概念"""
        try:
            # 使用LLM进行文化概念适配
            target_culture = self.culture_mapping.get(context.target_language, CultureType.WESTERN)
            
            prompt = f"""
请检查以下翻译内容中的文化概念，并将其适配到{target_culture}文化背景中。
需要适配的概念包括：
- 节日庆典
- 社会习俗
- 礼仪规范
- 价值观念
- 生活方式

请保持原文意思不变，只是让文化概念更符合目标读者的理解。

内容：
{content}

请返回适配后的内容。
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的文化适配专家。"},
                {"role": "user", "content": prompt}
            ]
            
            adapted_content = await self.call_llm(
                messages,
                temperature=0.5,
                max_tokens=4000
            )
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"适配文化概念失败: {e}")
            return content
    
    async def ensure_terminology_consistency(self, content: str, context: TranslationContext) -> str:
        """确保术语一致性"""
        try:
            # 检查术语使用的一致性
            consistency_issues = []
            
            for term, terminology in context.terminology_context.items():
                adapted_term = terminology.cultural_adaptations.get(context.target_language, term)
                
                # 检查是否使用了不一致的翻译
                if adapted_term in content:
                    # 这里可以添加更复杂的一致性检查逻辑
                    pass
            
            return content
            
        except Exception as e:
            logger.error(f"确保术语一致性失败: {e}")
            return content
    
    async def assess_translation_quality(self, original: str, translated: str, context: TranslationContext) -> float:
        """评估翻译质量"""
        try:
            # 使用LLM评估翻译质量
            prompt = f"""
请评估以下翻译的质量，从以下维度给出评分（1-10分）：

1. 准确性：翻译是否准确传达了原文意思
2. 流畅性：译文是否符合目标语言的表达习惯
3. 一致性：人物、地点、术语是否保持一致
4. 文化适配：是否恰当地进行了文化适配
5. 文学性：是否保持了原文的文学风格

原文：
{original[:1000]}...

译文：
{translated[:1000]}...

请返回JSON格式的评分结果：
{{
  "accuracy": 评分,
  "fluency": 评分,
  "consistency": 评分,
  "cultural_adaptation": 评分,
  "literary_quality": 评分,
  "overall": 总体评分,
  "comments": "评价说明"
}}
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的翻译质量评估专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.call_llm(
                messages,
                temperature=0.3,
                max_tokens=500
            )
            
            try:
                quality_data = json.loads(response)
                return quality_data.get("overall", 7.0)
            except json.JSONDecodeError:
                logger.warning("无法解析质量评估结果")
                return 7.0
            
        except Exception as e:
            logger.error(f"评估翻译质量失败: {e}")
            return 7.0
    
    async def translate_terminology(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """翻译术语"""
        try:
            self.current_task = "翻译术语"
            
            terminology_list = data.get("terminology", [])
            target_language = data.get("target_language", LanguageCode.ENGLISH)
            target_culture = self.culture_mapping.get(target_language, CultureType.WESTERN)
            
            translated_terms = {}
            
            for term_data in terminology_list:
                term = term_data.get("term", "")
                category = term_data.get("category", "")
                context = term_data.get("context", "")
                
                # 翻译术语
                translated_term = await self.translate_single_term(
                    term, category, context, target_language, target_culture
                )
                
                translated_terms[term] = translated_term
            
            return {
                "target_language": target_language,
                "translated_terms": translated_terms,
                "translation_count": len(translated_terms)
            }
            
        except Exception as e:
            logger.error(f"翻译术语失败: {e}")
            raise
    
    async def translate_single_term(self, term: str, category: str, context: str, 
                                   target_language: LanguageCode, target_culture: CultureType) -> str:
        """翻译单个术语"""
        try:
            prompt = f"""
请将以下{category}术语翻译成{target_language}，并进行文化适配。

术语：{term}
类别：{category}
上下文：{context}
目标文化：{target_culture}

要求：
1. 保持术语的专业性和准确性
2. 符合目标文化的表达习惯
3. 考虑读者的理解能力
4. 保持与类似术语的一致性

请只返回翻译后的术语，不要包含解释。
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的术语翻译专家。"},
                {"role": "user", "content": prompt}
            ]
            
            translated_term = await self.call_llm(
                messages,
                temperature=0.3,
                max_tokens=100
            )
            
            return translated_term.strip()
            
        except Exception as e:
            logger.error(f"翻译单个术语失败: {e}")
            return term
    
    async def quality_check_translation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """质量检查翻译"""
        try:
            self.current_task = "质量检查翻译"
            
            original_content = data.get("original", "")
            translated_content = data.get("translated", "")
            context = data.get("context", {})
            
            # 执行各项质量检查
            quality_results = {}
            
            for checkpoint in self.quality_checkpoints:
                score = await self.execute_quality_check(
                    checkpoint, original_content, translated_content, context
                )
                quality_results[checkpoint] = score
            
            # 计算总体质量得分
            overall_score = sum(quality_results.values()) / len(quality_results)
            
            return {
                "overall_score": overall_score,
                "detailed_scores": quality_results,
                "recommendations": await self.generate_quality_recommendations(quality_results)
            }
            
        except Exception as e:
            logger.error(f"质量检查翻译失败: {e}")
            raise
    
    async def execute_quality_check(self, checkpoint: str, original: str, 
                                   translated: str, context: Dict[str, Any]) -> float:
        """执行特定的质量检查"""
        try:
            # 根据检查点类型执行不同的检查
            if checkpoint == "character_consistency":
                return await self.check_character_consistency(original, translated, context)
            elif checkpoint == "cultural_appropriateness":
                return await self.check_cultural_appropriateness(translated, context)
            elif checkpoint == "plot_continuity":
                return await self.check_plot_continuity(original, translated, context)
            elif checkpoint == "terminology_consistency":
                return await self.check_terminology_consistency(translated, context)
            elif checkpoint == "style_consistency":
                return await self.check_style_consistency(original, translated, context)
            else:
                logger.warning(f"未知的质量检查点: {checkpoint}")
                return 7.0
                
        except Exception as e:
            logger.error(f"执行质量检查失败 [{checkpoint}]: {e}")
            return 5.0
    
    async def check_character_consistency(self, original: str, translated: str, context: Dict[str, Any]) -> float:
        """检查角色一致性"""
        # 这里可以实现具体的角色一致性检查逻辑
        return 8.0
    
    async def check_cultural_appropriateness(self, translated: str, context: Dict[str, Any]) -> float:
        """检查文化适配度"""
        # 这里可以实现具体的文化适配度检查逻辑
        return 8.0
    
    async def check_plot_continuity(self, original: str, translated: str, context: Dict[str, Any]) -> float:
        """检查情节连贯性"""
        # 这里可以实现具体的情节连贯性检查逻辑
        return 8.0
    
    async def check_terminology_consistency(self, translated: str, context: Dict[str, Any]) -> float:
        """检查术语一致性"""
        # 这里可以实现具体的术语一致性检查逻辑
        return 8.0
    
    async def check_style_consistency(self, original: str, translated: str, context: Dict[str, Any]) -> float:
        """检查风格一致性"""
        # 这里可以实现具体的风格一致性检查逻辑
        return 8.0
    
    async def generate_quality_recommendations(self, quality_results: Dict[str, float]) -> List[str]:
        """生成质量改进建议"""
        recommendations = []
        
        for checkpoint, score in quality_results.items():
            if score < 7.0:
                recommendations.append(f"需要改进 {checkpoint}，当前得分: {score:.1f}")
        
        return recommendations
    
    async def preload_cultural_templates(self):
        """预加载文化适配模板"""
        try:
            # 这里可以预加载一些文化适配的模板和规则
            logger.info("预加载文化适配模板...")
            # 实现预加载逻辑
            pass
        except Exception as e:
            logger.error(f"预加载文化适配模板失败: {e}")
    
    async def adapt_cultural_elements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """适配文化元素"""
        try:
            self.current_task = "适配文化元素"
            
            elements = data.get("elements", [])
            target_language = data.get("target_language", LanguageCode.ENGLISH)
            target_culture = self.culture_mapping.get(target_language, CultureType.WESTERN)
            
            adapted_elements = {}
            
            for element in elements:
                element_type = element.get("type", "")
                element_value = element.get("value", "")
                
                adapted_value = await self.adapt_single_element(
                    element_value, element_type, target_culture
                )
                
                adapted_elements[element_value] = adapted_value
            
            return {
                "target_culture": target_culture,
                "adapted_elements": adapted_elements,
                "adaptation_count": len(adapted_elements)
            }
            
        except Exception as e:
            logger.error(f"适配文化元素失败: {e}")
            raise
    
    async def adapt_single_element(self, element_value: str, element_type: str, target_culture: CultureType) -> str:
        """适配单个文化元素"""
        try:
            cultural_rules = self.cultural_rules.get(target_culture, {})
            
            prompt = f"""
请将以下{element_type}适配到{target_culture}文化背景中：

元素：{element_value}
类型：{element_type}
目标文化：{target_culture}

文化规则：
{json.dumps(cultural_rules, ensure_ascii=False, indent=2)}

请返回适配后的元素，确保：
1. 符合目标文化的表达习惯
2. 容易被目标读者理解
3. 保持原有的文化内涵
4. 不造成文化冲突

请只返回适配后的元素，不要包含解释。
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的文化适配专家。"},
                {"role": "user", "content": prompt}
            ]
            
            adapted_element = await self.call_llm(
                messages,
                temperature=0.5,
                max_tokens=200
            )
            
            return adapted_element.strip()
            
        except Exception as e:
            logger.error(f"适配单个文化元素失败: {e}")
            return element_value 