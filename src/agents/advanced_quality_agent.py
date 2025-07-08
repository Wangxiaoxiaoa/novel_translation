#!/usr/bin/env python3
"""
高级质量控制智能体
Advanced Quality Control Agent - 多层次质量监控和自动改进
"""

import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re

from .base_agent import BaseAgent
from ..models.base import (
    AgentMessage, AgentConfig, AgentType, 
    QualityMetrics, TranslationTask
)


@dataclass
class QualityDimension:
    """质量维度"""
    name: str
    weight: float
    threshold: float
    score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """质量报告"""
    translation_id: str
    timestamp: datetime
    overall_score: float
    dimensional_scores: Dict[str, QualityDimension]
    critical_issues: List[str]
    improvement_suggestions: List[str]
    quality_grade: str
    confidence_level: float
    auto_fix_applied: bool = False
    human_review_needed: bool = False


class AdvancedQualityAgent(BaseAgent):
    """高级质量控制智能体"""
    
    def __init__(self, config: AgentConfig, openai_client: AsyncOpenAI):
        super().__init__(config, openai_client)
        self.agent_type = AgentType.ADVANCED_QUALITY
        
        # 质量维度配置
        self.quality_dimensions = {
            "accuracy": QualityDimension("准确性", 0.25, 8.0),
            "fluency": QualityDimension("流畅性", 0.20, 7.5),
            "consistency": QualityDimension("一致性", 0.15, 8.0),
            "cultural_adaptation": QualityDimension("文化适配", 0.15, 7.0),
            "readability": QualityDimension("可读性", 0.10, 7.5),
            "style_preservation": QualityDimension("风格保持", 0.10, 7.0),
            "creativity": QualityDimension("创意性", 0.05, 6.5)
        }
        
        # 质量检测器
        self.quality_detectors = {
            "semantic_similarity": self._check_semantic_similarity,
            "fluency_analysis": self._check_fluency,
            "consistency_check": self._check_consistency,
            "cultural_appropriateness": self._check_cultural_appropriateness,
            "readability_assessment": self._check_readability,
            "style_analysis": self._check_style_preservation,
            "creativity_evaluation": self._check_creativity
        }
        
        # 自动修复器
        self.auto_fixers = {
            "grammar_fix": self._fix_grammar_issues,
            "consistency_fix": self._fix_consistency_issues,
            "fluency_fix": self._fix_fluency_issues,
            "cultural_fix": self._fix_cultural_issues,
            "style_fix": self._fix_style_issues
        }
        
        # 质量历史
        self.quality_history = []
        self.quality_trends = {}
        
        # 高级配置
        self.advanced_config = {
            "auto_fix_enabled": True,
            "human_review_threshold": 6.0,
            "confidence_threshold": 0.8,
            "max_auto_fix_attempts": 3,
            "quality_monitoring": True,
            "adaptive_thresholds": True,
            "genre_specific_evaluation": True
        }
    
    async def initialize(self) -> bool:
        """初始化高级质量控制智能体"""
        try:
            logger.info("初始化高级质量控制智能体")
            
            # 加载质量模型
            await self._load_quality_models()
            
            # 初始化质量标准
            await self._initialize_quality_standards()
            
            # 设置自动修复规则
            await self._setup_auto_fix_rules()
            
            logger.info("高级质量控制智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化高级质量控制智能体失败: {e}")
            return False
    
    async def _load_quality_models(self):
        """加载质量评估模型"""
        try:
            # 这里可以加载预训练的质量评估模型
            # 暂时使用基于规则的方法
            
            # 语义相似度模型
            self.semantic_model = None  # 可以使用sentence-transformers
            
            # 流畅性评估模型
            self.fluency_model = None
            
            # 一致性检查模型
            self.consistency_model = None
            
            logger.info("质量评估模型加载完成")
            
        except Exception as e:
            logger.error(f"加载质量评估模型失败: {e}")
    
    async def _initialize_quality_standards(self):
        """初始化质量标准"""
        try:
            # 不同文学类型的质量标准
            self.genre_standards = {
                "科幻小说": {
                    "technical_accuracy": 0.9,
                    "creativity": 0.8,
                    "consistency": 0.85
                },
                "言情小说": {
                    "emotional_expression": 0.9,
                    "cultural_sensitivity": 0.8,
                    "readability": 0.85
                },
                "悬疑小说": {
                    "atmosphere_preservation": 0.9,
                    "tension_maintenance": 0.85,
                    "logical_consistency": 0.9
                },
                "历史小说": {
                    "cultural_accuracy": 0.95,
                    "historical_authenticity": 0.9,
                    "formal_style": 0.8
                }
            }
            
            # 语言特定标准
            self.language_standards = {
                "en": {
                    "grammar_weight": 0.2,
                    "idiom_usage": 0.15,
                    "cultural_adaptation": 0.1
                },
                "ja": {
                    "honorific_system": 0.25,
                    "cultural_nuance": 0.2,
                    "style_consistency": 0.15
                },
                "ko": {
                    "honorific_system": 0.2,
                    "cultural_adaptation": 0.18,
                    "formality_level": 0.15
                }
            }
            
            logger.info("质量标准初始化完成")
            
        except Exception as e:
            logger.error(f"初始化质量标准失败: {e}")
    
    async def _setup_auto_fix_rules(self):
        """设置自动修复规则"""
        try:
            self.auto_fix_rules = {
                "grammar_errors": {
                    "detection_patterns": [
                        r"(?i)\b(a|an)\s+(?=[aeiou])",  # 冠词错误
                        r"(?i)\b(is|are|was|were)\s+(?=\w+ing)",  # 时态错误
                    ],
                    "severity": "major",
                    "auto_fixable": True
                },
                "consistency_errors": {
                    "character_names": True,
                    "terminology": True,
                    "style": True,
                    "auto_fixable": True
                },
                "fluency_issues": {
                    "awkward_phrases": True,
                    "word_order": True,
                    "redundancy": True,
                    "auto_fixable": True
                },
                "cultural_issues": {
                    "inappropriate_expressions": True,
                    "cultural_references": True,
                    "auto_fixable": False  # 需要人工审查
                }
            }
            
            logger.info("自动修复规则设置完成")
            
        except Exception as e:
            logger.error(f"设置自动修复规则失败: {e}")
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理质量控制消息"""
        try:
            if message.message_type == "quality_assessment":
                return await self._handle_quality_assessment(message)
            elif message.message_type == "auto_quality_improvement":
                return await self._handle_auto_improvement(message)
            elif message.message_type == "quality_monitoring":
                return await self._handle_quality_monitoring(message)
            elif message.message_type == "quality_comparison":
                return await self._handle_quality_comparison(message)
            else:
                return await self._handle_generic_quality_check(message)
                
        except Exception as e:
            logger.error(f"处理质量控制消息失败: {e}")
            return AgentMessage(
                sender=self.agent_type,
                receiver=message.sender,
                message_type="error",
                content={"error": str(e)}
            )
    
    async def _handle_quality_assessment(self, message: AgentMessage) -> AgentMessage:
        """处理质量评估请求"""
        try:
            content = message.content
            original_text = content.get("original_text", "")
            translated_text = content.get("translated_text", "")
            target_language = content.get("target_language", "en")
            genre = content.get("genre", "小说")
            
            # 执行全面质量评估
            quality_report = await self._conduct_comprehensive_quality_assessment(
                original_text, translated_text, target_language, genre
            )
            
            # 如果质量低于阈值，尝试自动修复
            if (quality_report.overall_score < self.advanced_config["human_review_threshold"] and
                self.advanced_config["auto_fix_enabled"]):
                
                improved_text = await self._apply_auto_improvements(
                    original_text, translated_text, quality_report
                )
                
                if improved_text != translated_text:
                    # 重新评估改进后的文本
                    improved_report = await self._conduct_comprehensive_quality_assessment(
                        original_text, improved_text, target_language, genre
                    )
                    improved_report.auto_fix_applied = True
                    
                    return AgentMessage(
                        sender=self.agent_type,
                        receiver=message.sender,
                        message_type="quality_assessment_completed",
                        content={
                            "original_report": quality_report.__dict__,
                            "improved_report": improved_report.__dict__,
                            "improved_text": improved_text,
                            "auto_fix_applied": True
                        }
                    )
            
            return AgentMessage(
                sender=self.agent_type,
                receiver=message.sender,
                message_type="quality_assessment_completed",
                content={
                    "quality_report": quality_report.__dict__,
                    "auto_fix_applied": False
                }
            )
            
        except Exception as e:
            logger.error(f"质量评估处理失败: {e}")
            raise
    
    async def _conduct_comprehensive_quality_assessment(
        self, 
        original_text: str, 
        translated_text: str, 
        target_language: str, 
        genre: str
    ) -> QualityReport:
        """执行全面质量评估"""
        try:
            # 初始化质量报告
            report = QualityReport(
                translation_id=f"trans_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                overall_score=0.0,
                dimensional_scores={},
                critical_issues=[],
                improvement_suggestions=[],
                quality_grade="",
                confidence_level=0.0
            )
            
            # 逐个评估质量维度
            total_weighted_score = 0.0
            total_weights = 0.0
            
            for dimension_name, dimension in self.quality_dimensions.items():
                # 评估该维度
                score = await self._evaluate_quality_dimension(
                    dimension_name, original_text, translated_text, target_language, genre
                )
                
                # 更新维度信息
                dimension.score = score
                
                # 计算加权分数
                weighted_score = score * dimension.weight
                total_weighted_score += weighted_score
                total_weights += dimension.weight
                
                # 收集问题和建议
                if score < dimension.threshold:
                    issues = await self._identify_dimension_issues(
                        dimension_name, original_text, translated_text, target_language
                    )
                    dimension.issues = issues
                    report.critical_issues.extend(issues)
                
                # 生成改进建议
                recommendations = await self._generate_dimension_recommendations(
                    dimension_name, score, dimension.threshold
                )
                dimension.recommendations = recommendations
                report.improvement_suggestions.extend(recommendations)
                
                # 存储维度评估结果
                report.dimensional_scores[dimension_name] = dimension
            
            # 计算总体评分
            report.overall_score = total_weighted_score / total_weights if total_weights > 0 else 0.0
            
            # 确定质量等级
            report.quality_grade = self._determine_quality_grade(report.overall_score)
            
            # 计算置信度
            report.confidence_level = await self._calculate_confidence_level(
                report.dimensional_scores, original_text, translated_text
            )
            
            # 判断是否需要人工审查
            report.human_review_needed = (
                report.overall_score < self.advanced_config["human_review_threshold"] or
                len(report.critical_issues) > 5 or
                report.confidence_level < self.advanced_config["confidence_threshold"]
            )
            
            return report
            
        except Exception as e:
            logger.error(f"全面质量评估失败: {e}")
            raise
    
    async def _evaluate_quality_dimension(
        self, 
        dimension_name: str, 
        original_text: str, 
        translated_text: str, 
        target_language: str, 
        genre: str
    ) -> float:
        """评估单个质量维度"""
        try:
            if dimension_name == "accuracy":
                return await self._check_semantic_similarity(original_text, translated_text)
            elif dimension_name == "fluency":
                return await self._check_fluency(translated_text, target_language)
            elif dimension_name == "consistency":
                return await self._check_consistency(translated_text)
            elif dimension_name == "cultural_adaptation":
                return await self._check_cultural_appropriateness(original_text, translated_text, target_language)
            elif dimension_name == "readability":
                return await self._check_readability(translated_text, target_language)
            elif dimension_name == "style_preservation":
                return await self._check_style_preservation(original_text, translated_text, genre)
            elif dimension_name == "creativity":
                return await self._check_creativity(original_text, translated_text)
            else:
                return 7.0  # 默认分数
                
        except Exception as e:
            logger.error(f"评估质量维度{dimension_name}失败: {e}")
            return 6.0  # 默认较低分数
    
    async def _check_semantic_similarity(self, original_text: str, translated_text: str) -> float:
        """检查语义相似度"""
        try:
            # 使用AI评估语义相似度
            prompt = f"""
            请评估以下原文和译文的语义相似度（1-10分）：
            
            原文：{original_text}
            译文：{translated_text}
            
            评估标准：
            - 意思是否准确传达
            - 关键信息是否完整
            - 语义层次是否保持
            - 细节是否准确
            
            请只返回数字分数。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是专业的翻译质量评估专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(re.search(r'\d+\.?\d*', score_text).group())
                return min(max(score, 1.0), 10.0)
            except:
                return 7.0
                
        except Exception as e:
            logger.error(f"检查语义相似度失败: {e}")
            return 7.0
    
    async def _check_fluency(self, translated_text: str, target_language: str) -> float:
        """检查流畅性"""
        try:
            prompt = f"""
            请评估以下{target_language}文本的流畅性（1-10分）：
            
            文本：{translated_text}
            
            评估标准：
            - 语法是否正确
            - 表达是否自然
            - 句式是否流畅
            - 用词是否恰当
            
            请只返回数字分数。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": f"你是专业的{target_language}语言专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(re.search(r'\d+\.?\d*', score_text).group())
                return min(max(score, 1.0), 10.0)
            except:
                return 7.0
                
        except Exception as e:
            logger.error(f"检查流畅性失败: {e}")
            return 7.0
    
    async def _check_consistency(self, translated_text: str) -> float:
        """检查一致性"""
        try:
            # 检查人名、地名、术语的一致性
            consistency_score = 8.0  # 基础分数
            
            # 简单的一致性检查
            # 检查重复词汇的翻译一致性
            words = translated_text.split()
            word_count = {}
            
            for word in words:
                if len(word) > 3:  # 只检查较长的词汇
                    word_count[word] = word_count.get(word, 0) + 1
            
            # 如果有重复词汇，假设一致性较好
            repeated_words = [word for word, count in word_count.items() if count > 1]
            
            if repeated_words:
                consistency_score += 0.5
            
            return min(consistency_score, 10.0)
            
        except Exception as e:
            logger.error(f"检查一致性失败: {e}")
            return 7.0
    
    async def _check_cultural_appropriateness(self, original_text: str, translated_text: str, target_language: str) -> float:
        """检查文化适配性"""
        try:
            prompt = f"""
            请评估以下翻译的文化适配性（1-10分）：
            
            原文：{original_text}
            译文：{translated_text}
            目标语言：{target_language}
            
            评估标准：
            - 文化背景是否恰当转换
            - 是否避免了文化冲突
            - 本土化程度是否合适
            - 文化敏感性是否得到尊重
            
            请只返回数字分数。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是跨文化交流专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(re.search(r'\d+\.?\d*', score_text).group())
                return min(max(score, 1.0), 10.0)
            except:
                return 7.0
                
        except Exception as e:
            logger.error(f"检查文化适配性失败: {e}")
            return 7.0
    
    async def _check_readability(self, translated_text: str, target_language: str) -> float:
        """检查可读性"""
        try:
            # 简单的可读性检查
            sentences = translated_text.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            # 理想句子长度
            ideal_length = 15
            length_score = max(0, 10 - abs(avg_sentence_length - ideal_length) / 2)
            
            return min(max(length_score, 1.0), 10.0)
            
        except Exception as e:
            logger.error(f"检查可读性失败: {e}")
            return 7.0
    
    async def _check_style_preservation(self, original_text: str, translated_text: str, genre: str) -> float:
        """检查风格保持"""
        try:
            prompt = f"""
            请评估以下{genre}翻译的风格保持度（1-10分）：
            
            原文：{original_text}
            译文：{translated_text}
            类型：{genre}
            
            评估标准：
            - 文学风格是否保持
            - 语言风格是否一致
            - 情感色彩是否准确
            - 语气是否恰当
            
            请只返回数字分数。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是文学风格分析专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(re.search(r'\d+\.?\d*', score_text).group())
                return min(max(score, 1.0), 10.0)
            except:
                return 7.0
                
        except Exception as e:
            logger.error(f"检查风格保持失败: {e}")
            return 7.0
    
    async def _check_creativity(self, original_text: str, translated_text: str) -> float:
        """检查创意性"""
        try:
            # 简单的创意性评估
            # 检查是否有创新的表达方式
            creative_indicators = [
                "metaphor", "simile", "alliteration", "wordplay",
                "比喻", "拟人", "排比", "对仗"
            ]
            
            creativity_score = 6.0  # 基础分数
            
            for indicator in creative_indicators:
                if indicator in translated_text.lower():
                    creativity_score += 0.3
            
            return min(creativity_score, 10.0)
            
        except Exception as e:
            logger.error(f"检查创意性失败: {e}")
            return 6.0
    
    async def _identify_dimension_issues(
        self, 
        dimension_name: str, 
        original_text: str, 
        translated_text: str, 
        target_language: str
    ) -> List[str]:
        """识别维度问题"""
        try:
            issues = []
            
            if dimension_name == "accuracy":
                # 检查准确性问题
                if "missing" in translated_text.lower():
                    issues.append("可能存在遗漏的信息")
                if len(translated_text) < len(original_text) * 0.5:
                    issues.append("译文过短，可能遗漏内容")
            
            elif dimension_name == "fluency":
                # 检查流畅性问题
                if "..." in translated_text:
                    issues.append("存在未完成的句子")
                if translated_text.count("that") > 5:
                    issues.append("可能存在重复或啰嗦的表达")
            
            elif dimension_name == "consistency":
                # 检查一致性问题
                words = translated_text.split()
                repeated_patterns = []
                for i in range(len(words) - 1):
                    if words[i] == words[i + 1]:
                        repeated_patterns.append(words[i])
                
                if repeated_patterns:
                    issues.append(f"存在重复词汇：{', '.join(set(repeated_patterns))}")
            
            return issues
            
        except Exception as e:
            logger.error(f"识别维度问题失败: {e}")
            return []
    
    async def _generate_dimension_recommendations(
        self, 
        dimension_name: str, 
        score: float, 
        threshold: float
    ) -> List[str]:
        """生成维度建议"""
        try:
            recommendations = []
            
            if score < threshold:
                gap = threshold - score
                
                if dimension_name == "accuracy":
                    recommendations.append("建议重新检查原文理解")
                    if gap > 2.0:
                        recommendations.append("需要重新翻译关键部分")
                
                elif dimension_name == "fluency":
                    recommendations.append("建议改进句式结构")
                    recommendations.append("检查语法和用词")
                
                elif dimension_name == "consistency":
                    recommendations.append("统一专有名词翻译")
                    recommendations.append("保持术语一致性")
                
                elif dimension_name == "cultural_adaptation":
                    recommendations.append("加强文化背景研究")
                    recommendations.append("调整文化适配策略")
                
                elif dimension_name == "readability":
                    recommendations.append("简化复杂句式")
                    recommendations.append("优化段落结构")
                
                elif dimension_name == "style_preservation":
                    recommendations.append("保持原文风格特色")
                    recommendations.append("调整语言风格")
                
                elif dimension_name == "creativity":
                    recommendations.append("增加创意表达")
                    recommendations.append("丰富语言色彩")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"生成维度建议失败: {e}")
            return []
    
    def _determine_quality_grade(self, score: float) -> str:
        """确定质量等级"""
        if score >= 9.0:
            return "A+"
        elif score >= 8.5:
            return "A"
        elif score >= 8.0:
            return "B+"
        elif score >= 7.5:
            return "B"
        elif score >= 7.0:
            return "C+"
        elif score >= 6.5:
            return "C"
        elif score >= 6.0:
            return "D"
        else:
            return "F"
    
    async def _calculate_confidence_level(
        self, 
        dimensional_scores: Dict[str, QualityDimension], 
        original_text: str, 
        translated_text: str
    ) -> float:
        """计算置信度"""
        try:
            # 基于各维度分数的方差计算置信度
            scores = [dim.score for dim in dimensional_scores.values()]
            
            if len(scores) > 1:
                variance = np.var(scores)
                # 方差越小，置信度越高
                confidence = max(0.1, 1.0 - variance / 10.0)
            else:
                confidence = 0.5
            
            # 文本长度因子
            length_factor = min(1.0, len(translated_text) / 100)
            
            return min(confidence * length_factor, 1.0)
            
        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 0.5
    
    async def _apply_auto_improvements(
        self, 
        original_text: str, 
        translated_text: str, 
        quality_report: QualityReport
    ) -> str:
        """应用自动改进"""
        try:
            improved_text = translated_text
            
            # 应用各种自动修复
            for fix_type, fixer in self.auto_fixers.items():
                if self._should_apply_fix(fix_type, quality_report):
                    improved_text = await fixer(original_text, improved_text)
            
            return improved_text
            
        except Exception as e:
            logger.error(f"应用自动改进失败: {e}")
            return translated_text
    
    def _should_apply_fix(self, fix_type: str, quality_report: QualityReport) -> bool:
        """判断是否应该应用修复"""
        try:
            # 根据质量报告决定是否应用特定修复
            if fix_type == "grammar_fix":
                return quality_report.dimensional_scores.get("fluency", QualityDimension("", 0, 0)).score < 7.0
            elif fix_type == "consistency_fix":
                return quality_report.dimensional_scores.get("consistency", QualityDimension("", 0, 0)).score < 7.0
            elif fix_type == "fluency_fix":
                return quality_report.dimensional_scores.get("fluency", QualityDimension("", 0, 0)).score < 7.0
            elif fix_type == "cultural_fix":
                return quality_report.dimensional_scores.get("cultural_adaptation", QualityDimension("", 0, 0)).score < 6.0
            elif fix_type == "style_fix":
                return quality_report.dimensional_scores.get("style_preservation", QualityDimension("", 0, 0)).score < 6.5
            
            return False
            
        except Exception as e:
            logger.error(f"判断修复应用失败: {e}")
            return False
    
    async def _fix_grammar_issues(self, original_text: str, translated_text: str) -> str:
        """修复语法问题"""
        try:
            prompt = f"""
            请修复以下翻译中的语法问题，保持原意不变：
            
            原文：{original_text}
            译文：{translated_text}
            
            请只返回修复后的译文，不要添加任何解释。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是专业的语言编辑。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=len(translated_text) * 2
            )
            
            fixed_text = response.choices[0].message.content.strip()
            return fixed_text if fixed_text else translated_text
            
        except Exception as e:
            logger.error(f"修复语法问题失败: {e}")
            return translated_text
    
    async def _fix_consistency_issues(self, original_text: str, translated_text: str) -> str:
        """修复一致性问题"""
        try:
            # 简单的一致性修复
            # 这里可以实现更复杂的一致性检查和修复逻辑
            return translated_text
            
        except Exception as e:
            logger.error(f"修复一致性问题失败: {e}")
            return translated_text
    
    async def _fix_fluency_issues(self, original_text: str, translated_text: str) -> str:
        """修复流畅性问题"""
        try:
            prompt = f"""
            请改善以下翻译的流畅性，使其更加自然：
            
            原文：{original_text}
            译文：{translated_text}
            
            请只返回改善后的译文，不要添加任何解释。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是专业的文字编辑。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=len(translated_text) * 2
            )
            
            improved_text = response.choices[0].message.content.strip()
            return improved_text if improved_text else translated_text
            
        except Exception as e:
            logger.error(f"修复流畅性问题失败: {e}")
            return translated_text
    
    async def _fix_cultural_issues(self, original_text: str, translated_text: str) -> str:
        """修复文化问题"""
        try:
            # 文化问题通常需要人工处理
            return translated_text
            
        except Exception as e:
            logger.error(f"修复文化问题失败: {e}")
            return translated_text
    
    async def _fix_style_issues(self, original_text: str, translated_text: str) -> str:
        """修复风格问题"""
        try:
            prompt = f"""
            请调整以下翻译的风格，使其更好地保持原文风格：
            
            原文：{original_text}
            译文：{translated_text}
            
            请只返回调整后的译文，不要添加任何解释。
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是专业的文学编辑。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=len(translated_text) * 2
            )
            
            adjusted_text = response.choices[0].message.content.strip()
            return adjusted_text if adjusted_text else translated_text
            
        except Exception as e:
            logger.error(f"修复风格问题失败: {e}")
            return translated_text
    
    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理高级质量控制智能体资源")
            
            # 保存质量历史
            if self.quality_history:
                # 可以保存到文件或数据库
                pass
            
            # 清理缓存
            self.quality_history.clear()
            self.quality_trends.clear()
            
            logger.info("高级质量控制智能体资源清理完成")
            
        except Exception as e:
            logger.error(f"清理高级质量控制智能体资源失败: {e}") 