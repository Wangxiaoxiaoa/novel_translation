#!/usr/bin/env python3
"""
专业审校智能体
Professional Review Agent - 模拟人工审校流程
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
    AgentMessage, AgentConfig, AgentType, 
    TranslationTask, QualityMetrics, ReviewResult
)


@dataclass
class ReviewCriteria:
    """审校标准"""
    accuracy: float = 0.0           # 准确性
    fluency: float = 0.0            # 流畅性
    consistency: float = 0.0        # 一致性
    cultural_adaptation: float = 0.0 # 文化适配
    readability: float = 0.0        # 可读性
    style_match: float = 0.0        # 风格匹配
    terminology: float = 0.0        # 术语准确性
    context_preservation: float = 0.0 # 语境保持
    creative_enhancement: float = 0.0 # 创意增强
    overall_quality: float = 0.0    # 整体质量


@dataclass
class ReviewIssue:
    """审校问题"""
    issue_type: str                 # 问题类型
    severity: str                   # 严重程度 (critical, major, minor)
    location: str                   # 位置描述
    description: str                # 问题描述
    original_text: str              # 原文
    problematic_text: str           # 有问题的文本
    suggested_fix: str              # 建议修复
    explanation: str                # 解释说明
    confidence: float               # 置信度


@dataclass
class ReviewReport:
    """审校报告"""
    review_id: str
    timestamp: datetime
    reviewer_type: str              # 审校类型
    criteria_scores: ReviewCriteria
    issues: List[ReviewIssue]
    recommendations: List[str]
    overall_assessment: str
    needs_revision: bool
    revision_priority: str          # high, medium, low
    estimated_revision_time: int    # 预估修改时间(分钟)


class ProfessionalReviewAgent(BaseAgent):
    """专业审校智能体"""
    
    def __init__(self, config: AgentConfig, openai_client: AsyncOpenAI):
        super().__init__(config, openai_client)
        self.agent_type = AgentType.PROFESSIONAL_REVIEW
        
        # 审校配置
        self.review_config = {
            "multi_stage_review": True,
            "quality_threshold": 8.5,
            "max_review_rounds": 3,
            "review_depth": "comprehensive",
            "cultural_sensitivity": True,
            "genre_specific": True,
            "terminology_check": True,
            "style_consistency": True,
            "readability_analysis": True,
            "creative_assessment": True
        }
        
        # 审校标准
        self.review_standards = {
            "accuracy": {
                "excellent": 9.5,
                "good": 8.0,
                "acceptable": 7.0,
                "needs_improvement": 5.0
            },
            "fluency": {
                "excellent": 9.0,
                "good": 7.5,
                "acceptable": 6.5,
                "needs_improvement": 5.0
            },
            "cultural_adaptation": {
                "excellent": 9.0,
                "good": 7.5,
                "acceptable": 6.0,
                "needs_improvement": 4.0
            }
        }
        
        # 问题分类
        self.issue_categories = {
            "translation_errors": ["mistranslation", "omission", "addition"],
            "language_issues": ["grammar", "syntax", "vocabulary", "punctuation"],
            "cultural_issues": ["cultural_insensitivity", "inappropriate_adaptation"],
            "consistency_issues": ["character_names", "terminology", "style"],
            "readability_issues": ["unclear_expression", "awkward_phrasing", "flow"],
            "creative_issues": ["loss_of_literary_quality", "bland_expression"]
        }
        
        # 审校员类型
        self.reviewer_types = {
            "linguistic_expert": "语言学专家",
            "cultural_consultant": "文化顾问",
            "genre_specialist": "类型专家",
            "quality_assurance": "质量保证",
            "creative_editor": "创意编辑",
            "final_proofreader": "最终校对"
        }
        
        self.review_history = []
        self.quality_trends = {}
        
    async def initialize(self) -> bool:
        """初始化审校智能体"""
        try:
            logger.info("初始化专业审校智能体")
            
            # 加载审校模板
            await self._load_review_templates()
            
            # 初始化质量标准
            await self._initialize_quality_standards()
            
            # 设置审校流程
            await self._setup_review_workflow()
            
            logger.info("专业审校智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化专业审校智能体失败: {e}")
            return False
    
    async def _load_review_templates(self):
        """加载审校模板"""
        self.review_templates = {
            "linguistic_review": """
            作为资深语言学专家，请对以下翻译进行专业审校：
            
            原文: {original_text}
            译文: {translated_text}
            目标语言: {target_language}
            文本类型: {text_type}
            
            请从以下角度进行评估：
            1. 翻译准确性：是否准确传达原文意思
            2. 语言流畅性：译文是否自然流畅
            3. 语法正确性：是否符合目标语言语法规范
            4. 术语一致性：专业术语是否准确一致
            5. 语言规范性：是否符合目标语言表达习惯
            
            请详细指出问题，并提供具体的修改建议。
            """,
            
            "cultural_review": """
            作为文化顾问，请对以下翻译的文化适配性进行专业审校：
            
            原文: {original_text}
            译文: {translated_text}
            源文化: {source_culture}
            目标文化: {target_culture}
            
            请评估：
            1. 文化背景是否准确传达
            2. 文化特色元素是否恰当处理
            3. 是否存在文化冲突或不当表达
            4. 本土化程度是否合适
            5. 文化敏感性是否得到尊重
            
            请提供文化适配的具体建议。
            """,
            
            "creative_review": """
            作为创意编辑，请对以下翻译的文学性和创意性进行评估：
            
            原文: {original_text}
            译文: {translated_text}
            文学类型: {genre}
            
            请评估：
            1. 文学性：是否保持原文的文学魅力
            2. 创意性：是否有创新的表达方式
            3. 情感传达：是否准确传达情感色彩
            4. 艺术效果：是否达到应有的艺术效果
            5. 读者体验：是否为读者提供良好的阅读体验
            
            请提供创意改进建议。
            """,
            
            "quality_assessment": """
            作为质量保证专家，请对以下翻译进行综合质量评估：
            
            原文: {original_text}
            译文: {translated_text}
            
            请按照以下标准进行评分（1-10分）：
            1. 准确性 (Accuracy)
            2. 流畅性 (Fluency)
            3. 一致性 (Consistency)
            4. 文化适配 (Cultural Adaptation)
            5. 可读性 (Readability)
            6. 风格匹配 (Style Match)
            7. 术语准确性 (Terminology)
            8. 语境保持 (Context Preservation)
            9. 创意增强 (Creative Enhancement)
            10. 整体质量 (Overall Quality)
            
            请提供详细的评分理由和改进建议。
            """
        }
    
    async def _initialize_quality_standards(self):
        """初始化质量标准"""
        self.quality_standards = {
            "professional_grade": {
                "accuracy": 9.5,
                "fluency": 9.0,
                "consistency": 9.0,
                "cultural_adaptation": 8.5,
                "overall_minimum": 8.8
            },
            "publication_ready": {
                "accuracy": 9.0,
                "fluency": 8.5,
                "consistency": 8.5,
                "cultural_adaptation": 8.0,
                "overall_minimum": 8.2
            },
            "draft_quality": {
                "accuracy": 8.0,
                "fluency": 7.5,
                "consistency": 7.5,
                "cultural_adaptation": 7.0,
                "overall_minimum": 7.5
            }
        }
    
    async def _setup_review_workflow(self):
        """设置审校流程"""
        self.review_workflow = {
            "stage_1": {
                "name": "初步审校",
                "reviewers": ["linguistic_expert", "quality_assurance"],
                "focus": ["accuracy", "fluency", "basic_issues"],
                "threshold": 7.0
            },
            "stage_2": {
                "name": "文化审校",
                "reviewers": ["cultural_consultant", "genre_specialist"],
                "focus": ["cultural_adaptation", "consistency", "style"],
                "threshold": 8.0
            },
            "stage_3": {
                "name": "创意审校",
                "reviewers": ["creative_editor"],
                "focus": ["creativity", "readability", "enhancement"],
                "threshold": 8.5
            },
            "stage_4": {
                "name": "最终校对",
                "reviewers": ["final_proofreader"],
                "focus": ["final_polish", "overall_quality"],
                "threshold": 9.0
            }
        }
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理消息"""
        try:
            if message.message_type == "professional_review":
                return await self._handle_professional_review(message)
            elif message.message_type == "multi_stage_review":
                return await self._handle_multi_stage_review(message)
            elif message.message_type == "quality_assessment":
                return await self._handle_quality_assessment(message)
            elif message.message_type == "review_comparison":
                return await self._handle_review_comparison(message)
            else:
                return await self._handle_generic_review(message)
                
        except Exception as e:
            logger.error(f"处理专业审校消息失败: {e}")
            return AgentMessage(
                sender=self.agent_type,
                receiver=message.sender,
                message_type="error",
                content={"error": str(e)}
            )
    
    async def _handle_professional_review(self, message: AgentMessage) -> AgentMessage:
        """处理专业审校请求"""
        try:
            content = message.content
            original_text = content.get("original_text", "")
            translated_text = content.get("translated_text", "")
            target_language = content.get("target_language", "en")
            text_type = content.get("text_type", "novel")
            
            # 执行多阶段审校
            review_result = await self._conduct_comprehensive_review(
                original_text, translated_text, target_language, text_type
            )
            
            return AgentMessage(
                sender=self.agent_type,
                receiver=message.sender,
                message_type="review_completed",
                content=review_result
            )
            
        except Exception as e:
            logger.error(f"专业审校处理失败: {e}")
            raise
    
    async def _conduct_comprehensive_review(
        self, 
        original_text: str, 
        translated_text: str, 
        target_language: str, 
        text_type: str
    ) -> Dict[str, Any]:
        """执行综合审校"""
        try:
            review_results = []
            
            # 多阶段审校
            for stage_name, stage_config in self.review_workflow.items():
                stage_results = await self._conduct_stage_review(
                    original_text, translated_text, target_language, 
                    text_type, stage_config
                )
                review_results.append({
                    "stage": stage_name,
                    "config": stage_config,
                    "results": stage_results
                })
            
            # 综合评估
            comprehensive_assessment = await self._generate_comprehensive_assessment(
                review_results
            )
            
            # 生成最终报告
            final_report = await self._generate_final_report(
                review_results, comprehensive_assessment
            )
            
            return {
                "review_id": f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "stage_results": review_results,
                "comprehensive_assessment": comprehensive_assessment,
                "final_report": final_report,
                "quality_grade": self._determine_quality_grade(comprehensive_assessment),
                "recommendations": self._generate_recommendations(review_results)
            }
            
        except Exception as e:
            logger.error(f"综合审校失败: {e}")
            raise
    
    async def _conduct_stage_review(
        self, 
        original_text: str, 
        translated_text: str, 
        target_language: str, 
        text_type: str, 
        stage_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行阶段审校"""
        try:
            stage_results = {}
            
            # 对每个审校员类型进行审校
            for reviewer_type in stage_config["reviewers"]:
                reviewer_result = await self._conduct_reviewer_assessment(
                    original_text, translated_text, target_language, 
                    text_type, reviewer_type, stage_config["focus"]
                )
                stage_results[reviewer_type] = reviewer_result
            
            # 计算阶段综合评分
            stage_score = self._calculate_stage_score(stage_results)
            
            return {
                "reviewer_results": stage_results,
                "stage_score": stage_score,
                "meets_threshold": stage_score >= stage_config["threshold"],
                "focus_areas": stage_config["focus"]
            }
            
        except Exception as e:
            logger.error(f"阶段审校失败: {e}")
            raise
    
    async def _conduct_reviewer_assessment(
        self, 
        original_text: str, 
        translated_text: str, 
        target_language: str, 
        text_type: str, 
        reviewer_type: str, 
        focus_areas: List[str]
    ) -> Dict[str, Any]:
        """执行审校员评估"""
        try:
            # 选择适当的审校模板
            if reviewer_type == "linguistic_expert":
                template = self.review_templates["linguistic_review"]
            elif reviewer_type == "cultural_consultant":
                template = self.review_templates["cultural_review"]
            elif reviewer_type == "creative_editor":
                template = self.review_templates["creative_review"]
            else:
                template = self.review_templates["quality_assessment"]
            
            # 准备提示词
            prompt = template.format(
                original_text=original_text,
                translated_text=translated_text,
                target_language=target_language,
                text_type=text_type,
                source_culture=self._detect_source_culture(original_text),
                target_culture=self._get_target_culture(target_language),
                genre=text_type
            )
            
            # 调用AI进行审校
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": f"你是一位专业的{self.reviewer_types[reviewer_type]}。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            review_text = response.choices[0].message.content
            
            # 解析审校结果
            parsed_result = await self._parse_review_result(review_text, reviewer_type)
            
            return {
                "reviewer_type": reviewer_type,
                "review_text": review_text,
                "parsed_result": parsed_result,
                "focus_areas": focus_areas,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"审校员评估失败: {e}")
            raise
    
    async def _parse_review_result(self, review_text: str, reviewer_type: str) -> Dict[str, Any]:
        """解析审校结果"""
        try:
            # 使用AI来解析审校结果
            parse_prompt = f"""
            请解析以下审校结果，提取关键信息：
            
            审校类型: {reviewer_type}
            审校内容: {review_text}
            
            请提取以下信息并以JSON格式返回：
            {{
                "issues": [
                    {{
                        "type": "问题类型",
                        "severity": "严重程度(critical/major/minor)",
                        "description": "问题描述",
                        "location": "位置",
                        "suggestion": "修改建议"
                    }}
                ],
                "scores": {{
                    "accuracy": 分数,
                    "fluency": 分数,
                    "consistency": 分数,
                    "cultural_adaptation": 分数,
                    "overall": 分数
                }},
                "recommendations": ["建议1", "建议2", "..."],
                "summary": "总结评价"
            }}
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本分析助手。"},
                    {"role": "user", "content": parse_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            parsed_text = response.choices[0].message.content
            
            # 尝试解析JSON
            try:
                parsed_result = json.loads(parsed_text)
                return parsed_result
            except json.JSONDecodeError:
                # 如果JSON解析失败，返回基本结构
                return {
                    "issues": [],
                    "scores": {"overall": 7.0},
                    "recommendations": [],
                    "summary": review_text[:200] + "..."
                }
                
        except Exception as e:
            logger.error(f"解析审校结果失败: {e}")
            return {
                "issues": [],
                "scores": {"overall": 6.0},
                "recommendations": [],
                "summary": "解析失败"
            }
    
    def _calculate_stage_score(self, stage_results: Dict[str, Any]) -> float:
        """计算阶段评分"""
        try:
            total_score = 0
            count = 0
            
            for reviewer_type, result in stage_results.items():
                parsed_result = result.get("parsed_result", {})
                scores = parsed_result.get("scores", {})
                overall_score = scores.get("overall", 6.0)
                
                total_score += overall_score
                count += 1
            
            return total_score / count if count > 0 else 6.0
            
        except Exception as e:
            logger.error(f"计算阶段评分失败: {e}")
            return 6.0
    
    async def _generate_comprehensive_assessment(self, review_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成综合评估"""
        try:
            # 收集所有评分
            all_scores = []
            all_issues = []
            all_recommendations = []
            
            for stage_result in review_results:
                stage_score = stage_result["results"]["stage_score"]
                all_scores.append(stage_score)
                
                # 收集所有问题和建议
                for reviewer_type, reviewer_result in stage_result["results"]["reviewer_results"].items():
                    parsed_result = reviewer_result.get("parsed_result", {})
                    issues = parsed_result.get("issues", [])
                    recommendations = parsed_result.get("recommendations", [])
                    
                    all_issues.extend(issues)
                    all_recommendations.extend(recommendations)
            
            # 计算综合评分
            comprehensive_score = sum(all_scores) / len(all_scores) if all_scores else 6.0
            
            # 分析问题分布
            issue_distribution = self._analyze_issue_distribution(all_issues)
            
            return {
                "comprehensive_score": comprehensive_score,
                "stage_scores": all_scores,
                "total_issues": len(all_issues),
                "issue_distribution": issue_distribution,
                "all_recommendations": list(set(all_recommendations)),  # 去重
                "quality_level": self._determine_quality_level(comprehensive_score),
                "needs_major_revision": comprehensive_score < 7.0,
                "needs_minor_revision": 7.0 <= comprehensive_score < 8.5,
                "publication_ready": comprehensive_score >= 8.5
            }
            
        except Exception as e:
            logger.error(f"生成综合评估失败: {e}")
            return {"comprehensive_score": 6.0, "error": str(e)}
    
    def _analyze_issue_distribution(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析问题分布"""
        try:
            issue_types = {}
            severity_counts = {"critical": 0, "major": 0, "minor": 0}
            
            for issue in issues:
                issue_type = issue.get("type", "unknown")
                severity = issue.get("severity", "minor")
                
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                severity_counts[severity] += 1
            
            return {
                "by_type": issue_types,
                "by_severity": severity_counts,
                "critical_issues": severity_counts["critical"],
                "major_issues": severity_counts["major"],
                "minor_issues": severity_counts["minor"]
            }
            
        except Exception as e:
            logger.error(f"分析问题分布失败: {e}")
            return {"by_type": {}, "by_severity": {}}
    
    def _determine_quality_level(self, score: float) -> str:
        """确定质量等级"""
        if score >= 9.0:
            return "excellent"
        elif score >= 8.5:
            return "very_good"
        elif score >= 8.0:
            return "good"
        elif score >= 7.0:
            return "acceptable"
        elif score >= 6.0:
            return "needs_improvement"
        else:
            return "poor"
    
    def _determine_quality_grade(self, assessment: Dict[str, Any]) -> str:
        """确定质量等级"""
        score = assessment.get("comprehensive_score", 6.0)
        
        if score >= 9.0:
            return "A+"
        elif score >= 8.5:
            return "A"
        elif score >= 8.0:
            return "A-"
        elif score >= 7.5:
            return "B+"
        elif score >= 7.0:
            return "B"
        elif score >= 6.5:
            return "B-"
        elif score >= 6.0:
            return "C"
        else:
            return "F"
    
    def _generate_recommendations(self, review_results: List[Dict[str, Any]]) -> List[str]:
        """生成建议"""
        try:
            recommendations = []
            
            for stage_result in review_results:
                stage_name = stage_result["stage"]
                stage_score = stage_result["results"]["stage_score"]
                
                if stage_score < 7.0:
                    recommendations.append(f"需要重点改进{stage_name}阶段的质量")
                elif stage_score < 8.0:
                    recommendations.append(f"{stage_name}阶段需要进一步优化")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            return []
    
    async def _generate_final_report(
        self, 
        review_results: List[Dict[str, Any]], 
        comprehensive_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成最终报告"""
        try:
            report = {
                "executive_summary": {
                    "overall_score": comprehensive_assessment.get("comprehensive_score", 6.0),
                    "quality_grade": self._determine_quality_grade(comprehensive_assessment),
                    "total_issues": comprehensive_assessment.get("total_issues", 0),
                    "recommendation": self._get_overall_recommendation(comprehensive_assessment)
                },
                "detailed_analysis": {
                    "stage_breakdown": review_results,
                    "issue_analysis": comprehensive_assessment.get("issue_distribution", {}),
                    "quality_metrics": comprehensive_assessment
                },
                "action_items": comprehensive_assessment.get("all_recommendations", []),
                "next_steps": self._generate_next_steps(comprehensive_assessment)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成最终报告失败: {e}")
            return {"error": str(e)}
    
    def _get_overall_recommendation(self, assessment: Dict[str, Any]) -> str:
        """获取总体建议"""
        score = assessment.get("comprehensive_score", 6.0)
        
        if score >= 9.0:
            return "优秀质量，可以发布"
        elif score >= 8.5:
            return "高质量，建议轻微修改后发布"
        elif score >= 8.0:
            return "良好质量，建议适度修改后发布"
        elif score >= 7.0:
            return "可接受质量，需要修改后发布"
        elif score >= 6.0:
            return "需要大幅改进后才能发布"
        else:
            return "质量不达标，需要重新翻译"
    
    def _generate_next_steps(self, assessment: Dict[str, Any]) -> List[str]:
        """生成下一步行动"""
        steps = []
        
        if assessment.get("needs_major_revision", False):
            steps.append("进行大幅修改")
            steps.append("重新进行质量审校")
        elif assessment.get("needs_minor_revision", False):
            steps.append("进行细节修改")
            steps.append("进行最终校对")
        else:
            steps.append("准备发布")
        
        return steps
    
    def _detect_source_culture(self, text: str) -> str:
        """检测源文化"""
        # 简单实现，可以根据文本内容判断
        if any(char in text for char in "中华国朝天子皇帝"):
            return "Chinese"
        elif any(word in text for word in ["samurai", "ninja", "sake", "sakura"]):
            return "Japanese"
        elif any(word in text for word in ["kimchi", "hanbok", "bulgogi"]):
            return "Korean"
        else:
            return "Unknown"
    
    def _get_target_culture(self, language_code: str) -> str:
        """获取目标文化"""
        culture_map = {
            "en": "Western",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
            "ar": "Arabic"
        }
        return culture_map.get(language_code, "Unknown")
    
    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理专业审校智能体资源")
            
            # 保存审校历史
            if self.review_history:
                # 可以保存到文件或数据库
                pass
            
            # 清理缓存
            self.review_history.clear()
            self.quality_trends.clear()
            
            logger.info("专业审校智能体资源清理完成")
            
        except Exception as e:
            logger.error(f"清理专业审校智能体资源失败: {e}") 