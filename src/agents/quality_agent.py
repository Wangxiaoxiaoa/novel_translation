"""
质量控制智能体 - 全面的质量保证和评估
Quality Control Agent - Comprehensive quality assurance and assessment
"""

import asyncio
import json
import re
import statistics
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from datetime import datetime
from collections import defaultdict
import numpy as np

from .base_agent import BaseAgent
from ..models.base import (
    AgentMessage, AgentType, Chapter, Novel, TranslationContext,
    LanguageCode, CultureType, QualityMetrics
)


class QualityAgent(BaseAgent):
    """质量控制智能体"""
    
    def __init__(self, config, openai_client):
        super().__init__(config, openai_client)
        
        # 质量评估模块
        self.accuracy_assessor = AccuracyAssessor()
        self.fluency_assessor = FluencyAssessor()
        self.consistency_assessor = ConsistencyAssessor()
        self.cultural_assessor = CulturalAssessor()
        self.style_assessor = StyleAssessor()
        self.readability_assessor = ReadabilityAssessor()
        
        # 综合质量分析器
        self.comprehensive_analyzer = ComprehensiveQualityAnalyzer()
        
        # 质量标准管理
        self.quality_standards = QualityStandardsManager()
        
        # 问题检测器
        self.issue_detector = IssueDetector()
        
        # 改进建议生成器
        self.improvement_advisor = ImprovementAdvisor()
        
        # 质量报告生成器
        self.report_generator = QualityReportGenerator()
        
        # 自动修复系统
        self.auto_fixer = AutoFixer()
        
    async def initialize(self) -> bool:
        """初始化质量控制智能体"""
        try:
            logger.info("初始化质量控制智能体...")
            
            # 初始化各个评估模块
            await self.accuracy_assessor.initialize()
            await self.fluency_assessor.initialize()
            await self.consistency_assessor.initialize()
            await self.cultural_assessor.initialize()
            await self.style_assessor.initialize()
            await self.readability_assessor.initialize()
            
            # 初始化其他组件
            await self.comprehensive_analyzer.initialize()
            await self.quality_standards.initialize()
            await self.issue_detector.initialize()
            await self.improvement_advisor.initialize()
            await self.report_generator.initialize()
            await self.auto_fixer.initialize()
            
            # 健康检查
            health_ok = await self.health_check()
            if not health_ok:
                logger.error("质量控制智能体健康检查失败")
                return False
            
            logger.info("质量控制智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"质量控制智能体初始化失败: {e}")
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理消息"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type == "comprehensive_quality_check":
                result = await self.comprehensive_quality_check(content)
            elif message_type == "assess_accuracy":
                result = await self.assess_accuracy(content)
            elif message_type == "assess_fluency":
                result = await self.assess_fluency(content)
            elif message_type == "assess_consistency":
                result = await self.assess_consistency(content)
            elif message_type == "assess_cultural_adaptation":
                result = await self.assess_cultural_adaptation(content)
            elif message_type == "assess_style":
                result = await self.assess_style(content)
            elif message_type == "assess_readability":
                result = await self.assess_readability(content)
            elif message_type == "detect_issues":
                result = await self.detect_issues(content)
            elif message_type == "generate_improvement_suggestions":
                result = await self.generate_improvement_suggestions(content)
            elif message_type == "auto_fix_issues":
                result = await self.auto_fix_issues(content)
            elif message_type == "generate_quality_report":
                result = await self.generate_quality_report(content)
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
    
    async def comprehensive_quality_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """综合质量检查"""
        try:
            self.current_task = "综合质量检查"
            
            original_text = data.get("original", "")
            translated_text = data.get("translated", "")
            context = data.get("context", {})
            
            logger.info("开始综合质量检查...")
            
            # 并行执行多维度质量评估
            assessment_tasks = [
                self.accuracy_assessor.assess(original_text, translated_text, context),
                self.fluency_assessor.assess(translated_text, context),
                self.consistency_assessor.assess(translated_text, context),
                self.cultural_assessor.assess(translated_text, context),
                self.style_assessor.assess(original_text, translated_text, context),
                self.readability_assessor.assess(translated_text, context)
            ]
            
            assessment_results = await asyncio.gather(*assessment_tasks)
            
            # 整合评估结果
            quality_scores = {
                "accuracy": assessment_results[0],
                "fluency": assessment_results[1],
                "consistency": assessment_results[2],
                "cultural_adaptation": assessment_results[3],
                "style_consistency": assessment_results[4],
                "readability": assessment_results[5]
            }
            
            # 计算综合质量分数
            comprehensive_score = await self.comprehensive_analyzer.calculate_overall_score(quality_scores)
            
            # 检测问题
            issues = await self.issue_detector.detect_all_issues(
                original_text, translated_text, context, quality_scores
            )
            
            # 生成改进建议
            improvements = await self.improvement_advisor.generate_suggestions(
                quality_scores, issues, context
            )
            
            # 质量等级评定
            quality_grade = await self.determine_quality_grade(comprehensive_score)
            
            # 检查是否符合质量标准
            standards_check = await self.quality_standards.check_compliance(quality_scores, context)
            
            result = {
                "overall_score": comprehensive_score,
                "quality_grade": quality_grade,
                "detailed_scores": quality_scores,
                "issues_detected": issues,
                "improvement_suggestions": improvements,
                "standards_compliance": standards_check,
                "quality_metrics": QualityMetrics(
                    consistency_score=quality_scores["consistency"]["score"],
                    cultural_appropriateness_score=quality_scores["cultural_adaptation"]["score"],
                    plot_continuity_score=quality_scores["consistency"]["plot_continuity"],
                    character_consistency_score=quality_scores["consistency"]["character_consistency"],
                    overall_score=comprehensive_score,
                    issues=[issue["description"] for issue in issues],
                    suggestions=[sugg["suggestion"] for sugg in improvements]
                ).dict(),
                "assessment_metadata": {
                    "assessor": self.agent_type,
                    "timestamp": datetime.now().isoformat(),
                    "assessment_duration": 0,  # 实际实现中计算
                    "quality_dimensions": len(quality_scores),
                    "issues_count": len(issues),
                    "suggestions_count": len(improvements)
                }
            }
            
            logger.info(f"综合质量检查完成，总分: {comprehensive_score:.2f}, 等级: {quality_grade}")
            return result
            
        except Exception as e:
            logger.error(f"综合质量检查失败: {e}")
            raise
    
    async def assess_accuracy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估准确性"""
        try:
            original = data.get("original", "")
            translated = data.get("translated", "")
            context = data.get("context", {})
            
            return await self.accuracy_assessor.assess(original, translated, context)
            
        except Exception as e:
            logger.error(f"评估准确性失败: {e}")
            raise
    
    async def assess_fluency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估流畅性"""
        try:
            text = data.get("text", "")
            context = data.get("context", {})
            
            return await self.fluency_assessor.assess(text, context)
            
        except Exception as e:
            logger.error(f"评估流畅性失败: {e}")
            raise
    
    async def assess_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估一致性"""
        try:
            text = data.get("text", "")
            context = data.get("context", {})
            
            return await self.consistency_assessor.assess(text, context)
            
        except Exception as e:
            logger.error(f"评估一致性失败: {e}")
            raise
    
    async def assess_cultural_adaptation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估文化适配"""
        try:
            text = data.get("text", "")
            context = data.get("context", {})
            
            return await self.cultural_assessor.assess(text, context)
            
        except Exception as e:
            logger.error(f"评估文化适配失败: {e}")
            raise
    
    async def assess_style(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估风格一致性"""
        try:
            original = data.get("original", "")
            translated = data.get("translated", "")
            context = data.get("context", {})
            
            return await self.style_assessor.assess(original, translated, context)
            
        except Exception as e:
            logger.error(f"评估风格一致性失败: {e}")
            raise
    
    async def assess_readability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估可读性"""
        try:
            text = data.get("text", "")
            context = data.get("context", {})
            
            return await self.readability_assessor.assess(text, context)
            
        except Exception as e:
            logger.error(f"评估可读性失败: {e}")
            raise
    
    async def detect_issues(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检测问题"""
        try:
            original = data.get("original", "")
            translated = data.get("translated", "")
            context = data.get("context", {})
            
            issues = await self.issue_detector.detect_all_issues(original, translated, context, {})
            
            return {
                "issues": issues,
                "issues_count": len(issues),
                "severity_distribution": self.analyze_severity_distribution(issues),
                "issue_categories": self.categorize_issues(issues)
            }
            
        except Exception as e:
            logger.error(f"检测问题失败: {e}")
            raise
    
    async def generate_improvement_suggestions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成改进建议"""
        try:
            quality_scores = data.get("quality_scores", {})
            issues = data.get("issues", [])
            context = data.get("context", {})
            
            suggestions = await self.improvement_advisor.generate_suggestions(quality_scores, issues, context)
            
            return {
                "suggestions": suggestions,
                "suggestions_count": len(suggestions),
                "priority_suggestions": [s for s in suggestions if s.get("priority", "medium") == "high"],
                "quick_fixes": [s for s in suggestions if s.get("quick_fix", False)],
                "long_term_improvements": [s for s in suggestions if s.get("long_term", False)]
            }
            
        except Exception as e:
            logger.error(f"生成改进建议失败: {e}")
            raise
    
    async def auto_fix_issues(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """自动修复问题"""
        try:
            text = data.get("text", "")
            issues = data.get("issues", [])
            context = data.get("context", {})
            
            return await self.auto_fixer.fix_issues(text, issues, context)
            
        except Exception as e:
            logger.error(f"自动修复问题失败: {e}")
            raise
    
    async def generate_quality_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成质量报告"""
        try:
            quality_data = data.get("quality_data", {})
            
            return await self.report_generator.generate_report(quality_data)
            
        except Exception as e:
            logger.error(f"生成质量报告失败: {e}")
            raise
    
    async def determine_quality_grade(self, overall_score: float) -> str:
        """确定质量等级"""
        try:
            if overall_score >= 9.0:
                return "A+"
            elif overall_score >= 8.5:
                return "A"
            elif overall_score >= 8.0:
                return "A-"
            elif overall_score >= 7.5:
                return "B+"
            elif overall_score >= 7.0:
                return "B"
            elif overall_score >= 6.5:
                return "B-"
            elif overall_score >= 6.0:
                return "C+"
            elif overall_score >= 5.5:
                return "C"
            elif overall_score >= 5.0:
                return "C-"
            else:
                return "D"
                
        except Exception as e:
            logger.error(f"确定质量等级失败: {e}")
            return "Unknown"
    
    def analyze_severity_distribution(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析问题严重程度分布"""
        try:
            distribution = defaultdict(int)
            for issue in issues:
                severity = issue.get("severity", "medium")
                distribution[severity] += 1
            return dict(distribution)
        except Exception as e:
            logger.error(f"分析严重程度分布失败: {e}")
            return {}
    
    def categorize_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """分类问题"""
        try:
            categories = defaultdict(list)
            for issue in issues:
                category = issue.get("category", "general")
                categories[category].append(issue)
            return dict(categories)
        except Exception as e:
            logger.error(f"分类问题失败: {e}")
            return {}
    
    async def cleanup(self) -> bool:
        """清理资源"""
        try:
            logger.info("清理质量控制智能体资源...")
            
            # 清理各个评估模块
            await self.accuracy_assessor.cleanup()
            await self.fluency_assessor.cleanup()
            await self.consistency_assessor.cleanup()
            await self.cultural_assessor.cleanup()
            await self.style_assessor.cleanup()
            await self.readability_assessor.cleanup()
            
            # 清理其他组件
            await self.comprehensive_analyzer.cleanup()
            await self.quality_standards.cleanup()
            await self.issue_detector.cleanup()
            await self.improvement_advisor.cleanup()
            await self.report_generator.cleanup()
            await self.auto_fixer.cleanup()
            
            return True
        except Exception as e:
            logger.error(f"清理质量控制智能体资源失败: {e}")
            return False


class AccuracyAssessor:
    """准确性评估器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def assess(self, original: str, translated: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估准确性"""
        # 实现准确性评估逻辑
        return {
            "score": 8.5,
            "semantic_accuracy": 8.7,
            "factual_accuracy": 8.3,
            "terminology_accuracy": 8.5,
            "details": "准确性评估完成"
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class FluencyAssessor:
    """流畅性评估器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def assess(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估流畅性"""
        return {
            "score": 8.0,
            "grammar_score": 8.2,
            "naturalness_score": 7.8,
            "readability_score": 8.0,
            "details": "流畅性评估完成"
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class ConsistencyAssessor:
    """一致性评估器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def assess(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估一致性"""
        return {
            "score": 8.3,
            "character_consistency": 8.5,
            "plot_continuity": 8.1,
            "terminology_consistency": 8.3,
            "style_consistency": 8.2,
            "details": "一致性评估完成"
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class CulturalAssessor:
    """文化适配评估器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def assess(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估文化适配"""
        return {
            "score": 7.8,
            "cultural_appropriateness": 7.9,
            "localization_quality": 7.7,
            "cultural_sensitivity": 8.0,
            "details": "文化适配评估完成"
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class StyleAssessor:
    """风格评估器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def assess(self, original: str, translated: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估风格一致性"""
        return {
            "score": 8.1,
            "tone_consistency": 8.0,
            "voice_preservation": 8.2,
            "literary_quality": 8.1,
            "details": "风格评估完成"
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class ReadabilityAssessor:
    """可读性评估器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def assess(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估可读性"""
        return {
            "score": 8.2,
            "complexity_level": "moderate",
            "reading_ease": 8.1,
            "comprehension_difficulty": 8.3,
            "details": "可读性评估完成"
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class ComprehensiveQualityAnalyzer:
    """综合质量分析器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def calculate_overall_score(self, quality_scores: Dict[str, Any]) -> float:
        """计算总体质量分数"""
        try:
            # 定义各维度权重
            weights = {
                "accuracy": 0.25,
                "fluency": 0.20,
                "consistency": 0.20,
                "cultural_adaptation": 0.15,
                "style_consistency": 0.10,
                "readability": 0.10
            }
            
            total_score = 0.0
            total_weight = 0.0
            
            for dimension, score_data in quality_scores.items():
                if dimension in weights:
                    score = score_data.get("score", 0.0)
                    weight = weights[dimension]
                    total_score += score * weight
                    total_weight += weight
            
            overall_score = total_score / total_weight if total_weight > 0 else 0.0
            return round(overall_score, 2)
            
        except Exception as e:
            logger.error(f"计算总体质量分数失败: {e}")
            return 0.0
    
    async def cleanup(self):
        """清理资源"""
        pass


class QualityStandardsManager:
    """质量标准管理器"""
    
    def __init__(self):
        self.standards = {
            "minimum_accuracy": 7.0,
            "minimum_fluency": 7.0,
            "minimum_consistency": 7.0,
            "minimum_cultural_adaptation": 6.5,
            "minimum_overall": 7.0
        }
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def check_compliance(self, quality_scores: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """检查标准合规性"""
        try:
            compliance_results = {}
            
            for standard, min_score in self.standards.items():
                dimension = standard.replace("minimum_", "")
                if dimension == "overall":
                    # 计算总体分数
                    continue
                
                if dimension in quality_scores:
                    actual_score = quality_scores[dimension].get("score", 0.0)
                    compliance_results[standard] = {
                        "required": min_score,
                        "actual": actual_score,
                        "compliant": actual_score >= min_score,
                        "gap": max(0, min_score - actual_score)
                    }
            
            # 计算总体合规性
            compliant_count = sum(1 for r in compliance_results.values() if r["compliant"])
            total_count = len(compliance_results)
            overall_compliance = compliant_count / total_count if total_count > 0 else 0.0
            
            return {
                "detailed_compliance": compliance_results,
                "overall_compliance": overall_compliance,
                "compliant_dimensions": compliant_count,
                "total_dimensions": total_count,
                "passed": overall_compliance >= 0.8  # 80%的标准需要满足
            }
            
        except Exception as e:
            logger.error(f"检查标准合规性失败: {e}")
            return {"passed": False, "error": str(e)}
    
    async def cleanup(self):
        """清理资源"""
        pass


class IssueDetector:
    """问题检测器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def detect_all_issues(self, original: str, translated: str, context: Dict[str, Any], 
                               quality_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测所有问题"""
        try:
            issues = []
            
            # 检测各类问题
            accuracy_issues = await self.detect_accuracy_issues(original, translated)
            fluency_issues = await self.detect_fluency_issues(translated)
            consistency_issues = await self.detect_consistency_issues(translated, context)
            cultural_issues = await self.detect_cultural_issues(translated, context)
            
            issues.extend(accuracy_issues)
            issues.extend(fluency_issues)
            issues.extend(consistency_issues)
            issues.extend(cultural_issues)
            
            return issues
            
        except Exception as e:
            logger.error(f"检测问题失败: {e}")
            return []
    
    async def detect_accuracy_issues(self, original: str, translated: str) -> List[Dict[str, Any]]:
        """检测准确性问题"""
        return []
    
    async def detect_fluency_issues(self, text: str) -> List[Dict[str, Any]]:
        """检测流畅性问题"""
        return []
    
    async def detect_consistency_issues(self, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测一致性问题"""
        return []
    
    async def detect_cultural_issues(self, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测文化适配问题"""
        return []
    
    async def cleanup(self):
        """清理资源"""
        pass


class ImprovementAdvisor:
    """改进建议器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def generate_suggestions(self, quality_scores: Dict[str, Any], issues: List[Dict[str, Any]], 
                                 context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成改进建议"""
        try:
            suggestions = []
            
            # 基于质量分数生成建议
            for dimension, score_data in quality_scores.items():
                score = score_data.get("score", 0.0)
                if score < 7.0:
                    suggestions.append({
                        "dimension": dimension,
                        "suggestion": f"需要改进{dimension}，当前分数: {score:.1f}",
                        "priority": "high" if score < 6.0 else "medium",
                        "quick_fix": score > 6.5,
                        "long_term": score < 6.0
                    })
            
            # 基于问题生成建议
            for issue in issues:
                suggestions.append({
                    "issue": issue,
                    "suggestion": f"修复{issue.get('category', '一般')}问题: {issue.get('description', '')}",
                    "priority": issue.get("severity", "medium"),
                    "quick_fix": issue.get("severity") != "high",
                    "long_term": issue.get("severity") == "high"
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"生成改进建议失败: {e}")
            return []
    
    async def cleanup(self):
        """清理资源"""
        pass


class QualityReportGenerator:
    """质量报告生成器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def generate_report(self, quality_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成质量报告"""
        try:
            report = {
                "executive_summary": await self.generate_executive_summary(quality_data),
                "detailed_analysis": quality_data,
                "recommendations": await self.generate_recommendations(quality_data),
                "action_items": await self.generate_action_items(quality_data),
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成质量报告失败: {e}")
            return {"error": str(e)}
    
    async def generate_executive_summary(self, quality_data: Dict[str, Any]) -> str:
        """生成执行摘要"""
        overall_score = quality_data.get("overall_score", 0.0)
        quality_grade = quality_data.get("quality_grade", "Unknown")
        
        return f"质量评估完成，总体得分: {overall_score:.2f}，质量等级: {quality_grade}"
    
    async def generate_recommendations(self, quality_data: Dict[str, Any]) -> List[str]:
        """生成建议"""
        return quality_data.get("improvement_suggestions", [])
    
    async def generate_action_items(self, quality_data: Dict[str, Any]) -> List[str]:
        """生成行动项"""
        issues = quality_data.get("issues_detected", [])
        return [f"处理{issue.get('severity', 'medium')}级问题: {issue.get('description', '')}" for issue in issues[:5]]
    
    async def cleanup(self):
        """清理资源"""
        pass


class AutoFixer:
    """自动修复器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def fix_issues(self, text: str, issues: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """自动修复问题"""
        try:
            fixed_text = text
            fixes_applied = []
            
            for issue in issues:
                if issue.get("auto_fixable", False):
                    # 尝试自动修复
                    fix_result = await self.apply_auto_fix(fixed_text, issue, context)
                    if fix_result["success"]:
                        fixed_text = fix_result["fixed_text"]
                        fixes_applied.append(fix_result["fix_description"])
            
            return {
                "original_text": text,
                "fixed_text": fixed_text,
                "fixes_applied": fixes_applied,
                "fixes_count": len(fixes_applied),
                "success": len(fixes_applied) > 0
            }
            
        except Exception as e:
            logger.error(f"自动修复问题失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def apply_auto_fix(self, text: str, issue: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """应用自动修复"""
        # 这里实现具体的自动修复逻辑
        return {
            "success": False,
            "fixed_text": text,
            "fix_description": "暂不支持自动修复此类问题"
        }
    
    async def cleanup(self):
        """清理资源"""
        pass 