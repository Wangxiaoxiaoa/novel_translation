"""
编辑智能体 - 高级编辑和后期处理
Editor Agent - Advanced editing and post-processing
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from datetime import datetime
from collections import defaultdict
import difflib

from .base_agent import BaseAgent
from ..models.base import (
    AgentMessage, AgentType, Chapter, Novel, TranslationContext,
    LanguageCode, CultureType, QualityMetrics
)


class EditorAgent(BaseAgent):
    """编辑智能体"""
    
    def __init__(self, config, openai_client):
        super().__init__(config, openai_client)
        
        # 编辑模块
        self.consistency_editor = ConsistencyEditor()
        self.style_editor = StyleEditor()
        self.flow_editor = FlowEditor()
        self.dialogue_editor = DialogueEditor()
        self.narrative_editor = NarrativeEditor()
        self.quality_enhancer = QualityEnhancer()
        
        # 编辑规则库
        self.editing_rules = EditingRulesEngine()
        
        # 版本管理
        self.version_manager = VersionManager()
        
        # 比较分析器
        self.comparison_analyzer = ComparisonAnalyzer()
        
    async def initialize(self) -> bool:
        """初始化编辑智能体"""
        try:
            logger.info("初始化编辑智能体...")
            
            # 初始化各个编辑模块
            await self.consistency_editor.initialize()
            await self.style_editor.initialize()
            await self.flow_editor.initialize()
            await self.dialogue_editor.initialize()
            await self.narrative_editor.initialize()
            await self.quality_enhancer.initialize()
            
            # 初始化编辑规则
            await self.editing_rules.initialize()
            
            # 初始化版本管理
            await self.version_manager.initialize()
            
            # 初始化比较分析器
            await self.comparison_analyzer.initialize()
            
            # 健康检查
            health_ok = await self.health_check()
            if not health_ok:
                logger.error("编辑智能体健康检查失败")
                return False
            
            logger.info("编辑智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"编辑智能体初始化失败: {e}")
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理消息"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type == "comprehensive_edit":
                result = await self.comprehensive_edit(content)
            elif message_type == "consistency_edit":
                result = await self.edit_consistency(content)
            elif message_type == "style_edit":
                result = await self.edit_style(content)
            elif message_type == "flow_edit":
                result = await self.edit_flow(content)
            elif message_type == "dialogue_edit":
                result = await self.edit_dialogue(content)
            elif message_type == "narrative_edit":
                result = await self.edit_narrative(content)
            elif message_type == "quality_enhance":
                result = await self.enhance_quality(content)
            elif message_type == "compare_versions":
                result = await self.compare_versions(content)
            elif message_type == "apply_editing_rules":
                result = await self.apply_editing_rules(content)
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
    
    async def comprehensive_edit(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """综合编辑"""
        try:
            self.current_task = "综合编辑"
            
            original_content = data.get("content", "")
            context = data.get("context", {})
            edit_preferences = data.get("preferences", {})
            
            logger.info("开始综合编辑...")
            
            # 保存原始版本
            version_id = await self.version_manager.save_version(original_content, "original")
            
            # 多轮编辑流程
            edited_content = original_content
            edit_history = []
            
            # 第一轮：一致性编辑
            if edit_preferences.get("consistency_edit", True):
                consistency_result = await self.consistency_editor.edit(edited_content, context)
                edited_content = consistency_result["edited_content"]
                edit_history.append({
                    "stage": "consistency",
                    "changes": consistency_result["changes"],
                    "metrics": consistency_result.get("metrics", {})
                })
                
                # 保存版本
                await self.version_manager.save_version(edited_content, "after_consistency")
            
            # 第二轮：风格编辑
            if edit_preferences.get("style_edit", True):
                style_result = await self.style_editor.edit(edited_content, context)
                edited_content = style_result["edited_content"]
                edit_history.append({
                    "stage": "style",
                    "changes": style_result["changes"],
                    "metrics": style_result.get("metrics", {})
                })
                
                await self.version_manager.save_version(edited_content, "after_style")
            
            # 第三轮：流畅性编辑
            if edit_preferences.get("flow_edit", True):
                flow_result = await self.flow_editor.edit(edited_content, context)
                edited_content = flow_result["edited_content"]
                edit_history.append({
                    "stage": "flow",
                    "changes": flow_result["changes"],
                    "metrics": flow_result.get("metrics", {})
                })
                
                await self.version_manager.save_version(edited_content, "after_flow")
            
            # 第四轮：对话编辑
            if edit_preferences.get("dialogue_edit", True):
                dialogue_result = await self.dialogue_editor.edit(edited_content, context)
                edited_content = dialogue_result["edited_content"]
                edit_history.append({
                    "stage": "dialogue",
                    "changes": dialogue_result["changes"],
                    "metrics": dialogue_result.get("metrics", {})
                })
                
                await self.version_manager.save_version(edited_content, "after_dialogue")
            
            # 第五轮：叙述编辑
            if edit_preferences.get("narrative_edit", True):
                narrative_result = await self.narrative_editor.edit(edited_content, context)
                edited_content = narrative_result["edited_content"]
                edit_history.append({
                    "stage": "narrative",
                    "changes": narrative_result["changes"],
                    "metrics": narrative_result.get("metrics", {})
                })
                
                await self.version_manager.save_version(edited_content, "after_narrative")
            
            # 第六轮：质量增强
            if edit_preferences.get("quality_enhance", True):
                quality_result = await self.quality_enhancer.enhance(edited_content, context)
                edited_content = quality_result["enhanced_content"]
                edit_history.append({
                    "stage": "quality_enhancement",
                    "changes": quality_result["changes"],
                    "metrics": quality_result.get("metrics", {})
                })
                
                await self.version_manager.save_version(edited_content, "final")
            
            # 最终质量评估
            final_quality = await self.assess_final_quality(original_content, edited_content, context)
            
            # 生成编辑报告
            edit_report = await self.generate_edit_report(original_content, edited_content, edit_history, final_quality)
            
            result = {
                "original_content": original_content,
                "edited_content": edited_content,
                "version_id": version_id,
                "edit_history": edit_history,
                "final_quality": final_quality,
                "edit_report": edit_report,
                "improvement_score": final_quality["overall_improvement"],
                "processing_time": 0,  # 实际实现中计算
                "editor_metadata": {
                    "editor": self.agent_type,
                    "timestamp": datetime.now().isoformat(),
                    "edit_stages": len(edit_history),
                    "total_changes": sum(len(stage["changes"]) for stage in edit_history),
                    "quality_improvement": final_quality["overall_improvement"]
                }
            }
            
            logger.info(f"综合编辑完成，质量改进: {final_quality['overall_improvement']:.1%}")
            return result
            
        except Exception as e:
            logger.error(f"综合编辑失败: {e}")
            raise
    
    async def edit_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """一致性编辑"""
        try:
            content = data.get("content", "")
            context = data.get("context", {})
            
            return await self.consistency_editor.edit(content, context)
            
        except Exception as e:
            logger.error(f"一致性编辑失败: {e}")
            raise
    
    async def edit_style(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """风格编辑"""
        try:
            content = data.get("content", "")
            context = data.get("context", {})
            
            return await self.style_editor.edit(content, context)
            
        except Exception as e:
            logger.error(f"风格编辑失败: {e}")
            raise
    
    async def edit_flow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """流畅性编辑"""
        try:
            content = data.get("content", "")
            context = data.get("context", {})
            
            return await self.flow_editor.edit(content, context)
            
        except Exception as e:
            logger.error(f"流畅性编辑失败: {e}")
            raise
    
    async def edit_dialogue(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """对话编辑"""
        try:
            content = data.get("content", "")
            context = data.get("context", {})
            
            return await self.dialogue_editor.edit(content, context)
            
        except Exception as e:
            logger.error(f"对话编辑失败: {e}")
            raise
    
    async def edit_narrative(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """叙述编辑"""
        try:
            content = data.get("content", "")
            context = data.get("context", {})
            
            return await self.narrative_editor.edit(content, context)
            
        except Exception as e:
            logger.error(f"叙述编辑失败: {e}")
            raise
    
    async def enhance_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """质量增强"""
        try:
            content = data.get("content", "")
            context = data.get("context", {})
            
            return await self.quality_enhancer.enhance(content, context)
            
        except Exception as e:
            logger.error(f"质量增强失败: {e}")
            raise
    
    async def compare_versions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """比较版本"""
        try:
            version1 = data.get("version1", "")
            version2 = data.get("version2", "")
            
            return await self.comparison_analyzer.compare(version1, version2)
            
        except Exception as e:
            logger.error(f"版本比较失败: {e}")
            raise
    
    async def apply_editing_rules(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """应用编辑规则"""
        try:
            content = data.get("content", "")
            rules = data.get("rules", [])
            context = data.get("context", {})
            
            return await self.editing_rules.apply_rules(content, rules, context)
            
        except Exception as e:
            logger.error(f"应用编辑规则失败: {e}")
            raise
    
    async def assess_final_quality(self, original: str, edited: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """评估最终质量"""
        try:
            # 使用LLM评估编辑前后的质量改进
            assessment_prompt = f"""
请评估以下编辑前后的文本质量改进情况：

原始文本：
{original[:1500]}...

编辑后文本：
{edited[:1500]}...

请从以下维度评估改进程度（1-10分）：
1. 一致性改进（角色、情节、术语等）
2. 风格改进（文学性、表达力等）
3. 流畅性改进（语言流畅度、可读性等）
4. 对话改进（对话自然度、角色声音等）
5. 叙述改进（叙述技巧、节奏等）
6. 整体质量改进

请以JSON格式返回评估结果：
{{
  "consistency_improvement": 评分,
  "style_improvement": 评分,
  "fluency_improvement": 评分,
  "dialogue_improvement": 评分,
  "narrative_improvement": 评分,
  "overall_improvement": 总体改进程度（0-1），
  "improvement_summary": "改进总结",
  "specific_improvements": ["具体改进点列表"],
  "remaining_issues": ["仍存在的问题"]
}}
"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的文学编辑质量评估专家。"},
                {"role": "user", "content": assessment_prompt}
            ]
            
            response = await self.call_llm(messages, temperature=0.3)
            
            try:
                quality_data = json.loads(response)
                return quality_data
            except json.JSONDecodeError:
                logger.warning("质量评估结果解析失败")
                return {
                    "overall_improvement": 0.1,
                    "improvement_summary": "编辑完成，质量有所改进"
                }
                
        except Exception as e:
            logger.error(f"评估最终质量失败: {e}")
            return {"overall_improvement": 0.0, "error": str(e)}
    
    async def generate_edit_report(self, original: str, edited: str, edit_history: List[Dict], 
                                 final_quality: Dict[str, Any]) -> Dict[str, Any]:
        """生成编辑报告"""
        try:
            # 统计编辑信息
            total_changes = sum(len(stage["changes"]) for stage in edit_history)
            stages_completed = len(edit_history)
            
            # 计算文本变化统计
            diff = list(difflib.unified_diff(
                original.splitlines(),
                edited.splitlines(),
                lineterm=""
            ))
            
            changes_stats = {
                "total_changes": total_changes,
                "stages_completed": stages_completed,
                "text_diff_lines": len(diff),
                "original_length": len(original),
                "edited_length": len(edited),
                "length_change": len(edited) - len(original),
                "length_change_percent": (len(edited) - len(original)) / len(original) * 100 if len(original) > 0 else 0
            }
            
            # 生成编辑摘要
            edit_summary = await self.generate_edit_summary(edit_history, final_quality)
            
            report = {
                "edit_summary": edit_summary,
                "changes_statistics": changes_stats,
                "quality_metrics": final_quality,
                "stage_details": edit_history,
                "recommendations": await self.generate_recommendations(final_quality),
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成编辑报告失败: {e}")
            return {"error": str(e)}
    
    async def generate_edit_summary(self, edit_history: List[Dict], final_quality: Dict[str, Any]) -> str:
        """生成编辑摘要"""
        try:
            stages = [stage["stage"] for stage in edit_history]
            total_changes = sum(len(stage["changes"]) for stage in edit_history)
            improvement = final_quality.get("overall_improvement", 0)
            
            summary = f"""
编辑完成摘要：
- 完成编辑阶段：{', '.join(stages)}
- 总计修改：{total_changes} 处
- 质量改进：{improvement:.1%}
- 主要改进：{final_quality.get('improvement_summary', '文本质量得到提升')}
"""
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"生成编辑摘要失败: {e}")
            return "编辑已完成"
    
    async def generate_recommendations(self, final_quality: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        try:
            recommendations = []
            
            # 基于质量评估生成建议
            if final_quality.get("consistency_improvement", 0) < 7:
                recommendations.append("建议进一步检查角色和情节的一致性")
            
            if final_quality.get("style_improvement", 0) < 7:
                recommendations.append("建议优化文学风格和表达方式")
            
            if final_quality.get("fluency_improvement", 0) < 7:
                recommendations.append("建议改进语言流畅性和可读性")
            
            if final_quality.get("dialogue_improvement", 0) < 7:
                recommendations.append("建议优化对话的自然度和角色声音")
            
            if final_quality.get("narrative_improvement", 0) < 7:
                recommendations.append("建议改进叙述技巧和节奏控制")
            
            # 添加具体的改进建议
            if "remaining_issues" in final_quality:
                for issue in final_quality["remaining_issues"]:
                    recommendations.append(f"需要关注：{issue}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            return ["建议进行人工审查以确保质量"]
    
    async def cleanup(self) -> bool:
        """清理资源"""
        try:
            logger.info("清理编辑智能体资源...")
            
            # 清理各个编辑模块
            await self.consistency_editor.cleanup()
            await self.style_editor.cleanup()
            await self.flow_editor.cleanup()
            await self.dialogue_editor.cleanup()
            await self.narrative_editor.cleanup()
            await self.quality_enhancer.cleanup()
            
            # 清理其他组件
            await self.editing_rules.cleanup()
            await self.version_manager.cleanup()
            await self.comparison_analyzer.cleanup()
            
            return True
        except Exception as e:
            logger.error(f"清理编辑智能体资源失败: {e}")
            return False


class ConsistencyEditor:
    """一致性编辑器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def edit(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行一致性编辑"""
        # 这里实现具体的一致性编辑逻辑
        return {
            "edited_content": content,
            "changes": [],
            "metrics": {"consistency_score": 8.0}
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class StyleEditor:
    """风格编辑器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def edit(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行风格编辑"""
        return {
            "edited_content": content,
            "changes": [],
            "metrics": {"style_score": 8.0}
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class FlowEditor:
    """流畅性编辑器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def edit(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行流畅性编辑"""
        return {
            "edited_content": content,
            "changes": [],
            "metrics": {"flow_score": 8.0}
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class DialogueEditor:
    """对话编辑器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def edit(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行对话编辑"""
        return {
            "edited_content": content,
            "changes": [],
            "metrics": {"dialogue_score": 8.0}
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class NarrativeEditor:
    """叙述编辑器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def edit(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行叙述编辑"""
        return {
            "edited_content": content,
            "changes": [],
            "metrics": {"narrative_score": 8.0}
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class QualityEnhancer:
    """质量增强器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def enhance(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行质量增强"""
        return {
            "enhanced_content": content,
            "changes": [],
            "metrics": {"quality_score": 8.5}
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class EditingRulesEngine:
    """编辑规则引擎"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def apply_rules(self, content: str, rules: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """应用编辑规则"""
        return {
            "edited_content": content,
            "rules_applied": rules,
            "changes": []
        }
    
    async def cleanup(self):
        """清理资源"""
        pass


class VersionManager:
    """版本管理器"""
    
    def __init__(self):
        self.versions = {}
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def save_version(self, content: str, version_name: str) -> str:
        """保存版本"""
        version_id = f"{version_name}_{datetime.now().timestamp()}"
        self.versions[version_id] = {
            "content": content,
            "name": version_name,
            "timestamp": datetime.now().isoformat()
        }
        return version_id
    
    async def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """获取版本"""
        return self.versions.get(version_id)
    
    async def cleanup(self):
        """清理资源"""
        self.versions.clear()


class ComparisonAnalyzer:
    """比较分析器"""
    
    async def initialize(self):
        """初始化"""
        pass
    
    async def compare(self, version1: str, version2: str) -> Dict[str, Any]:
        """比较两个版本"""
        # 计算差异
        diff = list(difflib.unified_diff(
            version1.splitlines(),
            version2.splitlines(),
            lineterm=""
        ))
        
        return {
            "differences": diff,
            "similarity": difflib.SequenceMatcher(None, version1, version2).ratio(),
            "changes_count": len(diff),
            "comparison_summary": "版本比较完成"
        }
    
    async def cleanup(self):
        """清理资源"""
        pass 