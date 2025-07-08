"""
协调智能体 - 负责管理整个翻译流程和智能体协调
Coordinator Agent - Responsible for managing translation workflow and agent coordination
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from datetime import datetime
from enum import Enum

from .base_agent import BaseAgent
from ..models.base import (
    AgentMessage, AgentType, Novel, Chapter, TranslationTask, 
    TranslationContext, ProcessingStatus, QualityMetrics
)


class WorkflowStage(str, Enum):
    """工作流阶段枚举"""
    DOCUMENT_PARSING = "document_parsing"
    CHAPTER_SPLITTING = "chapter_splitting"
    CONTEXT_BUILDING = "context_building"
    TRANSLATION = "translation"
    QUALITY_CONTROL = "quality_control"
    OUTPUT_GENERATION = "output_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class CoordinatorAgent(BaseAgent):
    """协调智能体"""
    
    def __init__(self, config, openai_client):
        super().__init__(config, openai_client)
        
        # 工作流管理
        self.current_workflow = None
        self.workflow_stages = [
            WorkflowStage.DOCUMENT_PARSING,
            WorkflowStage.CHAPTER_SPLITTING,
            WorkflowStage.CONTEXT_BUILDING,
            WorkflowStage.TRANSLATION,
            WorkflowStage.QUALITY_CONTROL,
            WorkflowStage.OUTPUT_GENERATION,
            WorkflowStage.COMPLETED
        ]
        
        # 任务管理
        self.active_tasks: Dict[str, TranslationTask] = {}
        self.completed_tasks: Dict[str, TranslationTask] = {}
        self.failed_tasks: Dict[str, TranslationTask] = {}
        
        # 智能体管理
        self.agents: Dict[AgentType, BaseAgent] = {}
        self.agent_status: Dict[AgentType, Dict[str, Any]] = {}
        
        # 质量控制配置
        self.quality_thresholds = {
            "translation_quality": 7.0,
            "consistency_score": 7.0,
            "cultural_appropriateness": 7.0,
            "overall_quality": 7.0
        }
        
        # 并发控制
        self.max_concurrent_translations = 3
        self.current_concurrent_count = 0
        
        # 重试配置
        self.max_retries = 3
        self.retry_delay = 5  # 秒
        
    async def initialize(self) -> bool:
        """初始化协调智能体"""
        try:
            logger.info("初始化协调智能体...")
            
            # 健康检查
            health_ok = await self.health_check()
            if not health_ok:
                logger.error("协调智能体健康检查失败")
                return False
            
            # 初始化工作流监控
            await self.setup_workflow_monitoring()
            
            logger.info("协调智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"协调智能体初始化失败: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理协调智能体资源"""
        try:
            logger.info("清理协调智能体资源...")
            
            # 停止所有活动任务
            for task_id, task in self.active_tasks.items():
                logger.info(f"停止任务: {task_id}")
                task.status = ProcessingStatus.FAILED
                task.error_message = "系统关闭"
            
            # 清理智能体
            for agent_type, agent in self.agents.items():
                await agent.cleanup()
            
            return True
        except Exception as e:
            logger.error(f"清理协调智能体资源失败: {e}")
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理消息"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type == "start_translation":
                result = await self.start_translation_workflow(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="translation_started",
                    content=result
                )
            
            elif message_type == "check_progress":
                result = await self.check_translation_progress(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="progress_report",
                    content=result
                )
            
            elif message_type == "cancel_translation":
                result = await self.cancel_translation_task(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="translation_cancelled",
                    content=result
                )
            
            elif message_type == "get_task_status":
                result = await self.get_task_status(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="task_status",
                    content=result
                )
            
            elif message_type == "agent_status_update":
                result = await self.handle_agent_status_update(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="status_acknowledged",
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
    
    async def register_agent(self, agent_type: AgentType, agent: BaseAgent):
        """注册智能体"""
        try:
            self.agents[agent_type] = agent
            self.agent_status[agent_type] = {
                "status": "registered",
                "last_heartbeat": datetime.now().isoformat(),
                "health": "unknown"
            }
            logger.info(f"智能体已注册: {agent_type}")
            
        except Exception as e:
            logger.error(f"注册智能体失败: {e}")
            raise
    
    async def start_translation_workflow(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """启动翻译工作流"""
        try:
            self.current_task = "启动翻译工作流"
            
            # 创建翻译任务
            task = TranslationTask(
                id=f"task_{datetime.now().timestamp()}",
                novel_id=request_data.get("novel_id", ""),
                source_language=request_data.get("source_language", "zh"),
                target_language=request_data.get("target_language", "en"),
                chapters_to_translate=request_data.get("chapters", []),
                status=ProcessingStatus.PENDING
            )
            
            # 添加到活动任务
            self.active_tasks[task.id] = task
            
            # 启动工作流
            workflow_success = await self.execute_workflow(task)
            
            if workflow_success:
                logger.info(f"翻译工作流启动成功: {task.id}")
                return {
                    "task_id": task.id,
                    "status": "started",
                    "estimated_completion": await self.estimate_completion_time(task)
                }
            else:
                logger.error(f"翻译工作流启动失败: {task.id}")
                task.status = ProcessingStatus.FAILED
                self.failed_tasks[task.id] = task
                del self.active_tasks[task.id]
                
                return {
                    "task_id": task.id,
                    "status": "failed",
                    "error": "工作流启动失败"
                }
                
        except Exception as e:
            logger.error(f"启动翻译工作流失败: {e}")
            raise
    
    async def execute_workflow(self, task: TranslationTask) -> bool:
        """执行工作流"""
        try:
            logger.info(f"开始执行工作流: {task.id}")
            
            # 更新任务状态
            task.status = ProcessingStatus.PROCESSING
            task.updated_at = datetime.now()
            
            # 按阶段执行工作流
            for stage in self.workflow_stages:
                if stage == WorkflowStage.COMPLETED:
                    break
                
                logger.info(f"执行工作流阶段: {stage}")
                
                success = await self.execute_workflow_stage(task, stage)
                
                if not success:
                    logger.error(f"工作流阶段执行失败: {stage}")
                    task.status = ProcessingStatus.FAILED
                    task.error_message = f"阶段 {stage} 执行失败"
                    return False
                
                # 更新进度
                stage_index = self.workflow_stages.index(stage)
                task.progress = (stage_index + 1) / len(self.workflow_stages)
            
            # 工作流完成
            task.status = ProcessingStatus.COMPLETED
            task.progress = 1.0
            task.updated_at = datetime.now()
            
            # 移动到完成任务
            self.completed_tasks[task.id] = task
            del self.active_tasks[task.id]
            
            logger.info(f"工作流执行完成: {task.id}")
            return True
            
        except Exception as e:
            logger.error(f"执行工作流失败: {e}")
            task.status = ProcessingStatus.FAILED
            task.error_message = str(e)
            return False
    
    async def execute_workflow_stage(self, task: TranslationTask, stage: WorkflowStage) -> bool:
        """执行工作流阶段"""
        try:
            if stage == WorkflowStage.DOCUMENT_PARSING:
                return await self.stage_document_parsing(task)
            
            elif stage == WorkflowStage.CHAPTER_SPLITTING:
                return await self.stage_chapter_splitting(task)
            
            elif stage == WorkflowStage.CONTEXT_BUILDING:
                return await self.stage_context_building(task)
            
            elif stage == WorkflowStage.TRANSLATION:
                return await self.stage_translation(task)
            
            elif stage == WorkflowStage.QUALITY_CONTROL:
                return await self.stage_quality_control(task)
            
            elif stage == WorkflowStage.OUTPUT_GENERATION:
                return await self.stage_output_generation(task)
            
            else:
                logger.warning(f"未知的工作流阶段: {stage}")
                return False
                
        except Exception as e:
            logger.error(f"执行工作流阶段失败 [{stage}]: {e}")
            return False
    
    async def stage_document_parsing(self, task: TranslationTask) -> bool:
        """文档解析阶段"""
        try:
            # 向解析智能体发送解析请求
            parser_agent = self.agents.get(AgentType.PARSER)
            if not parser_agent:
                logger.error("解析智能体未注册")
                return False
            
            # 发送解析请求
            parse_request = {
                "task_id": task.id,
                "novel_id": task.novel_id,
                "file_path": f"input/{task.novel_id}.txt"  # 假设文件路径
            }
            
            # 这里应该通过消息队列发送
            # 暂时直接调用
            logger.info(f"开始文档解析: {task.id}")
            
            # 模拟解析成功
            return True
            
        except Exception as e:
            logger.error(f"文档解析阶段失败: {e}")
            return False
    
    async def stage_chapter_splitting(self, task: TranslationTask) -> bool:
        """章节切分阶段"""
        try:
            # 向解析智能体发送章节切分请求
            logger.info(f"开始章节切分: {task.id}")
            
            # 模拟切分成功
            return True
            
        except Exception as e:
            logger.error(f"章节切分阶段失败: {e}")
            return False
    
    async def stage_context_building(self, task: TranslationTask) -> bool:
        """上下文构建阶段"""
        try:
            # 向记忆智能体发送上下文构建请求
            logger.info(f"开始上下文构建: {task.id}")
            
            # 模拟构建成功
            return True
            
        except Exception as e:
            logger.error(f"上下文构建阶段失败: {e}")
            return False
    
    async def stage_translation(self, task: TranslationTask) -> bool:
        """翻译阶段"""
        try:
            # 检查并发限制
            if self.current_concurrent_count >= self.max_concurrent_translations:
                logger.info(f"等待并发槽位: {task.id}")
                while self.current_concurrent_count >= self.max_concurrent_translations:
                    await asyncio.sleep(1)
            
            self.current_concurrent_count += 1
            
            try:
                # 向翻译智能体发送翻译请求
                logger.info(f"开始翻译: {task.id}")
                
                # 模拟翻译成功
                success = True
                
                return success
                
            finally:
                self.current_concurrent_count -= 1
            
        except Exception as e:
            logger.error(f"翻译阶段失败: {e}")
            return False
    
    async def stage_quality_control(self, task: TranslationTask) -> bool:
        """质量控制阶段"""
        try:
            logger.info(f"开始质量控制: {task.id}")
            
            # 执行质量检查
            quality_passed = await self.perform_quality_checks(task)
            
            if not quality_passed:
                # 质量不合格，尝试重新翻译
                retry_count = 0
                while retry_count < self.max_retries and not quality_passed:
                    logger.info(f"质量不合格，重新翻译 (尝试 {retry_count + 1}/{self.max_retries}): {task.id}")
                    
                    # 等待重试延迟
                    await asyncio.sleep(self.retry_delay)
                    
                    # 重新翻译
                    translation_success = await self.stage_translation(task)
                    if translation_success:
                        quality_passed = await self.perform_quality_checks(task)
                    
                    retry_count += 1
                
                if not quality_passed:
                    logger.error(f"质量控制失败，已达到最大重试次数: {task.id}")
                    return False
            
            logger.info(f"质量控制通过: {task.id}")
            return True
            
        except Exception as e:
            logger.error(f"质量控制阶段失败: {e}")
            return False
    
    async def stage_output_generation(self, task: TranslationTask) -> bool:
        """输出生成阶段"""
        try:
            logger.info(f"开始输出生成: {task.id}")
            
            # 生成输出文件
            output_success = await self.generate_output_files(task)
            
            return output_success
            
        except Exception as e:
            logger.error(f"输出生成阶段失败: {e}")
            return False
    
    async def perform_quality_checks(self, task: TranslationTask) -> bool:
        """执行质量检查"""
        try:
            # 调用各种质量检查
            quality_scores = {}
            
            # 翻译质量检查
            quality_scores["translation_quality"] = await self.check_translation_quality(task)
            
            # 一致性检查
            quality_scores["consistency_score"] = await self.check_consistency(task)
            
            # 文化适配检查
            quality_scores["cultural_appropriateness"] = await self.check_cultural_appropriateness(task)
            
            # 计算总体质量
            overall_quality = sum(quality_scores.values()) / len(quality_scores)
            quality_scores["overall_quality"] = overall_quality
            
            # 检查是否达到阈值
            for metric, score in quality_scores.items():
                threshold = self.quality_thresholds.get(metric, 7.0)
                if score < threshold:
                    logger.warning(f"质量检查未通过 [{metric}]: {score:.2f} < {threshold}")
                    return False
            
            logger.info(f"质量检查通过: {quality_scores}")
            return True
            
        except Exception as e:
            logger.error(f"执行质量检查失败: {e}")
            return False
    
    async def check_translation_quality(self, task: TranslationTask) -> float:
        """检查翻译质量"""
        try:
            # 这里应该调用翻译智能体的质量检查功能
            # 模拟返回质量分数
            return 8.0
        except Exception as e:
            logger.error(f"检查翻译质量失败: {e}")
            return 5.0
    
    async def check_consistency(self, task: TranslationTask) -> float:
        """检查一致性"""
        try:
            # 这里应该调用记忆智能体的一致性检查功能
            # 模拟返回一致性分数
            return 8.0
        except Exception as e:
            logger.error(f"检查一致性失败: {e}")
            return 5.0
    
    async def check_cultural_appropriateness(self, task: TranslationTask) -> float:
        """检查文化适配度"""
        try:
            # 这里应该调用翻译智能体的文化适配检查功能
            # 模拟返回文化适配分数
            return 8.0
        except Exception as e:
            logger.error(f"检查文化适配度失败: {e}")
            return 5.0
    
    async def generate_output_files(self, task: TranslationTask) -> bool:
        """生成输出文件"""
        try:
            # 生成翻译后的章节文件
            logger.info(f"生成输出文件: {task.id}")
            
            # 这里应该实现实际的文件生成逻辑
            # 模拟生成成功
            return True
            
        except Exception as e:
            logger.error(f"生成输出文件失败: {e}")
            return False
    
    async def check_translation_progress(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """检查翻译进度"""
        try:
            task_id = request_data.get("task_id", "")
            
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": task.status,
                    "progress": task.progress,
                    "current_stage": await self.get_current_stage(task),
                    "estimated_completion": await self.estimate_remaining_time(task)
                }
            
            elif task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": task.status,
                    "progress": 1.0,
                    "completed_at": task.updated_at.isoformat(),
                    "output_files": await self.get_output_files(task)
                }
            
            elif task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": task.status,
                    "error": task.error_message,
                    "failed_at": task.updated_at.isoformat()
                }
            
            else:
                return {
                    "task_id": task_id,
                    "status": "not_found",
                    "error": "任务不存在"
                }
                
        except Exception as e:
            logger.error(f"检查翻译进度失败: {e}")
            return {"error": str(e)}
    
    async def cancel_translation_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """取消翻译任务"""
        try:
            task_id = request_data.get("task_id", "")
            
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = ProcessingStatus.FAILED
                task.error_message = "用户取消"
                task.updated_at = datetime.now()
                
                # 移动到失败任务
                self.failed_tasks[task_id] = task
                del self.active_tasks[task_id]
                
                logger.info(f"翻译任务已取消: {task_id}")
                
                return {
                    "task_id": task_id,
                    "status": "cancelled",
                    "message": "任务已成功取消"
                }
            
            else:
                return {
                    "task_id": task_id,
                    "status": "not_found",
                    "error": "任务不存在或已完成"
                }
                
        except Exception as e:
            logger.error(f"取消翻译任务失败: {e}")
            return {"error": str(e)}
    
    async def get_task_status(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取任务状态"""
        try:
            # 返回所有任务的状态概览
            return {
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "total_tasks": len(self.active_tasks) + len(self.completed_tasks) + len(self.failed_tasks),
                "agent_status": await self.get_all_agent_status()
            }
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            return {"error": str(e)}
    
    async def handle_agent_status_update(self, status_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理智能体状态更新"""
        try:
            agent_type = status_data.get("agent_type", "")
            status_info = status_data.get("status", {})
            
            if agent_type in self.agent_status:
                self.agent_status[agent_type].update(status_info)
                self.agent_status[agent_type]["last_update"] = datetime.now().isoformat()
                
                logger.debug(f"智能体状态已更新: {agent_type}")
                
                return {
                    "agent_type": agent_type,
                    "status": "updated",
                    "message": "状态更新成功"
                }
            
            else:
                return {
                    "agent_type": agent_type,
                    "status": "not_found",
                    "error": "智能体未注册"
                }
                
        except Exception as e:
            logger.error(f"处理智能体状态更新失败: {e}")
            return {"error": str(e)}
    
    async def setup_workflow_monitoring(self):
        """设置工作流监控"""
        try:
            # 启动监控任务
            asyncio.create_task(self.monitor_workflow())
            logger.info("工作流监控已启动")
            
        except Exception as e:
            logger.error(f"设置工作流监控失败: {e}")
    
    async def monitor_workflow(self):
        """监控工作流执行"""
        try:
            while True:
                # 检查超时任务
                await self.check_timeout_tasks()
                
                # 检查智能体健康状态
                await self.check_agent_health()
                
                # 清理已完成的任务
                await self.cleanup_old_tasks()
                
                # 等待下一次检查
                await asyncio.sleep(30)  # 30秒检查一次
                
        except Exception as e:
            logger.error(f"工作流监控失败: {e}")
    
    async def check_timeout_tasks(self):
        """检查超时任务"""
        try:
            current_time = datetime.now()
            timeout_threshold = 3600  # 1小时超时
            
            for task_id, task in list(self.active_tasks.items()):
                if (current_time - task.updated_at).total_seconds() > timeout_threshold:
                    logger.warning(f"任务超时: {task_id}")
                    
                    task.status = ProcessingStatus.FAILED
                    task.error_message = "任务超时"
                    task.updated_at = current_time
                    
                    # 移动到失败任务
                    self.failed_tasks[task_id] = task
                    del self.active_tasks[task_id]
            
        except Exception as e:
            logger.error(f"检查超时任务失败: {e}")
    
    async def check_agent_health(self):
        """检查智能体健康状态"""
        try:
            for agent_type, agent in self.agents.items():
                try:
                    health_ok = await agent.health_check()
                    self.agent_status[agent_type]["health"] = "healthy" if health_ok else "unhealthy"
                    self.agent_status[agent_type]["last_health_check"] = datetime.now().isoformat()
                    
                except Exception as e:
                    logger.error(f"智能体健康检查失败 [{agent_type}]: {e}")
                    self.agent_status[agent_type]["health"] = "error"
                    self.agent_status[agent_type]["last_error"] = str(e)
            
        except Exception as e:
            logger.error(f"检查智能体健康状态失败: {e}")
    
    async def cleanup_old_tasks(self):
        """清理旧任务"""
        try:
            current_time = datetime.now()
            cleanup_threshold = 86400  # 24小时
            
            # 清理旧的已完成任务
            for task_id, task in list(self.completed_tasks.items()):
                if (current_time - task.updated_at).total_seconds() > cleanup_threshold:
                    logger.info(f"清理旧任务: {task_id}")
                    del self.completed_tasks[task_id]
            
            # 清理旧的失败任务
            for task_id, task in list(self.failed_tasks.items()):
                if (current_time - task.updated_at).total_seconds() > cleanup_threshold:
                    logger.info(f"清理失败任务: {task_id}")
                    del self.failed_tasks[task_id]
            
        except Exception as e:
            logger.error(f"清理旧任务失败: {e}")
    
    # 辅助方法
    async def estimate_completion_time(self, task: TranslationTask) -> str:
        """估算完成时间"""
        # 这里可以根据任务复杂度估算时间
        return "约30分钟"
    
    async def get_current_stage(self, task: TranslationTask) -> str:
        """获取当前阶段"""
        # 根据任务进度确定当前阶段
        stage_index = int(task.progress * len(self.workflow_stages))
        if stage_index < len(self.workflow_stages):
            return self.workflow_stages[stage_index]
        return WorkflowStage.COMPLETED
    
    async def estimate_remaining_time(self, task: TranslationTask) -> str:
        """估算剩余时间"""
        # 根据当前进度估算剩余时间
        remaining_progress = 1.0 - task.progress
        estimated_remaining = int(remaining_progress * 30)  # 假设总共30分钟
        return f"约{estimated_remaining}分钟"
    
    async def get_output_files(self, task: TranslationTask) -> List[str]:
        """获取输出文件列表"""
        # 返回生成的输出文件列表
        return [
            f"output/{task.novel_id}_translated_{task.target_language}.txt",
            f"output/{task.novel_id}_metadata_{task.target_language}.json"
        ]
    
    async def get_all_agent_status(self) -> Dict[str, Any]:
        """获取所有智能体状态"""
        return {
            agent_type: status
            for agent_type, status in self.agent_status.items()
        } 