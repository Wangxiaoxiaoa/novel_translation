"""
基础智能体抽象类和通用功能
Base agent abstract class and common functionality
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from datetime import datetime

from ..models.base import (
    AgentType, AgentConfig, AgentMessage, ProcessingStatus,
    Novel, Chapter, TranslationContext, QualityMetrics
)


class BaseAgent(ABC):
    """基础智能体抽象类"""
    
    def __init__(self, config: AgentConfig, openai_client: AsyncOpenAI):
        self.config = config
        self.client = openai_client
        self.agent_type = config.agent_type
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.max_retries = config.max_retries
        self.timeout = config.timeout
        self.custom_params = config.custom_params
        
        # 消息队列
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_queue: asyncio.Queue = asyncio.Queue()
        
        # 状态管理
        self.status = ProcessingStatus.PENDING
        self.current_task: Optional[str] = None
        self.error_count = 0
        self.last_error: Optional[str] = None
        
        # 性能统计
        self.processed_count = 0
        self.total_tokens_used = 0
        self.average_response_time = 0.0
        
        logger.info(f"初始化智能体: {self.agent_type}")
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理消息的抽象方法"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化智能体"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """清理智能体资源"""
        pass
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def call_llm(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """调用LLM的通用方法"""
        try:
            start_time = datetime.now()
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs
            )
            
            # 更新统计信息
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self.update_statistics(response_time, response.usage.total_tokens)
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM返回空内容")
            
            return content
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"LLM调用失败 [{self.agent_type}]: {e}")
            raise
    
    def update_statistics(self, response_time: float, tokens_used: int):
        """更新性能统计信息"""
        self.processed_count += 1
        self.total_tokens_used += tokens_used
        
        # 计算平均响应时间
        if self.processed_count == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.processed_count - 1) + response_time) 
                / self.processed_count
            )
    
    async def send_message(self, receiver: AgentType, message_type: str, content: Any) -> bool:
        """发送消息给其他智能体"""
        try:
            message = AgentMessage(
                sender=self.agent_type,
                receiver=receiver,
                message_type=message_type,
                content=content
            )
            
            # 这里应该通过消息总线发送，暂时先记录日志
            logger.info(f"发送消息: {self.agent_type} -> {receiver} [{message_type}]")
            return True
            
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """接收消息"""
        try:
            # 设置超时时间
            message = await asyncio.wait_for(
                self.message_queue.get(), 
                timeout=self.timeout
            )
            return message
        except asyncio.TimeoutError:
            logger.debug(f"智能体 {self.agent_type} 接收消息超时")
            return None
        except Exception as e:
            logger.error(f"接收消息失败: {e}")
            return None
    
    async def start_processing(self):
        """开始处理消息循环"""
        self.status = ProcessingStatus.PROCESSING
        logger.info(f"智能体 {self.agent_type} 开始处理消息")
        
        while self.status == ProcessingStatus.PROCESSING:
            try:
                message = await self.receive_message()
                if message:
                    response = await self.process_message(message)
                    if response:
                        await self.response_queue.put(response)
                        
            except Exception as e:
                logger.error(f"处理消息时发生错误 [{self.agent_type}]: {e}")
                self.error_count += 1
                self.last_error = str(e)
                
                # 如果错误次数过多，暂停处理
                if self.error_count > 5:
                    self.status = ProcessingStatus.FAILED
                    logger.error(f"智能体 {self.agent_type} 因错误过多而停止")
                    break
    
    async def stop_processing(self):
        """停止处理"""
        self.status = ProcessingStatus.COMPLETED
        logger.info(f"智能体 {self.agent_type} 停止处理")
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "agent_type": self.agent_type,
            "status": self.status,
            "current_task": self.current_task,
            "processed_count": self.processed_count,
            "total_tokens_used": self.total_tokens_used,
            "average_response_time": self.average_response_time,
            "error_count": self.error_count,
            "last_error": self.last_error
        }
    
    def load_prompt_template(self, template_name: str, **kwargs) -> str:
        """加载并格式化提示词模板"""
        try:
            # 这里应该从配置的模板目录加载
            # 暂时返回一个基础模板
            base_template = f"作为{self.agent_type}智能体，请完成以下任务：\n\n{{task}}\n\n请确保回答准确、专业且符合要求。"
            return base_template.format(**kwargs)
        except Exception as e:
            logger.error(f"加载提示词模板失败: {e}")
            return kwargs.get('task', '请完成指定任务')
    
    def validate_input(self, data: Any, expected_type: type = None) -> bool:
        """验证输入数据"""
        try:
            if expected_type and not isinstance(data, expected_type):
                logger.warning(f"输入数据类型错误，期望: {expected_type}, 实际: {type(data)}")
                return False
            return True
        except Exception as e:
            logger.error(f"验证输入数据失败: {e}")
            return False
    
    def format_output(self, data: Any) -> Dict[str, Any]:
        """格式化输出数据"""
        try:
            return {
                "agent_type": self.agent_type,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "data": data,
                "metadata": {
                    "processed_count": self.processed_count,
                    "tokens_used": self.total_tokens_used
                }
            }
        except Exception as e:
            logger.error(f"格式化输出数据失败: {e}")
            return {
                "agent_type": self.agent_type,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "data": None
            }
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 简单的健康检查：尝试调用LLM
            test_messages = [
                {"role": "system", "content": "你是一个测试智能体"},
                {"role": "user", "content": "请回复'健康检查通过'"}
            ]
            
            response = await self.call_llm(
                messages=test_messages,
                temperature=0.1,
                max_tokens=50
            )
            
            return "健康检查通过" in response
            
        except Exception as e:
            logger.error(f"健康检查失败 [{self.agent_type}]: {e}")
            return False 