#!/usr/bin/env python3
"""
小说翻译修改器主应用程序
Novel Translation Modifier Main Application
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import yaml
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# 导入自定义模块
from src.models.base import (
    SystemConfig, AgentConfig, AgentType, LanguageCode, 
    TranslationTask, ProcessingStatus
)
from src.agents.enhanced_parser_agent import EnhancedParserAgent
from src.agents.enhanced_translator_agent import EnhancedTranslatorAgent
from src.agents.enhanced_memory_agent import EnhancedMemoryAgent
from src.agents.editor_agent import EditorAgent
from src.agents.quality_agent import QualityAgent
from src.agents.super_intelligent_agent import SuperIntelligentAgent
from src.agents.coordinator_agent import CoordinatorAgent
from src.core.document_processor import DocumentProcessor
from src.agents.ai_reasoning_engine import AIReasoningEngine
from src.core.deep_learning_engine import DeepLearningEngine
from src.core.creative_thinking_engine import CreativeThinkingEngine
from src.core.expert_system import ExpertSystem


class NovelTranslationModifier:
    """小说翻译修改器主类 - 集成超级AI能力"""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
        self.console = Console()
        
        # 超级智能体系统
        self.super_intelligent_agent: Optional[SuperIntelligentAgent] = None
        
        # 增强智能体实例
        self.enhanced_parser_agent: Optional[EnhancedParserAgent] = None
        self.enhanced_translator_agent: Optional[EnhancedTranslatorAgent] = None
        self.enhanced_memory_agent: Optional[EnhancedMemoryAgent] = None
        self.editor_agent: Optional[EditorAgent] = None
        self.quality_agent: Optional[QualityAgent] = None
        self.coordinator_agent: Optional[CoordinatorAgent] = None
        
        # 核心AI引擎
        self.ai_reasoning_engine: Optional[AIReasoningEngine] = None
        self.deep_learning_engine: Optional[DeepLearningEngine] = None
        self.creative_thinking_engine: Optional[CreativeThinkingEngine] = None
        self.expert_system: Optional[ExpertSystem] = None
        
        # 文档处理器
        self.document_processor: Optional[DocumentProcessor] = None
        
        # OpenAI客户端
        self.openai_client: Optional[AsyncOpenAI] = None
        
        # 系统状态
        self.is_initialized = False
        self.is_running = False
        self.intelligence_level = "basic"  # basic, enhanced, super
        
    async def initialize(self) -> bool:
        """初始化系统"""
        try:
            self.console.print(Panel.fit("🚀 初始化小说翻译修改器", style="bold blue"))
            
            # 加载配置
            if not await self.load_config():
                return False
            
            # 初始化日志
            await self.setup_logging()
            
            # 初始化OpenAI客户端
            await self.setup_openai_client()
            
            # 初始化智能体
            if not await self.initialize_agents():
                return False
            
            # 初始化文档处理器
            if not await self.initialize_document_processor():
                return False
            
            # 注册智能体到协调器
            await self.register_agents()
            
            self.is_initialized = True
            self.console.print("✅ 系统初始化完成", style="bold green")
            return True
            
        except Exception as e:
            self.console.print(f"❌ 初始化失败: {e}", style="bold red")
            logger.error(f"初始化失败: {e}")
            return False
    
    async def load_config(self) -> bool:
        """加载配置文件"""
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                self.console.print(f"❌ 配置文件不存在: {config_path}", style="bold red")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            # 转换为SystemConfig对象
            self.config = self.parse_config(raw_config)
            
            self.console.print(f"✅ 配置文件加载完成: {config_path}", style="green")
            return True
            
        except Exception as e:
            self.console.print(f"❌ 加载配置失败: {e}", style="bold red")
            return False
    
    def parse_config(self, raw_config: Dict[str, Any]) -> SystemConfig:
        """解析配置"""
        try:
            # 创建智能体配置
            agent_configs = {}
            for agent_name, agent_config in raw_config.get("agents", {}).items():
                agent_configs[agent_name] = AgentConfig(
                    agent_type=AgentType(agent_name),
                    **agent_config
                )
            
            # 创建系统配置
            system_config = SystemConfig(
                openai_config=raw_config.get("openai", {}),
                translation_config=raw_config.get("translation", {}),
                document_config=raw_config.get("document", {}),
                chapter_config=raw_config.get("chapter_splitting", {}),
                agent_configs=agent_configs,
                memory_config=raw_config.get("memory", {}),
                output_config=raw_config.get("output", {}),
                quality_config=raw_config.get("quality_control", {}),
                logging_config=raw_config.get("logging", {}),
                concurrency_config=raw_config.get("concurrency", {}),
                cache_config=raw_config.get("cache", {})
            )
            
            return system_config
            
        except Exception as e:
            logger.error(f"解析配置失败: {e}")
            raise
    
    async def setup_logging(self):
        """设置日志"""
        try:
            log_config = self.config.logging_config
            
            # 配置loguru
            logger.remove()  # 移除默认处理器
            
            # 控制台日志
            if log_config.get("console", True):
                logger.add(
                    sys.stdout,
                    level=log_config.get("level", "INFO"),
                    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
                )
            
            # 文件日志
            log_file = log_config.get("file", "logs/translation.log")
            if log_file:
                # 确保日志目录存在
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                
                logger.add(
                    log_file,
                    level=log_config.get("level", "INFO"),
                    rotation="10 MB",
                    retention="7 days",
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
                )
            
            logger.info("日志系统初始化完成")
            
        except Exception as e:
            self.console.print(f"❌ 设置日志失败: {e}", style="bold red")
            raise
    
    async def setup_openai_client(self):
        """设置OpenAI客户端"""
        try:
            openai_config = self.config.openai_config
            
            self.openai_client = AsyncOpenAI(
                api_key=openai_config.get("api_key"),
                base_url=openai_config.get("base_url", "https://api.openai.com/v1")
            )
            
            # 测试连接
            try:
                models = await self.openai_client.models.list()
                self.console.print("✅ OpenAI连接测试成功", style="green")
                logger.info("OpenAI连接测试成功")
            except Exception as e:
                self.console.print(f"⚠️ OpenAI连接测试失败: {e}", style="yellow")
                logger.warning(f"OpenAI连接测试失败: {e}")
            
        except Exception as e:
            self.console.print(f"❌ 设置OpenAI客户端失败: {e}", style="bold red")
            raise
    
    async def initialize_agents(self) -> bool:
        """初始化智能体"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=self.console
            ) as progress:
                
                # 初始化超级智能体
                task = progress.add_task("初始化超级智能体...", total=None)
                super_config = self.config.agent_configs.get("super")
                if super_config:
                    self.super_intelligent_agent = SuperIntelligentAgent(super_config, self.openai_client)
                    if not await self.super_intelligent_agent.initialize():
                        self.console.print("❌ 超级智能体初始化失败", style="bold red")
                        return False
                progress.update(task, description="✅ 超级智能体初始化完成")
                
                # 初始化解析智能体
                task = progress.add_task("初始化解析智能体...", total=None)
                parser_config = self.config.agent_configs.get("parser")
                if parser_config:
                    self.enhanced_parser_agent = EnhancedParserAgent(parser_config, self.openai_client)
                    if not await self.enhanced_parser_agent.initialize():
                        self.console.print("❌ 解析智能体初始化失败", style="bold red")
                        return False
                progress.update(task, description="✅ 解析智能体初始化完成")
                
                # 初始化翻译智能体
                task = progress.add_task("初始化翻译智能体...", total=None)
                translator_config = self.config.agent_configs.get("translator")
                if translator_config:
                    self.enhanced_translator_agent = EnhancedTranslatorAgent(translator_config, self.openai_client)
                    if not await self.enhanced_translator_agent.initialize():
                        self.console.print("❌ 翻译智能体初始化失败", style="bold red")
                        return False
                progress.update(task, description="✅ 翻译智能体初始化完成")
                
                # 初始化记忆智能体
                task = progress.add_task("初始化记忆智能体...", total=None)
                memory_config = self.config.agent_configs.get("memory")
                if memory_config:
                    self.enhanced_memory_agent = EnhancedMemoryAgent(memory_config, self.openai_client)
                    if not await self.enhanced_memory_agent.initialize():
                        self.console.print("❌ 记忆智能体初始化失败", style="bold red")
                        return False
                progress.update(task, description="✅ 记忆智能体初始化完成")
                
                # 初始化编辑智能体
                task = progress.add_task("初始化编辑智能体...", total=None)
                editor_config = self.config.agent_configs.get("editor")
                if editor_config:
                    self.editor_agent = EditorAgent(editor_config, self.openai_client)
                    if not await self.editor_agent.initialize():
                        self.console.print("❌ 编辑智能体初始化失败", style="bold red")
                        return False
                progress.update(task, description="✅ 编辑智能体初始化完成")
                
                # 初始化质量控制智能体
                task = progress.add_task("初始化质量控制智能体...", total=None)
                quality_config = self.config.agent_configs.get("quality")
                if quality_config:
                    self.quality_agent = QualityAgent(quality_config, self.openai_client)
                    if not await self.quality_agent.initialize():
                        self.console.print("❌ 质量控制智能体初始化失败", style="bold red")
                        return False
                progress.update(task, description="✅ 质量控制智能体初始化完成")
                
                # 初始化核心AI引擎
                task = progress.add_task("初始化核心AI引擎...", total=None)
                await self.initialize_core_ai_engines()
                progress.update(task, description="✅ 核心AI引擎初始化完成")
                
                # 设置智能水平
                await self.set_intelligence_level()
                
                # 初始化协调智能体
                task = progress.add_task("初始化协调智能体...", total=None)
                coordinator_config = self.config.agent_configs.get("coordinator")
                if coordinator_config:
                    self.coordinator_agent = CoordinatorAgent(coordinator_config, self.openai_client)
                    if not await self.coordinator_agent.initialize():
                        self.console.print("❌ 协调智能体初始化失败", style="bold red")
                        return False
                progress.update(task, description="✅ 协调智能体初始化完成")
            
            return True
            
        except Exception as e:
            self.console.print(f"❌ 初始化智能体失败: {e}", style="bold red")
            logger.error(f"初始化智能体失败: {e}")
            return False
    
    async def initialize_core_ai_engines(self):
        """初始化核心AI引擎"""
        try:
            # 初始化AI推理引擎
            if self.config.get("ai_reasoning_engine", {}).get("enabled", False):
                self.ai_reasoning_engine = AIReasoningEngine(self.config.get("ai_reasoning_engine", {}))
                await self.ai_reasoning_engine.initialize()
            
            # 初始化深度学习引擎
            if self.config.get("deep_learning_engine", {}).get("enabled", False):
                self.deep_learning_engine = DeepLearningEngine(self.config.get("deep_learning_engine", {}))
                await self.deep_learning_engine.initialize()
            
            # 初始化创造性思维引擎
            if self.config.get("creative_thinking_engine", {}).get("enabled", False):
                self.creative_thinking_engine = CreativeThinkingEngine(self.config.get("creative_thinking_engine", {}))
                await self.creative_thinking_engine.initialize()
            
            # 初始化专家系统
            if self.config.get("expert_system", {}).get("enabled", False):
                self.expert_system = ExpertSystem(self.config.get("expert_system", {}))
                await self.expert_system.initialize()
            
            logger.info("核心AI引擎初始化完成")
            
        except Exception as e:
            logger.error(f"初始化核心AI引擎失败: {e}")
    
    async def set_intelligence_level(self):
        """设置智能水平"""
        try:
            # 根据配置和可用组件确定智能水平
            if self.super_intelligent_agent:
                self.intelligence_level = "super"
                self.console.print("🧠 启用超级智能模式", style="bold magenta")
            elif all([self.enhanced_parser_agent, self.enhanced_translator_agent, 
                     self.enhanced_memory_agent, self.editor_agent, self.quality_agent]):
                self.intelligence_level = "enhanced"
                self.console.print("🤖 启用增强智能模式", style="bold blue")
            else:
                self.intelligence_level = "basic"
                self.console.print("📝 使用基础智能模式", style="bold green")
            
        except Exception as e:
            logger.error(f"设置智能水平失败: {e}")
            self.intelligence_level = "basic"
    
    async def initialize_document_processor(self) -> bool:
        """初始化文档处理器"""
        try:
            doc_config = self.config.document_config
            self.document_processor = DocumentProcessor(doc_config)
            
            if not await self.document_processor.initialize(self.enhanced_parser_agent, self.enhanced_memory_agent):
                self.console.print("❌ 文档处理器初始化失败", style="bold red")
                return False
            
            self.console.print("✅ 文档处理器初始化完成", style="green")
            return True
            
        except Exception as e:
            self.console.print(f"❌ 初始化文档处理器失败: {e}", style="bold red")
            logger.error(f"初始化文档处理器失败: {e}")
            return False
    
    async def register_agents(self):
        """注册智能体到协调器"""
        try:
            if self.coordinator_agent:
                # 注册基础智能体
                if self.enhanced_parser_agent:
                    await self.coordinator_agent.register_agent(AgentType.PARSER, self.enhanced_parser_agent)
                if self.enhanced_translator_agent:
                    await self.coordinator_agent.register_agent(AgentType.TRANSLATOR, self.enhanced_translator_agent)
                if self.enhanced_memory_agent:
                    await self.coordinator_agent.register_agent(AgentType.MEMORY, self.enhanced_memory_agent)
                if self.editor_agent:
                    await self.coordinator_agent.register_agent(AgentType.EDITOR, self.editor_agent)
                if self.quality_agent:
                    await self.coordinator_agent.register_agent(AgentType.QUALITY, self.quality_agent)
                
                # 注册超级智能体（如果启用）
                if self.super_intelligent_agent:
                    await self.coordinator_agent.register_agent(AgentType.SUPER, self.super_intelligent_agent)
                    self.console.print("🧠 超级智能体已注册", style="bold magenta")
                
                self.console.print("✅ 智能体注册完成", style="green")
                
        except Exception as e:
            self.console.print(f"❌ 注册智能体失败: {e}", style="bold red")
            logger.error(f"注册智能体失败: {e}")
    
    async def translate_novel(self, input_file: str, target_language: str, **kwargs) -> bool:
        """翻译小说 - 支持多种智能水平"""
        try:
            if not self.is_initialized:
                self.console.print("❌ 系统未初始化", style="bold red")
                return False
            
            # 显示翻译开始信息
            intelligence_emoji = {"basic": "📝", "enhanced": "🤖", "super": "🧠"}
            self.console.print(Panel.fit(
                f"{intelligence_emoji[self.intelligence_level]} 开始翻译小说 ({self.intelligence_level.upper()}模式)\n"
                f"输入文件: {input_file}\n"
                f"目标语言: {target_language}\n"
                f"智能水平: {self.intelligence_level}",
                style="bold blue"
            ))
            
            # 根据智能水平选择处理方式
            if self.intelligence_level == "super":
                return await self.super_intelligent_translate(input_file, target_language, **kwargs)
            elif self.intelligence_level == "enhanced":
                return await self.enhanced_translate(input_file, target_language, **kwargs)
            else:
                return await self.basic_translate(input_file, target_language, **kwargs)
                
        except Exception as e:
            self.console.print(f"❌ 翻译失败: {e}", style="bold red")
            logger.error(f"翻译失败: {e}")
            return False
    
    async def super_intelligent_translate(self, input_file: str, target_language: str, **kwargs) -> bool:
        """超级智能翻译模式"""
        try:
            self.console.print("🧠 启动超级智能翻译流程...", style="bold magenta")
            
            # 使用超级智能体进行翻译
            super_task = {
                "task_type": "novel_translation",
                "description": f"使用超级AI翻译小说到{target_language}",
                "input_data": {
                    "input_file": input_file,
                    "target_language": target_language,
                    **kwargs
                },
                "requirements": [
                    "高质量翻译",
                    "文化深度适配", 
                    "创造性表达",
                    "专家级质量",
                    "智能编辑优化"
                ],
                "constraints": [
                    "保持原文意思",
                    "维护故事结构",
                    "确保文化适宜"
                ],
                "success_criteria": [
                    "翻译准确性 > 9.0",
                    "文化适配度 > 8.5",
                    "创造性指数 > 8.0",
                    "整体质量 > 9.0"
                ],
                "priority": 10
            }
            
            # 调用超级智能体
            from src.agents.super_intelligent_agent import SuperIntelligenceTask
            message = AgentMessage(
                sender=AgentType.COORDINATOR,
                receiver=AgentType.SUPER,
                message_type="super_intelligent_processing",
                content=super_task
            )
            
            result_message = await self.super_intelligent_agent.process_message(message)
            
            if result_message.message_type == "error":
                self.console.print(f"❌ 超级智能翻译失败: {result_message.content.get('error')}", style="bold red")
                return False
            
            result = result_message.content
            
            # 显示超级智能翻译结果
            self.console.print(f"✨ 超级智能翻译完成!", style="bold magenta")
            self.console.print(f"  📊 置信度: {result.get('confidence_score', 0):.2f}")
            self.console.print(f"  🎯 质量评分: {result.get('quality_metrics', {}).get('overall', 0):.2f}")
            self.console.print(f"  🧠 推理深度: {len(result.get('reasoning_trace', []))} 层")
            self.console.print(f"  💡 创意方案: {len(result.get('alternative_solutions', []))} 个")
            
            return True
            
        except Exception as e:
            logger.error(f"超级智能翻译失败: {e}")
            return False
    
    async def enhanced_translate(self, input_file: str, target_language: str, **kwargs) -> bool:
        """增强智能翻译模式"""
        try:
            self.console.print("🤖 启动增强智能翻译流程...", style="bold blue")
            
            # 使用增强智能体协同翻译
            # 处理文档
            novel = await self.document_processor.process_document(input_file, **kwargs)
            
            # 启动增强翻译工作流
            task_data = {
                "novel_id": novel.title,
                "source_language": novel.language,
                "target_language": target_language,
                "chapters": [chapter.id for chapter in novel.chapters],
                "enhancement_level": "advanced"
            }
            
            # 调用协调智能体启动翻译
            translation_result = await self.coordinator_agent.start_translation_workflow(task_data)
            
            if translation_result.get("status") == "started":
                task_id = translation_result.get("task_id")
                self.console.print(f"🤖 增强翻译任务启动成功: {task_id}", style="blue")
                
                # 监控翻译进度
                await self.monitor_translation_progress(task_id)
                
                return True
            else:
                self.console.print(f"❌ 增强翻译任务启动失败: {translation_result.get('error')}", style="bold red")
                return False
                
        except Exception as e:
            logger.error(f"增强智能翻译失败: {e}")
            return False
    
    async def basic_translate(self, input_file: str, target_language: str, **kwargs) -> bool:
        """基础翻译模式"""
        try:
            self.console.print("📝 启动基础翻译流程...", style="bold green")
            
            # 使用基础翻译流程
            # 处理文档
            novel = await self.document_processor.process_document(input_file, **kwargs)
            
            # 启动基础翻译工作流
            task_data = {
                "novel_id": novel.title,
                "source_language": novel.language,
                "target_language": target_language,
                "chapters": [chapter.id for chapter in novel.chapters]
            }
            
            # 调用协调智能体启动翻译
            translation_result = await self.coordinator_agent.start_translation_workflow(task_data)
            
            if translation_result.get("status") == "started":
                task_id = translation_result.get("task_id")
                self.console.print(f"📝 基础翻译任务启动成功: {task_id}", style="green")
                
                # 监控翻译进度
                await self.monitor_translation_progress(task_id)
                
                return True
            else:
                self.console.print(f"❌ 基础翻译任务启动失败: {translation_result.get('error')}", style="bold red")
                return False
                
        except Exception as e:
            logger.error(f"基础翻译失败: {e}")
            return False
    
    async def monitor_translation_progress(self, task_id: str):
        """监控翻译进度"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            ) as progress:
                
                task = progress.add_task("翻译进行中...", total=100)
                
                while True:
                    # 检查进度
                    progress_data = await self.coordinator_agent.check_translation_progress({"task_id": task_id})
                    
                    status = progress_data.get("status")
                    current_progress = progress_data.get("progress", 0) * 100
                    
                    progress.update(task, completed=current_progress)
                    
                    if status == ProcessingStatus.COMPLETED:
                        progress.update(task, description="✅ 翻译完成")
                        break
                    elif status == ProcessingStatus.FAILED:
                        progress.update(task, description="❌ 翻译失败")
                        break
                    
                    await asyncio.sleep(5)  # 每5秒检查一次
                
                # 显示最终结果
                if status == ProcessingStatus.COMPLETED:
                    self.console.print("🎉 翻译任务完成!", style="bold green")
                    
                    # 显示输出文件
                    output_files = progress_data.get("output_files", [])
                    if output_files:
                        self.console.print("📄 输出文件:", style="bold")
                        for file_path in output_files:
                            self.console.print(f"  - {file_path}", style="cyan")
                
                elif status == ProcessingStatus.FAILED:
                    error_msg = progress_data.get("error", "未知错误")
                    self.console.print(f"💥 翻译失败: {error_msg}", style="bold red")
                
        except Exception as e:
            self.console.print(f"❌ 监控翻译进度失败: {e}", style="bold red")
            logger.error(f"监控翻译进度失败: {e}")
    
    async def show_system_status(self):
        """显示系统状态"""
        try:
            if not self.is_initialized:
                self.console.print("❌ 系统未初始化", style="bold red")
                return
            
            # 获取系统状态
            task_status = await self.coordinator_agent.get_task_status({})
            
            # 创建状态表格
            table = Table(title="系统状态")
            table.add_column("项目", style="cyan")
            table.add_column("状态", style="magenta")
            table.add_column("详情", style="green")
            
            # 系统状态
            table.add_row("系统状态", "✅ 正常运行", "所有组件已初始化")
            
            # 任务状态
            table.add_row("活动任务", str(task_status.get("active_tasks", 0)), "")
            table.add_row("已完成任务", str(task_status.get("completed_tasks", 0)), "")
            table.add_row("失败任务", str(task_status.get("failed_tasks", 0)), "")
            
            # 智能体状态
            agent_status = task_status.get("agent_status", {})
            for agent_type, status in agent_status.items():
                health = status.get("health", "unknown")
                health_emoji = "✅" if health == "healthy" else "❌" if health == "unhealthy" else "❓"
                table.add_row(f"{agent_type}智能体", f"{health_emoji} {health}", "")
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"❌ 显示系统状态失败: {e}", style="bold red")
            logger.error(f"显示系统状态失败: {e}")
    
    async def cleanup(self):
        """清理资源"""
        try:
            self.console.print("🧹 清理系统资源...", style="yellow")
            
            # 清理超级智能体
            if self.super_intelligent_agent:
                await self.super_intelligent_agent.cleanup()
                self.console.print("  🧠 超级智能体资源已清理", style="dim yellow")
            
            # 清理核心AI引擎
            cleanup_tasks = []
            if self.ai_reasoning_engine:
                cleanup_tasks.append(self.ai_reasoning_engine.cleanup())
            if self.deep_learning_engine:
                cleanup_tasks.append(self.deep_learning_engine.cleanup())
            if self.creative_thinking_engine:
                cleanup_tasks.append(self.creative_thinking_engine.cleanup())
            if self.expert_system:
                cleanup_tasks.append(self.expert_system.cleanup())
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                self.console.print("  🔧 核心AI引擎已清理", style="dim yellow")
            
            # 清理增强智能体
            if self.enhanced_parser_agent:
                await self.enhanced_parser_agent.cleanup()
            if self.enhanced_translator_agent:
                await self.enhanced_translator_agent.cleanup()
            if self.enhanced_memory_agent:
                await self.enhanced_memory_agent.cleanup()
            if self.editor_agent:
                await self.editor_agent.cleanup()
            if self.quality_agent:
                await self.quality_agent.cleanup()
            if self.coordinator_agent:
                await self.coordinator_agent.cleanup()
            
            self.console.print("  🤖 增强智能体已清理", style="dim yellow")
            
            # 关闭OpenAI客户端
            if self.openai_client:
                await self.openai_client.close()
            
            self.console.print("✅ 资源清理完成", style="green")
            
        except Exception as e:
            self.console.print(f"❌ 清理资源失败: {e}", style="bold red")
            logger.error(f"清理资源失败: {e}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="小说翻译修改器")
    parser.add_argument("--config", "-c", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--input", "-i", help="输入文件路径")
    parser.add_argument("--target-language", "-t", help="目标语言")
    parser.add_argument("--title", help="小说标题")
    parser.add_argument("--author", help="小说作者")
    parser.add_argument("--genre", help="小说类型")
    parser.add_argument("--status", "-s", action="store_true", help="显示系统状态")
    
    # 高级功能参数
    parser.add_argument("--super-intelligence", action="store_true", help="启用超级智能模式")
    parser.add_argument("--enhanced", action="store_true", help="启用增强智能模式")
    parser.add_argument("--professional-review", action="store_true", help="启用专业审校")
    parser.add_argument("--cultural-adaptation", action="store_true", help="启用深度文化适配")
    parser.add_argument("--quality-mode", choices=["basic", "premium", "professional"], 
                        default="basic", help="质量模式")
    parser.add_argument("--localization-level", choices=["minimal", "moderate", "extensive", "complete"],
                        default="moderate", help="本土化级别")
    
    args = parser.parse_args()
    
    # 创建主应用
    app = NovelTranslationModifier(args.config)
    
    try:
        # 初始化系统
        if not await app.initialize():
            return 1
        
        # 根据参数执行不同操作
        if args.status:
            await app.show_system_status()
        
        elif args.input and args.target_language:
            # 翻译小说
            kwargs = {}
            if args.title:
                kwargs["title"] = args.title
            if args.author:
                kwargs["author"] = args.author
            if args.genre:
                kwargs["genre"] = args.genre
            
            # 设置高级功能
            if args.super_intelligence:
                kwargs["intelligence_mode"] = "super"
            elif args.enhanced:
                kwargs["intelligence_mode"] = "enhanced"
            else:
                kwargs["intelligence_mode"] = "basic"
            
            kwargs["professional_review"] = args.professional_review
            kwargs["cultural_adaptation"] = args.cultural_adaptation
            kwargs["quality_mode"] = args.quality_mode
            kwargs["localization_level"] = args.localization_level
            
            success = await app.translate_novel(args.input, args.target_language, **kwargs)
            return 0 if success else 1
        
        else:
            # 显示帮助信息
            parser.print_help()
            
            # 显示支持的语言
            console = Console()
            console.print("\n🌍 支持的语言 (42种):", style="bold")
            languages = [
                # 主要语言
                ("zh", "中文 (Chinese)"),
                ("en", "英语 (English)"),
                ("ja", "日语 (Japanese)"),
                ("ko", "韩语 (Korean)"),
                ("fr", "法语 (French)"),
                ("de", "德语 (German)"),
                ("es", "西班牙语 (Spanish)"),
                ("ru", "俄语 (Russian)"),
                ("ar", "阿拉伯语 (Arabic)"),
                
                # 欧洲语言
                ("it", "意大利语 (Italian)"),
                ("pt", "葡萄牙语 (Portuguese)"),
                ("nl", "荷兰语 (Dutch)"),
                ("pl", "波兰语 (Polish)"),
                ("cs", "捷克语 (Czech)"),
                ("sk", "斯洛伐克语 (Slovak)"),
                ("hu", "匈牙利语 (Hungarian)"),
                ("ro", "罗马尼亚语 (Romanian)"),
                ("bg", "保加利亚语 (Bulgarian)"),
                ("hr", "克罗地亚语 (Croatian)"),
                ("sl", "斯洛文尼亚语 (Slovenian)"),
                ("et", "爱沙尼亚语 (Estonian)"),
                ("lv", "拉脱维亚语 (Latvian)"),
                ("lt", "立陶宛语 (Lithuanian)"),
                
                # 北欧语言
                ("sv", "瑞典语 (Swedish)"),
                ("no", "挪威语 (Norwegian)"),
                ("da", "丹麦语 (Danish)"),
                ("fi", "芬兰语 (Finnish)"),
                ("is", "冰岛语 (Icelandic)"),
                
                # 亚洲语言
                ("hi", "印地语 (Hindi)"),
                ("th", "泰语 (Thai)"),
                ("vi", "越南语 (Vietnamese)"),
                ("ms", "马来语 (Malay)"),
                ("id", "印尼语 (Indonesian)"),
                ("ta", "泰米尔语 (Tamil)"),
                ("te", "泰卢固语 (Telugu)"),
                ("bn", "孟加拉语 (Bengali)"),
                ("ur", "乌尔都语 (Urdu)"),
                ("my", "缅甸语 (Myanmar)"),
                ("km", "高棉语 (Khmer)"),
                ("lo", "老挝语 (Lao)"),
                ("tr", "土耳其语 (Turkish)"),
                ("fa", "波斯语 (Persian)"),
                ("he", "希伯来语 (Hebrew)"),
                
                # 非洲语言
                ("sw", "斯瓦希里语 (Swahili)"),
                ("am", "阿姆哈拉语 (Amharic)"),
                
                # 美洲语言
                ("pt-BR", "巴西葡萄牙语 (Brazilian Portuguese)"),
                ("es-MX", "墨西哥西班牙语 (Mexican Spanish)")
            ]
            
            # 分列显示
            for i in range(0, len(languages), 3):
                row = languages[i:i+3]
                row_text = "  ".join([f"{code:<8}: {name}" for code, name in row])
                console.print(f"  {row_text}")
            
            console.print("\n🎯 使用示例:", style="bold")
            console.print("  基础翻译:")
            console.print("    python main.py -i novel.txt -t en --title 'Novel Title' --author 'Author'")
            console.print("\n  高级翻译:")
            console.print("    python main.py -i novel.txt -t ja --super-intelligence --professional-review")
            console.print("    python main.py -i novel.txt -t ko --enhanced --cultural-adaptation")
            console.print("    python main.py -i novel.txt -t fr --quality-mode premium --localization-level extensive")
            console.print("\n  系统状态:")
            console.print("    python main.py --status")
            
            console.print("\n🚀 新功能特性:", style="bold")
            console.print("  📚 42种语言支持 (包括小语种)")
            console.print("  🧠 超级智能翻译模式")
            console.print("  👨‍🎓 专业审校系统")
            console.print("  🌍 深度文化适配")
            console.print("  📊 多层次质量控制")
            console.print("  🎨 文学类型特化")
            console.print("  🔧 自动质量改进")
        
        return 0
        
    except KeyboardInterrupt:
        app.console.print("\n🛑 用户中断", style="yellow")
        return 0
    except Exception as e:
        app.console.print(f"❌ 程序错误: {e}", style="bold red")
        logger.error(f"程序错误: {e}")
        return 1
    finally:
        await app.cleanup()


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 运行主程序
    sys.exit(asyncio.run(main())) 