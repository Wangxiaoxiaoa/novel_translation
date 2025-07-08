"""
小说翻译修改器 - 源代码包
Novel Translation Modifier - Source Package
"""

__version__ = "1.0.0"
__author__ = "Novel Translation Team"
__description__ = "AI-powered novel translation system with multi-agent architecture"

# 导入主要组件
from .models.base import (
    Novel, Chapter, Character, Location, Item, Terminology,
    TranslationTask, TranslationContext, QualityMetrics,
    LanguageCode, CultureType, ProcessingStatus, AgentType
)

from .agents.base_agent import BaseAgent
from .agents.parser_agent import ParserAgent
from .agents.translator_agent import TranslatorAgent
from .agents.memory_agent import MemoryAgent
from .agents.coordinator_agent import CoordinatorAgent

from .core.document_processor import DocumentProcessor

__all__ = [
    # 数据模型
    "Novel", "Chapter", "Character", "Location", "Item", "Terminology",
    "TranslationTask", "TranslationContext", "QualityMetrics",
    "LanguageCode", "CultureType", "ProcessingStatus", "AgentType",
    
    # 智能体
    "BaseAgent", "ParserAgent", "TranslatorAgent", "MemoryAgent", "CoordinatorAgent",
    
    # 核心组件
    "DocumentProcessor",
    
    # 版本信息
    "__version__", "__author__", "__description__"
] 