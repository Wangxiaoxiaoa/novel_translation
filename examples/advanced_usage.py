#!/usr/bin/env python3
"""
高级使用示例 - 展示增强智能体的强大功能
Advanced Usage Examples - Demonstrating enhanced agent capabilities
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base import (
    SystemConfig, AgentConfig, AgentType, LanguageCode, 
    TranslationContext, Chapter, Novel
)
from src.agents.enhanced_parser_agent import EnhancedParserAgent
from src.agents.enhanced_translator_agent import EnhancedTranslatorAgent
from src.agents.enhanced_memory_agent import EnhancedMemoryAgent
from src.agents.editor_agent import EditorAgent
from src.agents.quality_agent import QualityAgent
from src.agents.coordinator_agent import CoordinatorAgent
from openai import AsyncOpenAI
import yaml


async def advanced_parsing_example():
    """高级解析示例"""
    print("=== 高级解析示例 ===")
    
    # 配置
    config = AgentConfig(
        agent_type=AgentType.PARSER,
        enabled=True,
        max_concurrent_tasks=3,
        timeout=300,
        advanced_features={
            "deep_analysis": True,
            "plot_structure_analysis": True,
            "character_network_analysis": True,
            "writing_style_analysis": True,
            "timeline_extraction": True,
            "emotion_analysis": True,
            "knowledge_graph_building": True
        }
    )
    
    # 创建OpenAI客户端
    openai_client = AsyncOpenAI(api_key="your-api-key")
    
    # 初始化增强解析智能体
    parser = EnhancedParserAgent(config, openai_client)
    await parser.initialize()
    
    # 深度解析文档
    print("开始深度解析文档...")
    parse_result = await parser.deep_parse_document("ori_novel/sample.txt")
    
    print(f"解析完成!")
    print(f"- 复杂度评分: {parse_result['deep_analysis']['complexity_score']}")
    print(f"- 可读性评分: {parse_result['deep_analysis']['readability_score']}")
    print(f"- 类型分类: {parse_result['deep_analysis']['genre_classification']}")
    print(f"- 角色网络节点数: {len(parse_result['deep_analysis']['character_network'])}")
    print(f"- 知识图谱规模: {parse_result['deep_analysis']['knowledge_graph']['statistics']}")
    
    # 智能章节切分
    print("\n开始智能章节切分...")
    chapters = await parser.intelligent_chapter_split({
        "content": parse_result["content"]
    })
    
    print(f"智能切分完成，共 {len(chapters)} 章")
    for i, chapter in enumerate(chapters[:3]):  # 显示前3章信息
        print(f"第{i+1}章: {chapter.title}")
        print(f"  - 字数: {chapter.word_count}")
        print(f"  - 角色数: {len(chapter.characters)}")
        print(f"  - 复杂度: {chapter.metadata.get('complexity', 'N/A')}")
        print(f"  - 情感弧线: {chapter.metadata.get('emotion_arc', {}).get('dominant_emotion', 'N/A')}")
    
    await parser.cleanup()
    return parse_result, chapters


async def advanced_translation_example():
    """高级翻译示例"""
    print("\n=== 高级翻译示例 ===")
    
    # 配置
    config = AgentConfig(
        agent_type=AgentType.TRANSLATOR,
        enabled=True,
        max_concurrent_tasks=2,
        timeout=600,
        advanced_features={
            "multi_strategy_translation": True,
            "deep_cultural_adaptation": True,
            "dialogue_optimization": True,
            "style_transfer": True,
            "quality_assessment": True,
            "adaptive_improvement": True,
            "translation_memory": True
        }
    )
    
    openai_client = AsyncOpenAI(api_key="your-api-key")
    
    # 初始化增强翻译智能体
    translator = EnhancedTranslatorAgent(config, openai_client)
    await translator.initialize()
    
    # 创建翻译上下文
    sample_chapter = Chapter(
        id="chapter_001",
        title="第一章：开始",
        content="这是一个示例章节内容，包含了复杂的对话和描述...",
        original_content="这是一个示例章节内容，包含了复杂的对话和描述...",
        chapter_number=1,
        word_count=100
    )
    
    context = TranslationContext(
        source_language=LanguageCode.ZH,
        target_language=LanguageCode.EN,
        target_culture="Western",
        current_chapter=sample_chapter,
        character_context={},
        location_context={},
        terminology_context={},
        plot_context="这是一个现代都市小说的开篇",
        previous_chapters=[]
    )
    
    # 高级翻译
    print("开始高级章节翻译...")
    translation_result = await translator.advanced_translate_chapter(context)
    
    print(f"翻译完成!")
    print(f"- 使用策略: {', '.join(translation_result['translation_strategies_used'])}")
    print(f"- 总体质量评分: {translation_result['quality_metrics']['overall_score']:.2f}")
    print(f"- 准确性: {translation_result['quality_metrics']['accuracy']:.2f}")
    print(f"- 流畅性: {translation_result['quality_metrics']['fluency']:.2f}")
    print(f"- 文化适配: {translation_result['quality_metrics']['cultural_adaptation']:.2f}")
    print(f"- 风格一致性: {translation_result['quality_metrics']['style_consistency']:.2f}")
    
    # 多策略翻译对比
    print("\n开始多策略翻译对比...")
    multi_strategy_result = await translator.multi_strategy_translate_internal(context)
    
    print("不同策略的翻译结果:")
    for strategy, translation in multi_strategy_result.items():
        print(f"- {strategy}: {translation[:100]}...")
    
    await translator.cleanup()
    return translation_result


async def advanced_memory_example():
    """高级记忆示例"""
    print("\n=== 高级记忆示例 ===")
    
    # 配置
    config = AgentConfig(
        agent_type=AgentType.MEMORY,
        enabled=True,
        max_concurrent_tasks=5,
        timeout=200,
        advanced_features={
            "vector_storage": True,
            "temporal_memory": True,
            "semantic_memory": True,
            "episodic_memory": True,
            "relationship_analysis": True,
            "contextual_retrieval": True,
            "memory_consolidation": True,
            "importance_calculation": True
        }
    )
    
    openai_client = AsyncOpenAI(api_key="your-api-key")
    
    # 初始化增强记忆智能体
    memory = EnhancedMemoryAgent(config, openai_client)
    await memory.initialize()
    
    # 存储综合记忆
    print("开始存储综合记忆...")
    memory_data = {
        "characters": [
            {
                "name": "李华",
                "description": "主角，年轻的程序员",
                "attributes": {"age": 25, "profession": "程序员", "personality": "内向但聪明"}
            },
            {
                "name": "王明",
                "description": "李华的好友，设计师",
                "attributes": {"age": 26, "profession": "设计师", "personality": "外向开朗"}
            }
        ],
        "locations": [
            {
                "name": "科技园",
                "description": "现代化的办公区域",
                "attributes": {"type": "商业区", "atmosphere": "繁忙"}
            }
        ],
        "terminologies": [
            {
                "term": "人工智能",
                "definition": "模拟人类智能的计算机系统",
                "attributes": {"category": "技术", "importance": "high"}
            }
        ],
        "plot_events": [
            {
                "name": "初次相遇",
                "description": "李华和王明在咖啡厅第一次见面",
                "attributes": {"importance": "high", "emotion": "neutral"}
            }
        ],
        "relationships": [
            {
                "source": "李华",
                "target": "王明", 
                "type": "友谊",
                "description": "大学同学，现在是好朋友"
            }
        ]
    }
    
    context = {"chapter_id": "chapter_001"}
    
    storage_result = await memory.store_comprehensive_memory({
        "memory_data": memory_data,
        "context": context
    })
    
    print(f"记忆存储完成!")
    print(f"- 存储节点数: {storage_result['stored_nodes']}")
    print(f"- 节点类型分布: {storage_result['node_types']}")
    print(f"- 创建关系数: {storage_result['relationships_created']}")
    print(f"- 知识图谱统计: {storage_result['knowledge_graph_stats']}")
    
    # 上下文记忆检索
    print("\n开始上下文记忆检索...")
    from src.agents.enhanced_memory_agent import MemoryQuery, MemoryType
    
    query = MemoryQuery(
        query_text="李华的基本信息和相关人物",
        query_type=MemoryType.CHARACTER,
        context_filter={"chapter_id": "chapter_001"},
        similarity_threshold=0.7,
        max_results=5
    )
    
    retrieval_result = await memory.retrieve_contextual_memory(query.__dict__)
    
    print(f"记忆检索完成!")
    print(f"- 检索结果数: {retrieval_result['result_count']}")
    print(f"- 使用检索方法: {', '.join(retrieval_result['retrieval_methods'])}")
    
    await memory.cleanup()
    return storage_result, retrieval_result


async def advanced_editing_example():
    """高级编辑示例"""
    print("\n=== 高级编辑示例 ===")
    
    # 配置
    config = AgentConfig(
        agent_type=AgentType.EDITOR,
        enabled=True,
        max_concurrent_tasks=2,
        timeout=400,
        advanced_features={
            "comprehensive_editing": True,
            "consistency_editing": True,
            "style_editing": True,
            "flow_editing": True,
            "dialogue_editing": True,
            "narrative_editing": True,
            "quality_enhancement": True,
            "version_management": True
        }
    )
    
    openai_client = AsyncOpenAI(api_key="your-api-key")
    
    # 初始化编辑智能体
    editor = EditorAgent(config, openai_client)
    await editor.initialize()
    
    # 综合编辑
    print("开始综合编辑...")
    
    sample_text = """
    这是一个需要编辑的文本示例。文本中可能存在一些一致性问题，
    风格不统一的地方，以及可以改进的对话和叙述部分。
    """
    
    edit_data = {
        "content": sample_text,
        "context": {"genre": "现代小说", "target_audience": "年轻读者"},
        "preferences": {
            "consistency_edit": True,
            "style_edit": True,
            "flow_edit": True,
            "dialogue_edit": True,
            "narrative_edit": True,
            "quality_enhance": True
        }
    }
    
    edit_result = await editor.comprehensive_edit(edit_data)
    
    print(f"编辑完成!")
    print(f"- 编辑阶段数: {len(edit_result['edit_history'])}")
    print(f"- 总修改数: {edit_result['editor_metadata']['total_changes']}")
    print(f"- 质量改进: {edit_result['improvement_score']:.1%}")
    print(f"- 最终质量评分: {edit_result['final_quality']['overall_improvement']:.1%}")
    
    print("\n编辑历史:")
    for stage in edit_result['edit_history']:
        print(f"- {stage['stage']}: {len(stage['changes'])} 处修改")
    
    await editor.cleanup()
    return edit_result


async def advanced_quality_example():
    """高级质量控制示例"""
    print("\n=== 高级质量控制示例 ===")
    
    # 配置
    config = AgentConfig(
        agent_type=AgentType.QUALITY,
        enabled=True,
        max_concurrent_tasks=3,
        timeout=300,
        advanced_features={
            "comprehensive_assessment": True,
            "accuracy_assessment": True,
            "fluency_assessment": True,
            "consistency_assessment": True,
            "cultural_assessment": True,
            "style_assessment": True,
            "readability_assessment": True,
            "issue_detection": True,
            "auto_fixing": True
        }
    )
    
    openai_client = AsyncOpenAI(api_key="your-api-key")
    
    # 初始化质量控制智能体
    quality = QualityAgent(config, openai_client)
    await quality.initialize()
    
    # 综合质量检查
    print("开始综合质量检查...")
    
    original_text = "这是原始中文文本，描述了一个复杂的故事情节。"
    translated_text = "This is the translated English text, describing a complex story plot."
    
    quality_data = {
        "original": original_text,
        "translated": translated_text,
        "context": {
            "source_language": "zh",
            "target_language": "en",
            "genre": "小说",
            "target_culture": "Western"
        }
    }
    
    quality_result = await quality.comprehensive_quality_check(quality_data)
    
    print(f"质量检查完成!")
    print(f"- 总体评分: {quality_result['overall_score']:.2f}")
    print(f"- 质量等级: {quality_result['quality_grade']}")
    print(f"- 检测到问题: {len(quality_result['issues_detected'])} 个")
    print(f"- 改进建议: {len(quality_result['improvement_suggestions'])} 条")
    print(f"- 标准合规性: {'通过' if quality_result['standards_compliance']['passed'] else '未通过'}")
    
    print("\n详细评分:")
    for dimension, score_data in quality_result['detailed_scores'].items():
        print(f"- {dimension}: {score_data['score']:.2f}")
    
    # 问题检测
    print("\n开始专项问题检测...")
    issue_result = await quality.detect_issues({
        "original": original_text,
        "translated": translated_text,
        "context": quality_data["context"]
    })
    
    print(f"问题检测完成!")
    print(f"- 发现问题: {issue_result['issues_count']} 个")
    print(f"- 严重程度分布: {issue_result['severity_distribution']}")
    
    await quality.cleanup()
    return quality_result


async def integrated_workflow_example():
    """集成工作流示例"""
    print("\n=== 集成工作流示例 ===")
    
    print("这个示例展示了所有增强智能体协作的完整工作流程:")
    print("1. 增强解析 -> 2. 高级翻译 -> 3. 智能编辑 -> 4. 质量控制 -> 5. 记忆存储")
    
    # 由于这是一个完整的工作流，需要协调器来管理
    # 这里展示概念性的流程
    
    workflow_steps = [
        "📖 深度解析文档结构和内容",
        "🧠 构建知识图谱和角色关系网络", 
        "🌍 多策略翻译和深度文化适配",
        "✏️ 多轮编辑优化（一致性、风格、流畅性）",
        "🎯 全方位质量评估和问题检测",
        "💾 智能记忆存储和上下文管理",
        "📊 生成详细的处理报告"
    ]
    
    print("\n完整工作流程:")
    for i, step in enumerate(workflow_steps, 1):
        print(f"{i}. {step}")
    
    print("\n这个集成工作流提供了:")
    print("- 🎯 更高的翻译质量和准确性")
    print("- 🧠 更强的上下文理解和一致性保持")
    print("- 🌍 更深层的文化适配和本土化")
    print("- ✨ 更流畅自然的语言表达")
    print("- 📈 实时质量监控和持续改进")
    print("- 🔄 可追溯的版本管理和变更历史")


async def main():
    """主函数"""
    print("🚀 小说翻译修改器 - 高级功能演示")
    print("=" * 50)
    
    try:
        # 运行各个示例
        await advanced_parsing_example()
        await advanced_translation_example()
        await advanced_memory_example()
        await advanced_editing_example()
        await advanced_quality_example()
        await integrated_workflow_example()
        
        print("\n✨ 所有高级功能演示完成!")
        print("\n这些增强的智能体为小说翻译提供了:")
        print("- 🎯 专业级的翻译质量")
        print("- 🧠 智能的上下文管理")
        print("- 🌍 深度的文化适配")
        print("- ✏️ 全面的编辑优化")
        print("- 🔍 严格的质量控制")
        print("- 📊 详细的分析报告")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # 运行高级功能演示
    import sys
    sys.exit(asyncio.run(main())) 