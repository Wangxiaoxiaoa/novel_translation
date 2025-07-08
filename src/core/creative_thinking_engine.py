"""
创造性思维引擎 - 提供创新和创意能力
Creative Thinking Engine - Providing innovation and creative capabilities
"""

import asyncio
import random
import itertools
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from loguru import logger
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import networkx as nx
import openai


class CreativeMethod(Enum):
    """创造性方法"""
    BRAINSTORMING = "brainstorming"           # 头脑风暴
    LATERAL_THINKING = "lateral_thinking"     # 横向思维
    MIND_MAPPING = "mind_mapping"             # 思维导图
    SCAMPER = "scamper"                       # SCAMPER技法
    SYNECTICS = "synectics"                   # 类推创新法
    MORPHOLOGICAL = "morphological"          # 形态分析法
    CONSTRAINT_RELAXATION = "constraint_relaxation"  # 约束放松法
    ANALOGICAL_REASONING = "analogical_reasoning"    # 类比推理
    REVERSE_THINKING = "reverse_thinking"     # 逆向思维
    RANDOM_STIMULATION = "random_stimulation" # 随机刺激


@dataclass
class CreativeChallenge:
    """创造性挑战"""
    description: str
    constraints: List[str]
    goals: List[str]
    domain: str
    difficulty_level: int
    creative_requirements: List[str]
    evaluation_criteria: List[str]


@dataclass
class CreativeIdea:
    """创意想法"""
    title: str
    description: str
    feasibility_score: float
    originality_score: float
    usefulness_score: float
    overall_score: float
    inspiration_sources: List[str]
    implementation_steps: List[str]
    potential_issues: List[str]
    variations: List[str]


class CreativeThinkingEngine:
    """创造性思维引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 创造性思维方法
        self.brainstormer = BrainstormingEngine()
        self.lateral_thinker = LateralThinkingEngine()
        self.mind_mapper = MindMappingEngine()
        self.scamper_engine = ScamperEngine()
        self.synectics_engine = SynecticsEngine()
        self.morphological_analyzer = MorphologicalAnalyzer()
        
        # 创意生成器
        self.idea_generator = IdeaGenerator()
        self.concept_combiner = ConceptCombiner()
        self.analogy_engine = AnalogyEngine()
        
        # 创意评估器
        self.creativity_evaluator = CreativityEvaluator()
        self.feasibility_assessor = FeasibilityAssessor()
        self.originality_checker = OriginalityChecker()
        
        # 灵感管理器
        self.inspiration_manager = InspirationManager()
        self.serendipity_engine = SerendipityEngine()
        
        # 创造性记忆
        self.creative_memory = CreativeMemory()
        
        # 创意优化器
        self.idea_optimizer = IdeaOptimizer()
        
    async def initialize(self):
        """初始化创造性思维引擎"""
        try:
            logger.info("初始化创造性思维引擎...")
            
            # 初始化各个组件
            await asyncio.gather(
                self.brainstormer.initialize(),
                self.lateral_thinker.initialize(),
                self.mind_mapper.initialize(),
                self.scamper_engine.initialize(),
                self.synectics_engine.initialize(),
                self.morphological_analyzer.initialize(),
                self.idea_generator.initialize(),
                self.concept_combiner.initialize(),
                self.analogy_engine.initialize(),
                self.creativity_evaluator.initialize(),
                self.feasibility_assessor.initialize(),
                self.originality_checker.initialize(),
                self.inspiration_manager.initialize(),
                self.serendipity_engine.initialize(),
                self.creative_memory.initialize(),
                self.idea_optimizer.initialize()
            )
            
            logger.info("创造性思维引擎初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"创造性思维引擎初始化失败: {e}")
            return False
    
    async def generate_creative_solutions(self, challenge: CreativeChallenge, 
                                        methods: Optional[List[CreativeMethod]] = None) -> List[CreativeIdea]:
        """生成创造性解决方案"""
        try:
            logger.info(f"开始生成创造性解决方案: {challenge.description}")
            
            # 如果没有指定方法，自动选择最适合的方法
            if methods is None:
                methods = await self.select_optimal_methods(challenge)
            
            # 使用多种创造性方法生成想法
            idea_generation_tasks = []
            
            for method in methods:
                if method == CreativeMethod.BRAINSTORMING:
                    idea_generation_tasks.append(
                        self.brainstormer.generate_ideas(challenge)
                    )
                elif method == CreativeMethod.LATERAL_THINKING:
                    idea_generation_tasks.append(
                        self.lateral_thinker.generate_ideas(challenge)
                    )
                elif method == CreativeMethod.MIND_MAPPING:
                    idea_generation_tasks.append(
                        self.mind_mapper.generate_ideas(challenge)
                    )
                elif method == CreativeMethod.SCAMPER:
                    idea_generation_tasks.append(
                        self.scamper_engine.generate_ideas(challenge)
                    )
                elif method == CreativeMethod.SYNECTICS:
                    idea_generation_tasks.append(
                        self.synectics_engine.generate_ideas(challenge)
                    )
                elif method == CreativeMethod.MORPHOLOGICAL:
                    idea_generation_tasks.append(
                        self.morphological_analyzer.generate_ideas(challenge)
                    )
            
            # 并行执行所有创意生成任务
            idea_batches = await asyncio.gather(*idea_generation_tasks, return_exceptions=True)
            
            # 合并所有想法
            all_ideas = []
            for batch in idea_batches:
                if isinstance(batch, list):
                    all_ideas.extend(batch)
            
            # 概念组合生成新想法
            combined_ideas = await self.concept_combiner.combine_concepts(all_ideas, challenge)
            all_ideas.extend(combined_ideas)
            
            # 类比推理生成想法
            analogical_ideas = await self.analogy_engine.generate_analogical_ideas(challenge)
            all_ideas.extend(analogical_ideas)
            
            # 评估和筛选想法
            evaluated_ideas = await self.evaluate_ideas(all_ideas, challenge)
            
            # 优化最佳想法
            optimized_ideas = await self.optimize_ideas(evaluated_ideas, challenge)
            
            # 存储到创造性记忆
            await self.creative_memory.store_ideas(optimized_ideas, challenge)
            
            logger.info(f"生成创造性解决方案完成，共 {len(optimized_ideas)} 个想法")
            return optimized_ideas
            
        except Exception as e:
            logger.error(f"生成创造性解决方案失败: {e}")
            return []
    
    async def select_optimal_methods(self, challenge: CreativeChallenge) -> List[CreativeMethod]:
        """选择最优创造性方法"""
        try:
            # 根据挑战特征选择方法
            methods = []
            
            # 基于领域选择
            if challenge.domain in ["technology", "engineering"]:
                methods.extend([CreativeMethod.SCAMPER, CreativeMethod.MORPHOLOGICAL])
            elif challenge.domain in ["art", "design"]:
                methods.extend([CreativeMethod.BRAINSTORMING, CreativeMethod.SYNECTICS])
            elif challenge.domain in ["literature", "writing"]:
                methods.extend([CreativeMethod.LATERAL_THINKING, CreativeMethod.ANALOGICAL_REASONING])
            
            # 基于困难程度选择
            if challenge.difficulty_level >= 7:
                methods.extend([CreativeMethod.CONSTRAINT_RELAXATION, CreativeMethod.REVERSE_THINKING])
            
            # 基于创造性要求选择
            if "originality" in challenge.creative_requirements:
                methods.append(CreativeMethod.RANDOM_STIMULATION)
            if "feasibility" in challenge.creative_requirements:
                methods.append(CreativeMethod.MORPHOLOGICAL)
            
            # 确保至少有基础方法
            if not methods:
                methods = [CreativeMethod.BRAINSTORMING, CreativeMethod.LATERAL_THINKING]
            
            # 去重并限制数量
            methods = list(set(methods))[:4]
            
            logger.info(f"选择创造性方法: {[m.value for m in methods]}")
            return methods
            
        except Exception as e:
            logger.error(f"选择创造性方法失败: {e}")
            return [CreativeMethod.BRAINSTORMING]
    
    async def evaluate_ideas(self, ideas: List[CreativeIdea], challenge: CreativeChallenge) -> List[CreativeIdea]:
        """评估想法"""
        try:
            evaluation_tasks = []
            
            for idea in ideas:
                evaluation_tasks.append(
                    self.evaluate_single_idea(idea, challenge)
                )
            
            # 并行评估所有想法
            evaluated_ideas = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
            
            # 过滤成功评估的想法
            valid_ideas = [idea for idea in evaluated_ideas if isinstance(idea, CreativeIdea)]
            
            # 按总分排序
            valid_ideas.sort(key=lambda x: x.overall_score, reverse=True)
            
            # 返回前20个最佳想法
            return valid_ideas[:20]
            
        except Exception as e:
            logger.error(f"评估想法失败: {e}")
            return ideas
    
    async def evaluate_single_idea(self, idea: CreativeIdea, challenge: CreativeChallenge) -> CreativeIdea:
        """评估单个想法"""
        try:
            # 可行性评估
            feasibility_score = await self.feasibility_assessor.assess(idea, challenge)
            
            # 原创性评估
            originality_score = await self.originality_checker.check(idea)
            
            # 有用性评估
            usefulness_score = await self.creativity_evaluator.evaluate_usefulness(idea, challenge)
            
            # 计算总体分数
            overall_score = (feasibility_score * 0.3 + originality_score * 0.4 + usefulness_score * 0.3)
            
            # 更新想法
            idea.feasibility_score = feasibility_score
            idea.originality_score = originality_score
            idea.usefulness_score = usefulness_score
            idea.overall_score = overall_score
            
            return idea
            
        except Exception as e:
            logger.error(f"评估单个想法失败: {e}")
            return idea
    
    async def optimize_ideas(self, ideas: List[CreativeIdea], challenge: CreativeChallenge) -> List[CreativeIdea]:
        """优化想法"""
        try:
            optimization_tasks = []
            
            # 对前10个想法进行优化
            for idea in ideas[:10]:
                optimization_tasks.append(
                    self.idea_optimizer.optimize(idea, challenge)
                )
            
            # 并行优化
            optimized_ideas = await asyncio.gather(*optimization_tasks, return_exceptions=True)
            
            # 过滤成功优化的想法
            valid_optimized = [idea for idea in optimized_ideas if isinstance(idea, CreativeIdea)]
            
            # 合并未优化的想法
            remaining_ideas = ideas[10:] if len(ideas) > 10 else []
            
            return valid_optimized + remaining_ideas
            
        except Exception as e:
            logger.error(f"优化想法失败: {e}")
            return ideas
    
    async def generate_serendipitous_ideas(self, context: Dict[str, Any]) -> List[CreativeIdea]:
        """生成偶然发现的想法"""
        try:
            serendipitous_ideas = await self.serendipity_engine.generate_ideas(context)
            return serendipitous_ideas
            
        except Exception as e:
            logger.error(f"生成偶然想法失败: {e}")
            return []
    
    async def get_inspiration(self, topic: str, num_inspirations: int = 5) -> List[Dict[str, Any]]:
        """获取灵感"""
        try:
            inspirations = await self.inspiration_manager.get_inspirations(topic, num_inspirations)
            return inspirations
            
        except Exception as e:
            logger.error(f"获取灵感失败: {e}")
            return []
    
    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理创造性思维引擎资源...")
            
            # 清理各个组件
            await asyncio.gather(
                self.brainstormer.cleanup(),
                self.lateral_thinker.cleanup(),
                self.mind_mapper.cleanup(),
                self.scamper_engine.cleanup(),
                self.synectics_engine.cleanup(),
                self.morphological_analyzer.cleanup(),
                self.idea_generator.cleanup(),
                self.concept_combiner.cleanup(),
                self.analogy_engine.cleanup(),
                self.creativity_evaluator.cleanup(),
                self.feasibility_assessor.cleanup(),
                self.originality_checker.cleanup(),
                self.inspiration_manager.cleanup(),
                self.serendipity_engine.cleanup(),
                self.creative_memory.cleanup(),
                self.idea_optimizer.cleanup(),
                return_exceptions=True
            )
            
            logger.info("创造性思维引擎资源清理完成")
            
        except Exception as e:
            logger.error(f"清理创造性思维引擎资源失败: {e}")


# 支持类的实现
class BrainstormingEngine:
    """头脑风暴引擎"""
    
    async def initialize(self):
        pass
    
    async def generate_ideas(self, challenge: CreativeChallenge) -> List[CreativeIdea]:
        """生成头脑风暴想法"""
        ideas = []
        for i in range(5):
            idea = CreativeIdea(
                title=f"头脑风暴想法 {i+1}",
                description=f"针对 {challenge.description} 的头脑风暴解决方案",
                feasibility_score=0.0,
                originality_score=0.0,
                usefulness_score=0.0,
                overall_score=0.0,
                inspiration_sources=["头脑风暴"],
                implementation_steps=[],
                potential_issues=[],
                variations=[]
            )
            ideas.append(idea)
        return ideas
    
    async def cleanup(self):
        pass


class LateralThinkingEngine:
    """横向思维引擎"""
    
    async def initialize(self):
        pass
    
    async def generate_ideas(self, challenge: CreativeChallenge) -> List[CreativeIdea]:
        """生成横向思维想法"""
        ideas = []
        for i in range(4):
            idea = CreativeIdea(
                title=f"横向思维想法 {i+1}",
                description=f"使用横向思维解决 {challenge.description}",
                feasibility_score=0.0,
                originality_score=0.0,
                usefulness_score=0.0,
                overall_score=0.0,
                inspiration_sources=["横向思维"],
                implementation_steps=[],
                potential_issues=[],
                variations=[]
            )
            ideas.append(idea)
        return ideas
    
    async def cleanup(self):
        pass


class MindMappingEngine:
    """思维导图引擎"""
    
    async def initialize(self):
        pass
    
    async def generate_ideas(self, challenge: CreativeChallenge) -> List[CreativeIdea]:
        """生成思维导图想法"""
        ideas = []
        for i in range(3):
            idea = CreativeIdea(
                title=f"思维导图想法 {i+1}",
                description=f"通过思维导图分析得出的解决方案",
                feasibility_score=0.0,
                originality_score=0.0,
                usefulness_score=0.0,
                overall_score=0.0,
                inspiration_sources=["思维导图"],
                implementation_steps=[],
                potential_issues=[],
                variations=[]
            )
            ideas.append(idea)
        return ideas
    
    async def cleanup(self):
        pass


class ScamperEngine:
    """SCAMPER技法引擎"""
    
    async def initialize(self):
        pass
    
    async def generate_ideas(self, challenge: CreativeChallenge) -> List[CreativeIdea]:
        """生成SCAMPER想法"""
        scamper_methods = ["Substitute", "Combine", "Adapt", "Modify", "Put to another use", "Eliminate", "Reverse"]
        ideas = []
        
        for method in scamper_methods[:3]:
            idea = CreativeIdea(
                title=f"SCAMPER-{method} 想法",
                description=f"使用 {method} 方法解决问题",
                feasibility_score=0.0,
                originality_score=0.0,
                usefulness_score=0.0,
                overall_score=0.0,
                inspiration_sources=[f"SCAMPER-{method}"],
                implementation_steps=[],
                potential_issues=[],
                variations=[]
            )
            ideas.append(idea)
        
        return ideas
    
    async def cleanup(self):
        pass


class SynecticsEngine:
    """类推创新法引擎"""
    
    async def initialize(self):
        pass
    
    async def generate_ideas(self, challenge: CreativeChallenge) -> List[CreativeIdea]:
        """生成类推创新想法"""
        ideas = []
        analogies = ["自然界", "机械系统", "人体结构", "社会组织"]
        
        for analogy in analogies:
            idea = CreativeIdea(
                title=f"基于{analogy}的类推想法",
                description=f"从{analogy}中获得启发的解决方案",
                feasibility_score=0.0,
                originality_score=0.0,
                usefulness_score=0.0,
                overall_score=0.0,
                inspiration_sources=[f"类推-{analogy}"],
                implementation_steps=[],
                potential_issues=[],
                variations=[]
            )
            ideas.append(idea)
        
        return ideas
    
    async def cleanup(self):
        pass


class MorphologicalAnalyzer:
    """形态分析器"""
    
    async def initialize(self):
        pass
    
    async def generate_ideas(self, challenge: CreativeChallenge) -> List[CreativeIdea]:
        """生成形态分析想法"""
        ideas = []
        for i in range(3):
            idea = CreativeIdea(
                title=f"形态分析想法 {i+1}",
                description="通过形态分析法系统性探索解决方案",
                feasibility_score=0.0,
                originality_score=0.0,
                usefulness_score=0.0,
                overall_score=0.0,
                inspiration_sources=["形态分析"],
                implementation_steps=[],
                potential_issues=[],
                variations=[]
            )
            ideas.append(idea)
        return ideas
    
    async def cleanup(self):
        pass


class IdeaGenerator:
    async def initialize(self): pass
    async def cleanup(self): pass

class ConceptCombiner:
    async def initialize(self): pass
    async def combine_concepts(self, ideas: List[CreativeIdea], challenge: CreativeChallenge) -> List[CreativeIdea]:
        return []
    async def cleanup(self): pass

class AnalogyEngine:
    async def initialize(self): pass
    async def generate_analogical_ideas(self, challenge: CreativeChallenge) -> List[CreativeIdea]:
        return []
    async def cleanup(self): pass

class CreativityEvaluator:
    async def initialize(self): pass
    async def evaluate_usefulness(self, idea: CreativeIdea, challenge: CreativeChallenge) -> float:
        return 0.8
    async def cleanup(self): pass

class FeasibilityAssessor:
    async def initialize(self): pass
    async def assess(self, idea: CreativeIdea, challenge: CreativeChallenge) -> float:
        return 0.7
    async def cleanup(self): pass

class OriginalityChecker:
    async def initialize(self): pass
    async def check(self, idea: CreativeIdea) -> float:
        return 0.85
    async def cleanup(self): pass

class InspirationManager:
    async def initialize(self): pass
    async def get_inspirations(self, topic: str, num: int) -> List[Dict[str, Any]]:
        return []
    async def cleanup(self): pass

class SerendipityEngine:
    async def initialize(self): pass
    async def generate_ideas(self, context: Dict[str, Any]) -> List[CreativeIdea]:
        return []
    async def cleanup(self): pass

class CreativeMemory:
    async def initialize(self): pass
    async def store_ideas(self, ideas: List[CreativeIdea], challenge: CreativeChallenge): pass
    async def cleanup(self): pass

class IdeaOptimizer:
    async def initialize(self): pass
    async def optimize(self, idea: CreativeIdea, challenge: CreativeChallenge) -> CreativeIdea:
        return idea
    async def cleanup(self): pass 