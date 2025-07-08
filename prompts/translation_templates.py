"""
翻译提示词模板
Translation Prompt Templates
"""

from typing import Dict, Any, Optional
from jinja2 import Template


class TranslationTemplates:
    """翻译提示词模板类"""
    
    def __init__(self):
        self.templates = {
            "base_translation": self._get_base_translation_template(),
            "character_translation": self._get_character_translation_template(),
            "cultural_adaptation": self._get_cultural_adaptation_template(),
            "quality_assessment": self._get_quality_assessment_template(),
            "consistency_check": self._get_consistency_check_template(),
            "chapter_summary": self._get_chapter_summary_template(),
            "terminology_translation": self._get_terminology_translation_template(),
        }
    
    def get_template(self, template_name: str) -> Optional[Template]:
        """获取模板"""
        template_content = self.templates.get(template_name)
        if template_content:
            return Template(template_content)
        return None
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """渲染模板"""
        template = self.get_template(template_name)
        if template:
            return template.render(**kwargs)
        return ""
    
    def _get_base_translation_template(self) -> str:
        """基础翻译模板"""
        return """
作为专业的小说翻译专家，请将以下中文小说内容翻译成{{ target_language }}，并进行适当的文化适配。

## 翻译要求：
1. 保持原文的情感色彩和文学风格
2. 确保人物性格和对话符合角色特征
3. 保持情节连贯性和逻辑性
4. 进行适当的文化适配，使目标读者更容易理解
5. 保持术语和人名的一致性

## 目标语言：{{ target_language }}
## 目标文化：{{ target_culture }}

{% if character_context %}
## 主要角色：
{% for char_name, char_info in character_context.items() %}
- {{ char_name }}{% if char_info.cultural_adaptations.get(target_language) %} ({{ char_info.cultural_adaptations[target_language] }}){% endif %}: {{ char_info.description }}
{% endfor %}
{% endif %}

{% if location_context %}
## 主要地点：
{% for loc_name, loc_info in location_context.items() %}
- {{ loc_name }}{% if loc_info.cultural_adaptations.get(target_language) %} ({{ loc_info.cultural_adaptations[target_language] }}){% endif %}: {{ loc_info.description }}
{% endfor %}
{% endif %}

{% if terminology_context %}
## 专业术语：
{% for term, term_info in terminology_context.items() %}
- {{ term }}{% if term_info.cultural_adaptations.get(target_language) %} ({{ term_info.cultural_adaptations[target_language] }}){% endif %}: {{ term_info.definition }}
{% endfor %}
{% endif %}

{% if plot_context %}
## 情节背景：
{{ plot_context }}
{% endif %}

{% if previous_chapters %}
## 前置章节摘要：
{% for chapter in previous_chapters %}
- {{ chapter.title }}: {{ chapter.summary }}
{% endfor %}
{% endif %}

{% if style_guide %}
## 翻译风格指南：
{{ style_guide }}
{% endif %}

## 文化适配要求：
- 人名翻译：根据目标文化习惯调整人名
- 地名翻译：保持地域特色但便于理解
- 文化概念：适当本土化，避免文化冲突
- 社会习俗：符合目标文化的表达方式
- 价值观念：保持原意但适应目标文化

请直接返回翻译后的内容，不要包含任何解释或额外信息。

## 原文：
{{ source_content }}
"""
    
    def _get_character_translation_template(self) -> str:
        """角色翻译模板"""
        return """
请为以下角色进行文化适配翻译：

## 角色信息：
- 原名：{{ character_name }}
- 描述：{{ character_description }}
- 角色特征：{{ character_traits }}
- 文化背景：{{ cultural_background }}

## 目标语言：{{ target_language }}
## 目标文化：{{ target_culture }}

## 翻译要求：
1. 保持角色的核心特征和个性
2. 适应目标文化的命名习惯
3. 确保名字易于发音和记忆
4. 避免文化冲突和误解
5. 保持与其他角色的关系一致性

{% if cultural_rules %}
## 文化规则：
{% for rule_type, rule_desc in cultural_rules.items() %}
- {{ rule_type }}: {{ rule_desc }}
{% endfor %}
{% endif %}

{% if similar_characters %}
## 相似角色参考：
{% for char in similar_characters %}
- {{ char.original_name }} -> {{ char.adapted_name }}
{% endfor %}
{% endif %}

请提供适配后的角色名称和简短说明。

格式：
适配名称：[适配后的名称]
说明：[适配理由和特点]
"""
    
    def _get_cultural_adaptation_template(self) -> str:
        """文化适配模板"""
        return """
请将以下内容中的文化元素适配到{{ target_culture }}文化背景中：

## 原文内容：
{{ source_content }}

## 目标文化：{{ target_culture }}
## 适配要求：

### 1. 社会习俗适配
- 礼仪规范：{{ cultural_rules.get('social_customs', '根据目标文化调整') }}
- 节日庆典：{{ cultural_rules.get('festivals', '适当本土化') }}
- 家庭关系：{{ cultural_rules.get('family_relations', '符合目标文化价值观') }}

### 2. 文化概念适配
- 哲学思想：转换为目标文化易理解的概念
- 宗教信仰：避免宗教冲突，适当调整
- 价值观念：保持核心思想但适应目标文化

### 3. 生活方式适配
- 饮食文化：{{ cultural_rules.get('food_culture', '本土化食物名称') }}
- 服饰文化：{{ cultural_rules.get('clothing_culture', '描述适应目标文化') }}
- 居住环境：{{ cultural_rules.get('living_environment', '符合目标文化背景') }}

### 4. 语言表达适配
- 敬语系统：{{ cultural_rules.get('honorifics', '根据目标语言调整') }}
- 称谓方式：{{ cultural_rules.get('addressing_style', '符合目标文化习惯') }}
- 表达方式：{{ cultural_rules.get('expression_style', '自然流畅') }}

## 注意事项：
1. 保持原文的情感和意图
2. 避免造成文化误解
3. 确保目标读者能够理解
4. 保持故事的连贯性
5. 尊重两种文化的差异

请返回适配后的内容，保持原文结构不变。
"""
    
    def _get_quality_assessment_template(self) -> str:
        """质量评估模板"""
        return """
请评估以下翻译的质量，从多个维度给出详细评分：

## 原文：
{{ original_content }}

## 译文：
{{ translated_content }}

## 评估维度：

### 1. 准确性 (Accuracy) - 权重: 30%
- 是否准确传达了原文意思
- 是否有遗漏或误译的内容
- 专业术语翻译是否正确

### 2. 流畅性 (Fluency) - 权重: 25%
- 译文是否符合目标语言的表达习惯
- 语法是否正确
- 语言是否自然流畅

### 3. 一致性 (Consistency) - 权重: 20%
- 人物、地点、术语是否保持一致
- 翻译风格是否统一
- 前后文是否连贯

### 4. 文化适配 (Cultural Adaptation) - 权重: 15%
- 是否恰当地进行了文化适配
- 是否避免了文化冲突
- 目标读者是否容易理解

### 5. 文学性 (Literary Quality) - 权重: 10%
- 是否保持了原文的文学风格
- 情感色彩是否得到体现
- 艺术效果是否保持

## 评分标准：
- 9-10分：优秀，几乎无瑕疵
- 7-8分：良好，有少量可改进之处
- 5-6分：一般，需要明显改进
- 3-4分：较差，存在严重问题
- 1-2分：很差，需要重新翻译

请以JSON格式返回评分结果：
{
  "accuracy": 评分,
  "fluency": 评分,
  "consistency": 评分,
  "cultural_adaptation": 评分,
  "literary_quality": 评分,
  "overall": 总体评分,
  "detailed_comments": {
    "strengths": ["优点列表"],
    "weaknesses": ["缺点列表"],
    "suggestions": ["改进建议"]
  },
  "specific_issues": [
    {
      "type": "问题类型",
      "description": "问题描述",
      "suggestion": "改进建议"
    }
  ]
}
"""
    
    def _get_consistency_check_template(self) -> str:
        """一致性检查模板"""
        return """
请检查以下翻译内容的一致性：

## 检查类型：{{ check_type }}

## 当前内容：
{{ current_content }}

## 历史记录：
{% for record in historical_records %}
### 记录 {{ loop.index }}：
- 来源：{{ record.source }}
- 内容：{{ record.content }}
- 时间：{{ record.timestamp }}
{% endfor %}

## 检查要点：

{% if check_type == "character_consistency" %}
### 角色一致性检查：
1. 角色名称是否一致
2. 角色性格特征是否保持
3. 角色能力设定是否统一
4. 角色关系是否前后一致
5. 角色发展是否合理
{% endif %}

{% if check_type == "terminology_consistency" %}
### 术语一致性检查：
1. 专业术语翻译是否统一
2. 术语定义是否一致
3. 术语使用场景是否恰当
4. 术语层级关系是否正确
5. 新术语是否与已有术语冲突
{% endif %}

{% if check_type == "plot_consistency" %}
### 情节一致性检查：
1. 时间线是否逻辑合理
2. 因果关系是否清晰
3. 情节发展是否连贯
4. 伏笔是否得到呼应
5. 世界观设定是否一致
{% endif %}

{% if check_type == "cultural_consistency" %}
### 文化一致性检查：
1. 文化适配是否统一
2. 文化元素是否协调
3. 文化冲突是否避免
4. 文化表达是否恰当
5. 文化背景是否一致
{% endif %}

## 检查结果要求：
请以JSON格式返回检查结果：
{
  "consistency_score": 一致性评分（1-10分）,
  "issues_found": [
    {
      "type": "问题类型",
      "description": "问题描述",
      "severity": "严重程度（high/medium/low）",
      "suggestion": "解决建议"
    }
  ],
  "recommendations": [
    "改进建议列表"
  ],
  "overall_assessment": "总体评估"
}
"""
    
    def _get_chapter_summary_template(self) -> str:
        """章节摘要模板"""
        return """
请为以下章节内容生成一个简洁而全面的摘要：

## 章节信息：
- 标题：{{ chapter_title }}
- 章节号：{{ chapter_number }}
- 字数：{{ word_count }}

## 章节内容：
{{ chapter_content }}

## 摘要要求：
1. 长度控制在100-200字之间
2. 突出主要情节发展
3. 包含重要人物和事件
4. 体现关键转折点
5. 为后续章节提供背景

## 摘要应包含：
- 主要事件：本章发生的核心事件
- 关键人物：出现的重要角色及其作用
- 情节发展：推进主线剧情的要素
- 情感基调：本章的整体氛围
- 重要信息：影响后续情节的关键信息

{% if previous_summaries %}
## 前置章节摘要（参考）：
{% for summary in previous_summaries %}
- 第{{ summary.chapter_number }}章：{{ summary.content }}
{% endfor %}
{% endif %}

请生成符合要求的章节摘要，确保信息准确、简洁明了。
"""
    
    def _get_terminology_translation_template(self) -> str:
        """术语翻译模板"""
        return """
请将以下专业术语翻译成{{ target_language }}，并进行文化适配：

## 术语信息：
- 原术语：{{ original_term }}
- 类别：{{ category }}
- 定义：{{ definition }}
- 使用场景：{{ context }}

## 目标语言：{{ target_language }}
## 目标文化：{{ target_culture }}

## 翻译原则：
1. 保持术语的专业性和准确性
2. 符合目标文化的表达习惯
3. 考虑目标读者的理解能力
4. 与同类术语保持一致性
5. 避免产生歧义和误解

{% if similar_terms %}
## 相关术语参考：
{% for term in similar_terms %}
- {{ term.original }} -> {{ term.translated }}
{% endfor %}
{% endif %}

{% if cultural_context %}
## 文化背景：
{{ cultural_context }}
{% endif %}

## 翻译要求：
请提供以下信息：
1. 翻译结果：最佳的术语翻译
2. 备选方案：2-3个备选翻译（如适用）
3. 翻译说明：解释翻译选择的理由
4. 使用建议：在什么情况下使用这个术语
5. 注意事项：使用时需要注意的问题

请以JSON格式返回：
{
  "primary_translation": "主要翻译",
  "alternatives": ["备选翻译1", "备选翻译2"],
  "explanation": "翻译说明",
  "usage_context": "使用场景",
  "notes": "注意事项",
  "confidence": "置信度（1-10）"
}
"""


# 创建全局模板实例
translation_templates = TranslationTemplates()


def get_translation_prompt(template_name: str, **kwargs) -> str:
    """获取翻译提示词"""
    return translation_templates.render_template(template_name, **kwargs)


def get_available_templates() -> list:
    """获取可用的模板列表"""
    return list(translation_templates.templates.keys())


# 使用示例
if __name__ == "__main__":
    # 示例：生成基础翻译提示词
    prompt = get_translation_prompt(
        "base_translation",
        target_language="English",
        target_culture="Western",
        source_content="这是一个测试内容。",
        character_context={},
        location_context={},
        terminology_context={},
        plot_context="",
        previous_chapters=[],
        style_guide="保持原文风格"
    )
    
    print("Generated Translation Prompt:")
    print(prompt) 