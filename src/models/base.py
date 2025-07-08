"""
基础数据模型和类型定义
Base data models and type definitions
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import json


class LanguageCode(Enum):
    """语言代码枚举 - 支持40+种语言"""
    
    # 主要语言
    ZH = "zh"           # 中文
    EN = "en"           # 英语
    JA = "ja"           # 日语
    KO = "ko"           # 韩语
    FR = "fr"           # 法语
    DE = "de"           # 德语
    ES = "es"           # 西班牙语
    RU = "ru"           # 俄语
    AR = "ar"           # 阿拉伯语
    
    # 欧洲语言
    IT = "it"           # 意大利语
    PT = "pt"           # 葡萄牙语
    NL = "nl"           # 荷兰语
    PL = "pl"           # 波兰语
    CS = "cs"           # 捷克语
    SK = "sk"           # 斯洛伐克语
    HU = "hu"           # 匈牙利语
    RO = "ro"           # 罗马尼亚语
    BG = "bg"           # 保加利亚语
    HR = "hr"           # 克罗地亚语
    SL = "sl"           # 斯洛文尼亚语
    ET = "et"           # 爱沙尼亚语
    LV = "lv"           # 拉脱维亚语
    LT = "lt"           # 立陶宛语
    
    # 北欧语言
    SV = "sv"           # 瑞典语
    NO = "no"           # 挪威语
    DA = "da"           # 丹麦语
    FI = "fi"           # 芬兰语
    IS = "is"           # 冰岛语
    
    # 南亚和东南亚语言
    HI = "hi"           # 印地语
    TH = "th"           # 泰语
    VI = "vi"           # 越南语
    MS = "ms"           # 马来语
    ID = "id"           # 印尼语
    TA = "ta"           # 泰米尔语
    TE = "te"           # 泰卢固语
    BN = "bn"           # 孟加拉语
    UR = "ur"           # 乌尔都语
    MY = "my"           # 缅甸语
    KM = "km"           # 高棉语
    LO = "lo"           # 老挝语
    
    # 其他亚洲语言
    TR = "tr"           # 土耳其语
    FA = "fa"           # 波斯语
    HE = "he"           # 希伯来语
    
    # 非洲语言
    SW = "sw"           # 斯瓦希里语
    AM = "am"           # 阿姆哈拉语
    
    # 美洲语言
    PT_BR = "pt-BR"     # 巴西葡萄牙语
    ES_MX = "es-MX"     # 墨西哥西班牙语
    
    @classmethod
    def get_language_name(cls, code: str) -> str:
        """获取语言名称"""
        language_names = {
            "zh": "中文",
            "en": "English",
            "ja": "日本語",
            "ko": "한국어",
            "fr": "Français",
            "de": "Deutsch",
            "es": "Español",
            "ru": "Русский",
            "ar": "العربية",
            "it": "Italiano",
            "pt": "Português",
            "nl": "Nederlands",
            "pl": "Polski",
            "cs": "Čeština",
            "sk": "Slovenčina",
            "hu": "Magyar",
            "ro": "Română",
            "bg": "български",
            "hr": "hrvatski",
            "sl": "slovenščina",
            "et": "eesti",
            "lv": "latviešu",
            "lt": "lietuvių",
            "sv": "svenska",
            "no": "norsk",
            "da": "dansk",
            "fi": "suomi",
            "is": "íslenska",
            "hi": "हिन्दी",
            "th": "ไทย",
            "vi": "Tiếng Việt",
            "ms": "Bahasa Melayu",
            "id": "Bahasa Indonesia",
            "ta": "தமிழ்",
            "te": "తెలుగు",
            "bn": "বাংলা",
            "ur": "اردو",
            "my": "မြန်မာ",
            "km": "ខ្មែរ",
            "lo": "ລາວ",
            "tr": "Türkçe",
            "fa": "فارسی",
            "he": "עברית",
            "sw": "Kiswahili",
            "am": "አማርኛ",
            "pt-BR": "Português (Brasil)",
            "es-MX": "Español (México)"
        }
        return language_names.get(code, code)
    
    @classmethod
    def get_language_family(cls, code: str) -> str:
        """获取语言族"""
        language_families = {
            "zh": "sino_tibetan",
            "en": "germanic",
            "ja": "japonic",
            "ko": "koreanic",
            "fr": "romance",
            "de": "germanic",
            "es": "romance",
            "ru": "slavic",
            "ar": "semitic",
            "it": "romance",
            "pt": "romance",
            "nl": "germanic",
            "pl": "slavic",
            "cs": "slavic",
            "sk": "slavic",
            "hu": "uralic",
            "ro": "romance",
            "bg": "slavic",
            "hr": "slavic",
            "sl": "slavic",
            "et": "uralic",
            "lv": "baltic",
            "lt": "baltic",
            "sv": "germanic",
            "no": "germanic",
            "da": "germanic",
            "fi": "uralic",
            "is": "germanic",
            "hi": "indo_aryan",
            "th": "tai_kadai",
            "vi": "austroasiatic",
            "ms": "austronesian",
            "id": "austronesian",
            "ta": "dravidian",
            "te": "dravidian",
            "bn": "indo_aryan",
            "ur": "indo_aryan",
            "my": "sino_tibetan",
            "km": "austroasiatic",
            "lo": "tai_kadai",
            "tr": "turkic",
            "fa": "iranian",
            "he": "semitic",
            "sw": "niger_congo",
            "am": "afroasiatic",
            "pt-BR": "romance",
            "es-MX": "romance"
        }
        return language_families.get(code, "other")
    
    @classmethod
    def get_writing_system(cls, code: str) -> str:
        """获取书写系统"""
        writing_systems = {
            "zh": "chinese",
            "en": "latin",
            "ja": "japanese",
            "ko": "hangul",
            "fr": "latin",
            "de": "latin",
            "es": "latin",
            "ru": "cyrillic",
            "ar": "arabic",
            "it": "latin",
            "pt": "latin",
            "nl": "latin",
            "pl": "latin",
            "cs": "latin",
            "sk": "latin",
            "hu": "latin",
            "ro": "latin",
            "bg": "cyrillic",
            "hr": "latin",
            "sl": "latin",
            "et": "latin",
            "lv": "latin",
            "lt": "latin",
            "sv": "latin",
            "no": "latin",
            "da": "latin",
            "fi": "latin",
            "is": "latin",
            "hi": "devanagari",
            "th": "thai",
            "vi": "latin",
            "ms": "latin",
            "id": "latin",
            "ta": "tamil",
            "te": "telugu",
            "bn": "bengali",
            "ur": "arabic",
            "my": "myanmar",
            "km": "khmer",
            "lo": "lao",
            "tr": "latin",
            "fa": "arabic",
            "he": "hebrew",
            "sw": "latin",
            "am": "ethiopic",
            "pt-BR": "latin",
            "es-MX": "latin"
        }
        return writing_systems.get(code, "latin")
    
    @classmethod
    def get_cultural_context(cls, code: str) -> Dict[str, Any]:
        """获取文化背景信息"""
        cultural_contexts = {
            "zh": {
                "region": "East Asia",
                "traditions": ["confucianism", "taoism", "buddhism"],
                "literary_forms": ["classical_poetry", "historical_fiction", "martial_arts"],
                "naming_conventions": "family_name_first",
                "honorific_system": "complex",
                "cultural_values": ["harmony", "respect", "family"]
            },
            "en": {
                "region": "Global",
                "traditions": ["christianity", "individualism", "democracy"],
                "literary_forms": ["novel", "drama", "poetry"],
                "naming_conventions": "given_name_first",
                "honorific_system": "simple",
                "cultural_values": ["freedom", "equality", "innovation"]
            },
            "ja": {
                "region": "East Asia",
                "traditions": ["shintoism", "buddhism", "bushido"],
                "literary_forms": ["haiku", "light_novel", "manga"],
                "naming_conventions": "family_name_first",
                "honorific_system": "very_complex",
                "cultural_values": ["harmony", "respect", "perfection"]
            },
            "ko": {
                "region": "East Asia",
                "traditions": ["confucianism", "buddhism", "christianity"],
                "literary_forms": ["sijo", "modern_literature", "webtoon"],
                "naming_conventions": "family_name_first",
                "honorific_system": "complex",
                "cultural_values": ["hierarchy", "education", "family"]
            },
            "ar": {
                "region": "Middle East/North Africa",
                "traditions": ["islam", "arab_culture", "bedouin"],
                "literary_forms": ["poetry", "epic", "religious_texts"],
                "naming_conventions": "patronymic",
                "honorific_system": "formal",
                "cultural_values": ["honor", "hospitality", "family"]
            },
            "hi": {
                "region": "South Asia",
                "traditions": ["hinduism", "caste_system", "karma"],
                "literary_forms": ["epic", "devotional", "modern_literature"],
                "naming_conventions": "given_name_first",
                "honorific_system": "complex",
                "cultural_values": ["dharma", "respect", "spiritual"]
            },
            "th": {
                "region": "Southeast Asia",
                "traditions": ["buddhism", "monarchy", "thai_culture"],
                "literary_forms": ["classical_literature", "folk_tales", "modern_fiction"],
                "naming_conventions": "given_name_first",
                "honorific_system": "hierarchical",
                "cultural_values": ["respect", "harmony", "monarchy"]
            },
            "ru": {
                "region": "Eastern Europe",
                "traditions": ["orthodox_christianity", "soviet_history", "slavic_culture"],
                "literary_forms": ["novel", "poetry", "drama"],
                "naming_conventions": "given_name_patronymic",
                "honorific_system": "formal",
                "cultural_values": ["endurance", "literature", "community"]
            },
            "tr": {
                "region": "Western Asia",
                "traditions": ["islam", "ottoman_culture", "secular_modern"],
                "literary_forms": ["classical_poetry", "modern_literature", "folk_tales"],
                "naming_conventions": "given_name_first",
                "honorific_system": "respectful",
                "cultural_values": ["hospitality", "honor", "family"]
            },
            "fa": {
                "region": "Middle East",
                "traditions": ["islam", "persian_culture", "poetry"],
                "literary_forms": ["classical_poetry", "epic", "mystical_literature"],
                "naming_conventions": "given_name_first",
                "honorific_system": "formal",
                "cultural_values": ["poetry", "honor", "mysticism"]
            }
        }
        return cultural_contexts.get(code, {
            "region": "Global",
            "traditions": ["local_culture"],
            "literary_forms": ["modern_literature"],
            "naming_conventions": "given_name_first",
            "honorific_system": "simple",
            "cultural_values": ["respect", "tradition"]
        })


class CultureType(str, Enum):
    """文化类型枚举"""
    WESTERN = "Western"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    SLAVIC = "Slavic"
    MIDDLE_EASTERN = "Middle Eastern"
    CHINESE = "Chinese"


class ProcessingStatus(str, Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class AgentType(Enum):
    """智能体类型"""
    PARSER = "parser"
    TRANSLATOR = "translator"
    MEMORY = "memory"
    COORDINATOR = "coordinator"
    EDITOR = "editor"
    QUALITY = "quality"
    STYLE = "style"
    CULTURAL = "cultural"
    SUPER = "super"  # 超级智能体
    AI_REASONING = "ai_reasoning"  # AI推理引擎
    DEEP_LEARNING = "deep_learning"  # 深度学习引擎
    CREATIVE_THINKING = "creative_thinking"  # 创造性思维引擎
    EXPERT_SYSTEM = "expert_system"  # 专家系统


class DocumentFormat(str, Enum):
    """文档格式枚举"""
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"


class Character(BaseModel):
    """角色模型"""
    name: str
    aliases: List[str] = []
    description: str = ""
    relationships: Dict[str, str] = {}
    appearance_chapters: List[int] = []
    cultural_adaptations: Dict[str, str] = {}
    
    class Config:
        extra = "allow"


class Location(BaseModel):
    """地点模型"""
    name: str
    description: str = ""
    type: str = ""  # 城市、国家、建筑物等
    cultural_adaptations: Dict[str, str] = {}
    
    class Config:
        extra = "allow"


class Item(BaseModel):
    """物品模型"""
    name: str
    description: str = ""
    type: str = ""  # 武器、功法、道具等
    cultural_adaptations: Dict[str, str] = {}
    
    class Config:
        extra = "allow"


class Terminology(BaseModel):
    """术语模型"""
    term: str
    definition: str
    context: str = ""
    category: str = ""
    cultural_adaptations: Dict[str, str] = {}
    
    class Config:
        extra = "allow"


class Chapter(BaseModel):
    """章节模型"""
    id: str
    title: str
    content: str
    original_content: str = ""
    chapter_number: int
    word_count: int = 0
    characters: List[str] = []
    locations: List[str] = []
    items: List[str] = []
    terminologies: List[str] = []
    summary: str = ""
    translated_content: Dict[str, str] = {}  # 语言代码 -> 翻译内容
    translation_status: ProcessingStatus = ProcessingStatus.PENDING
    metadata: Dict[str, Any] = {}
    
    class Config:
        extra = "allow"


class Novel(BaseModel):
    """小说模型"""
    title: str
    author: str = ""
    description: str = ""
    genre: str = ""
    language: LanguageCode = LanguageCode.CHINESE
    chapters: List[Chapter] = []
    characters: Dict[str, Character] = {}
    locations: Dict[str, Location] = {}
    items: Dict[str, Item] = {}
    terminologies: Dict[str, Terminology] = {}
    plot_summary: str = ""
    metadata: Dict[str, Any] = {}
    
    class Config:
        extra = "allow"


class TranslationTask(BaseModel):
    """翻译任务模型"""
    id: str
    novel_id: str
    source_language: LanguageCode
    target_language: LanguageCode
    status: ProcessingStatus = ProcessingStatus.PENDING
    chapters_to_translate: List[str] = []
    translated_chapters: List[str] = []
    progress: float = 0.0
    error_message: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        extra = "allow"


class AgentMessage(BaseModel):
    """智能体消息模型"""
    sender: AgentType
    receiver: AgentType
    message_type: str
    content: Any
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        extra = "allow"


class TranslationContext(BaseModel):
    """翻译上下文模型"""
    novel: Novel
    target_language: LanguageCode
    target_culture: CultureType
    previous_chapters: List[Chapter] = []
    current_chapter: Chapter
    character_context: Dict[str, Character] = {}
    location_context: Dict[str, Location] = {}
    item_context: Dict[str, Item] = {}
    terminology_context: Dict[str, Terminology] = {}
    plot_context: str = ""
    style_guide: str = ""
    
    class Config:
        extra = "allow"


class QualityMetrics(BaseModel):
    """质量评估指标模型"""
    consistency_score: float = 0.0
    cultural_appropriateness_score: float = 0.0
    plot_continuity_score: float = 0.0
    character_consistency_score: float = 0.0
    overall_score: float = 0.0
    issues: List[str] = []
    suggestions: List[str] = []
    
    class Config:
        extra = "allow"


class AgentConfig(BaseModel):
    """智能体配置模型"""
    agent_type: AgentType
    model: str
    temperature: float = 0.7
    max_tokens: int = 4000
    max_retries: int = 3
    timeout: int = 300
    custom_params: Dict[str, Any] = {}
    
    class Config:
        extra = "allow"


class SystemConfig(BaseModel):
    """系统配置模型"""
    openai_config: Dict[str, Any]
    translation_config: Dict[str, Any]
    document_config: Dict[str, Any]
    chapter_config: Dict[str, Any]
    agent_configs: Dict[str, AgentConfig]
    memory_config: Dict[str, Any]
    output_config: Dict[str, Any]
    quality_config: Dict[str, Any]
    logging_config: Dict[str, Any]
    concurrency_config: Dict[str, Any]
    cache_config: Dict[str, Any]
    
    class Config:
        extra = "allow" 