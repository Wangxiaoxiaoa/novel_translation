"""
解析智能体 - 负责文档解析和章节切分
Parser Agent - Responsible for document parsing and chapter segmentation
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import PyPDF2
from docx import Document
from loguru import logger
import tiktoken

from .base_agent import BaseAgent
from ..models.base import (
    AgentMessage, AgentType, Novel, Chapter, ProcessingStatus,
    DocumentFormat, LanguageCode
)


class ParserAgent(BaseAgent):
    """解析智能体"""
    
    def __init__(self, config, openai_client):
        super().__init__(config, openai_client)
        self.encoding = tiktoken.encoding_for_model(self.model)
        self.chapter_patterns = [
            r'第[零一二三四五六七八九十百千万\d]+章',
            r'Chapter\s+\d+',
            r'第\d+章',
            r'卷[零一二三四五六七八九十百千万\d]+',
            r'CHAPTER\s+\d+',
            r'第\s*[零一二三四五六七八九十百千万\d]+\s*章',
        ]
        
    async def initialize(self) -> bool:
        """初始化解析智能体"""
        try:
            logger.info("初始化解析智能体...")
            
            # 健康检查
            health_ok = await self.health_check()
            if not health_ok:
                logger.error("解析智能体健康检查失败")
                return False
                
            self.status = ProcessingStatus.COMPLETED
            logger.info("解析智能体初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"解析智能体初始化失败: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """清理解析智能体资源"""
        try:
            logger.info("清理解析智能体资源...")
            # 清理临时文件等
            return True
        except Exception as e:
            logger.error(f"清理解析智能体资源失败: {e}")
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """处理消息"""
        try:
            message_type = message.message_type
            content = message.content
            
            if message_type == "parse_document":
                result = await self.parse_document(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="parse_result",
                    content=result
                )
            
            elif message_type == "split_chapters":
                result = await self.split_chapters(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="chapters_split",
                    content=result
                )
            
            elif message_type == "analyze_structure":
                result = await self.analyze_structure(content)
                return AgentMessage(
                    sender=self.agent_type,
                    receiver=message.sender,
                    message_type="structure_analysis",
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
    
    async def parse_document(self, file_path: str) -> Dict[str, Any]:
        """解析文档文件"""
        try:
            self.current_task = f"解析文档: {file_path}"
            logger.info(f"开始解析文档: {file_path}")
            
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.txt':
                content = await self.parse_txt(file_path)
            elif file_extension == '.pdf':
                content = await self.parse_pdf(file_path)
            elif file_extension == '.docx':
                content = await self.parse_docx(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")
            
            # 分析文档基本信息
            doc_info = await self.analyze_document_info(content)
            
            result = {
                "file_path": str(file_path),
                "format": file_extension.lstrip('.'),
                "content": content,
                "word_count": len(content),
                "char_count": len(content),
                "estimated_chapters": doc_info.get("estimated_chapters", 0),
                "detected_language": doc_info.get("language", "zh"),
                "title": doc_info.get("title", "未知标题"),
                "author": doc_info.get("author", "未知作者"),
                "genre": doc_info.get("genre", "未知类型")
            }
            
            logger.info(f"文档解析完成: {file_path}, 字符数: {len(content)}")
            return result
            
        except Exception as e:
            logger.error(f"解析文档失败: {e}")
            raise
    
    async def parse_txt(self, file_path: Path) -> str:
        """解析TXT文件"""
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.info(f"使用 {encoding} 编码成功解析TXT文件")
                    return content
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("无法识别文件编码")
            
        except Exception as e:
            logger.error(f"解析TXT文件失败: {e}")
            raise
    
    async def parse_pdf(self, file_path: Path) -> str:
        """解析PDF文件"""
        try:
            content = ""
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    content += page.extract_text()
            
            if not content.strip():
                raise ValueError("PDF文件为空或无法提取文本")
            
            logger.info(f"成功解析PDF文件，共 {len(pdf_reader.pages)} 页")
            return content
            
        except Exception as e:
            logger.error(f"解析PDF文件失败: {e}")
            raise
    
    async def parse_docx(self, file_path: Path) -> str:
        """解析DOCX文件"""
        try:
            doc = Document(file_path)
            content = ""
            
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            
            if not content.strip():
                raise ValueError("DOCX文件为空")
            
            logger.info("成功解析DOCX文件")
            return content
            
        except Exception as e:
            logger.error(f"解析DOCX文件失败: {e}")
            raise
    
    async def analyze_document_info(self, content: str) -> Dict[str, Any]:
        """分析文档基本信息"""
        try:
            # 使用LLM分析文档信息
            prompt = self.load_prompt_template(
                "analyze_document",
                content=content[:2000]  # 只分析前2000字符
            )
            
            messages = [
                {"role": "system", "content": "你是一个专业的文档分析助手，擅长识别小说的基本信息。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.call_llm(messages, temperature=0.1)
            
            # 解析响应
            try:
                import json
                info = json.loads(response)
            except:
                # 如果无法解析JSON，使用默认值
                info = {
                    "title": "未知标题",
                    "author": "未知作者",
                    "genre": "未知类型",
                    "language": "zh",
                    "estimated_chapters": 0
                }
            
            # 估算章节数
            if info.get("estimated_chapters", 0) == 0:
                info["estimated_chapters"] = await self.estimate_chapters(content)
            
            return info
            
        except Exception as e:
            logger.error(f"分析文档信息失败: {e}")
            return {
                "title": "未知标题",
                "author": "未知作者",
                "genre": "未知类型",
                "language": "zh",
                "estimated_chapters": await self.estimate_chapters(content)
            }
    
    async def estimate_chapters(self, content: str) -> int:
        """估算章节数"""
        try:
            chapter_count = 0
            
            for pattern in self.chapter_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                chapter_count = max(chapter_count, len(matches))
            
            # 如果没有找到章节标记，根据内容长度估算
            if chapter_count == 0:
                # 假设每章平均2000字符
                chapter_count = max(1, len(content) // 2000)
            
            return chapter_count
            
        except Exception as e:
            logger.error(f"估算章节数失败: {e}")
            return 1
    
    async def split_chapters(self, data: Dict[str, Any]) -> List[Chapter]:
        """切分章节"""
        try:
            self.current_task = "切分章节"
            content = data.get("content", "")
            title = data.get("title", "未知标题")
            
            logger.info(f"开始切分章节: {title}")
            
            # 首先尝试按照章节标记切分
            chapters = await self.split_by_markers(content)
            
            # 如果没有找到章节标记，尝试智能切分
            if len(chapters) <= 1:
                chapters = await self.smart_split(content)
            
            # 处理每个章节
            processed_chapters = []
            for i, chapter_content in enumerate(chapters):
                chapter = Chapter(
                    id=f"chapter_{i+1:03d}",
                    title=await self.extract_chapter_title(chapter_content),
                    content=chapter_content.strip(),
                    original_content=chapter_content.strip(),
                    chapter_number=i + 1,
                    word_count=len(chapter_content),
                    summary=await self.generate_chapter_summary(chapter_content)
                )
                processed_chapters.append(chapter)
            
            logger.info(f"章节切分完成，共 {len(processed_chapters)} 章")
            return processed_chapters
            
        except Exception as e:
            logger.error(f"切分章节失败: {e}")
            raise
    
    async def split_by_markers(self, content: str) -> List[str]:
        """按章节标记切分"""
        try:
            best_pattern = None
            best_matches = []
            
            # 找到最佳的章节模式
            for pattern in self.chapter_patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                if len(matches) > len(best_matches):
                    best_pattern = pattern
                    best_matches = matches
            
            if not best_matches:
                return [content]
            
            chapters = []
            start = 0
            
            for match in best_matches:
                if start < match.start():
                    # 添加前面的内容作为章节
                    chapters.append(content[start:match.start()])
                start = match.start()
            
            # 添加最后一个章节
            if start < len(content):
                chapters.append(content[start:])
            
            # 过滤掉太短的章节
            filtered_chapters = []
            for chapter in chapters:
                if len(chapter.strip()) > 100:  # 至少100字符
                    filtered_chapters.append(chapter)
            
            return filtered_chapters if filtered_chapters else [content]
            
        except Exception as e:
            logger.error(f"按标记切分失败: {e}")
            return [content]
    
    async def smart_split(self, content: str) -> List[str]:
        """智能切分章节"""
        try:
            # 使用LLM进行智能切分
            prompt = f"""
            请将以下小说内容智能切分成章节。请识别自然的章节分界点，
            考虑情节发展、场景转换、时间跨度等因素。
            
            内容：
            {content[:4000]}...
            
            请返回每个章节的起始位置（字符索引）。
            """
            
            messages = [
                {"role": "system", "content": "你是一个专业的小说编辑，擅长识别章节分界点。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.call_llm(messages, temperature=0.3)
            
            # 解析响应中的分界点
            split_points = self.parse_split_points(response)
            
            # 如果LLM分析失败，按固定长度切分
            if not split_points:
                split_points = self.split_by_length(content, 2000)
            
            # 根据分界点切分内容
            chapters = []
            for i in range(len(split_points)):
                start = split_points[i]
                end = split_points[i + 1] if i + 1 < len(split_points) else len(content)
                chapters.append(content[start:end])
            
            return chapters
            
        except Exception as e:
            logger.error(f"智能切分失败: {e}")
            return self.split_by_length(content, 2000)
    
    def parse_split_points(self, response: str) -> List[int]:
        """解析LLM返回的分界点"""
        try:
            # 尝试从响应中提取数字
            import re
            numbers = re.findall(r'\d+', response)
            split_points = [int(num) for num in numbers]
            split_points.sort()
            return split_points
        except Exception as e:
            logger.error(f"解析分界点失败: {e}")
            return []
    
    def split_by_length(self, content: str, max_length: int) -> List[str]:
        """按长度切分"""
        try:
            chapters = []
            start = 0
            
            while start < len(content):
                end = min(start + max_length, len(content))
                
                # 尝试在句号或段落结束处切分
                if end < len(content):
                    # 向后寻找句号
                    for i in range(end, min(end + 200, len(content))):
                        if content[i] in '。！？\n':
                            end = i + 1
                            break
                
                chapters.append(content[start:end])
                start = end
            
            return chapters
            
        except Exception as e:
            logger.error(f"按长度切分失败: {e}")
            return [content]
    
    async def extract_chapter_title(self, content: str) -> str:
        """提取章节标题"""
        try:
            # 先尝试正则匹配
            for pattern in self.chapter_patterns:
                match = re.search(pattern, content[:200], re.IGNORECASE)
                if match:
                    return match.group(0).strip()
            
            # 使用LLM提取标题
            prompt = f"""
            请为以下章节内容提取或生成一个合适的标题（不超过20字）：
            
            {content[:500]}...
            
            请只返回标题，不要包含其他内容。
            """
            
            messages = [
                {"role": "system", "content": "你是一个专业的小说编辑，擅长为章节命名。"},
                {"role": "user", "content": prompt}
            ]
            
            title = await self.call_llm(messages, temperature=0.7, max_tokens=100)
            return title.strip()
            
        except Exception as e:
            logger.error(f"提取章节标题失败: {e}")
            return "未命名章节"
    
    async def generate_chapter_summary(self, content: str) -> str:
        """生成章节摘要"""
        try:
            prompt = f"""
            请为以下章节内容生成一个简洁的摘要（不超过100字）：
            
            {content}
            
            摘要应该包含：
            1. 主要情节发展
            2. 重要人物和事件
            3. 关键信息点
            """
            
            messages = [
                {"role": "system", "content": "你是一个专业的小说编辑，擅长总结章节内容。"},
                {"role": "user", "content": prompt}
            ]
            
            summary = await self.call_llm(messages, temperature=0.5, max_tokens=200)
            return summary.strip()
            
        except Exception as e:
            logger.error(f"生成章节摘要失败: {e}")
            return "暂无摘要"
    
    async def analyze_structure(self, chapters: List[Chapter]) -> Dict[str, Any]:
        """分析小说结构"""
        try:
            self.current_task = "分析小说结构"
            
            # 分析章节长度分布
            lengths = [len(chapter.content) for chapter in chapters]
            avg_length = sum(lengths) / len(lengths)
            
            # 分析章节标题模式
            title_patterns = self.analyze_title_patterns(chapters)
            
            # 分析内容特征
            content_features = await self.analyze_content_features(chapters)
            
            structure_info = {
                "total_chapters": len(chapters),
                "average_chapter_length": avg_length,
                "min_chapter_length": min(lengths),
                "max_chapter_length": max(lengths),
                "title_patterns": title_patterns,
                "content_features": content_features,
                "quality_score": self.calculate_structure_quality(chapters)
            }
            
            return structure_info
            
        except Exception as e:
            logger.error(f"分析小说结构失败: {e}")
            return {}
    
    def analyze_title_patterns(self, chapters: List[Chapter]) -> Dict[str, int]:
        """分析标题模式"""
        patterns = {}
        
        for chapter in chapters:
            title = chapter.title
            
            # 分析标题模式
            if re.search(r'第\d+章', title):
                patterns["numbered_zh"] = patterns.get("numbered_zh", 0) + 1
            elif re.search(r'Chapter\s+\d+', title):
                patterns["numbered_en"] = patterns.get("numbered_en", 0) + 1
            elif re.search(r'第[零一二三四五六七八九十百千万]+章', title):
                patterns["chinese_numerals"] = patterns.get("chinese_numerals", 0) + 1
            else:
                patterns["custom"] = patterns.get("custom", 0) + 1
        
        return patterns
    
    async def analyze_content_features(self, chapters: List[Chapter]) -> Dict[str, Any]:
        """分析内容特征"""
        try:
            # 简单的内容特征分析
            total_words = sum(len(chapter.content) for chapter in chapters)
            
            features = {
                "total_words": total_words,
                "average_words_per_chapter": total_words / len(chapters),
                "estimated_reading_time": total_words / 300,  # 假设每分钟300字
            }
            
            return features
            
        except Exception as e:
            logger.error(f"分析内容特征失败: {e}")
            return {}
    
    def calculate_structure_quality(self, chapters: List[Chapter]) -> float:
        """计算结构质量评分"""
        try:
            score = 0.0
            
            # 章节数量合理性 (10-100章为最佳)
            chapter_count = len(chapters)
            if 10 <= chapter_count <= 100:
                score += 0.3
            elif 5 <= chapter_count <= 200:
                score += 0.2
            else:
                score += 0.1
            
            # 章节长度一致性
            lengths = [len(chapter.content) for chapter in chapters]
            avg_length = sum(lengths) / len(lengths)
            length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            
            # 方差越小，一致性越好
            if length_variance < avg_length * 0.5:
                score += 0.3
            elif length_variance < avg_length:
                score += 0.2
            else:
                score += 0.1
            
            # 章节标题规范性
            titled_chapters = sum(1 for chapter in chapters if chapter.title and chapter.title != "未命名章节")
            title_ratio = titled_chapters / len(chapters)
            score += 0.4 * title_ratio
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"计算结构质量评分失败: {e}")
            return 0.0 