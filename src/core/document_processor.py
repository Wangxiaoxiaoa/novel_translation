"""
文档处理器 - 整合各种文档解析功能
Document Processor - Integrates various document parsing functions
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from loguru import logger
import aiofiles
from datetime import datetime

from ..models.base import Novel, Chapter, DocumentFormat, LanguageCode, ProcessingStatus
from ..agents.parser_agent import ParserAgent
from ..agents.memory_agent import MemoryAgent


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.parser_agent = None
        self.memory_agent = None
        
        # 支持的文档格式
        self.supported_formats = {
            '.txt': DocumentFormat.TXT,
            '.pdf': DocumentFormat.PDF,
            '.docx': DocumentFormat.DOCX
        }
        
        # 处理历史
        self.processing_history = []
        
    async def initialize(self, parser_agent: ParserAgent, memory_agent: MemoryAgent) -> bool:
        """初始化文档处理器"""
        try:
            logger.info("初始化文档处理器...")
            
            self.parser_agent = parser_agent
            self.memory_agent = memory_agent
            
            # 确保输出目录存在
            output_dir = Path(self.config.get("output_directory", "output"))
            output_dir.mkdir(exist_ok=True)
            
            # 确保缓存目录存在
            cache_dir = Path(self.config.get("cache_directory", "cache"))
            cache_dir.mkdir(exist_ok=True)
            
            logger.info("文档处理器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化文档处理器失败: {e}")
            return False
    
    async def process_document(self, file_path: str, **kwargs) -> Novel:
        """处理文档"""
        try:
            logger.info(f"开始处理文档: {file_path}")
            
            # 验证文件
            file_info = await self.validate_document(file_path)
            if not file_info:
                raise ValueError(f"无效的文档文件: {file_path}")
            
            # 解析文档
            parsed_data = await self.parse_document(file_path, file_info)
            
            # 创建小说对象
            novel = await self.create_novel_from_data(parsed_data, **kwargs)
            
            # 切分章节
            chapters = await self.split_chapters(novel, parsed_data)
            
            # 分析内容
            await self.analyze_content(novel, chapters)
            
            # 构建记忆
            await self.build_memory(novel, chapters)
            
            # 记录处理历史
            self.processing_history.append({
                "file_path": file_path,
                "novel_id": novel.title,
                "processed_at": datetime.now().isoformat(),
                "chapters_count": len(chapters),
                "success": True
            })
            
            logger.info(f"文档处理完成: {file_path}, 共 {len(chapters)} 章")
            return novel
            
        except Exception as e:
            logger.error(f"处理文档失败: {e}")
            
            # 记录失败历史
            self.processing_history.append({
                "file_path": file_path,
                "error": str(e),
                "processed_at": datetime.now().isoformat(),
                "success": False
            })
            
            raise
    
    async def validate_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """验证文档"""
        try:
            file_path = Path(file_path)
            
            # 检查文件是否存在
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                return None
            
            # 检查文件格式
            file_extension = file_path.suffix.lower()
            if file_extension not in self.supported_formats:
                logger.error(f"不支持的文件格式: {file_extension}")
                return None
            
            # 检查文件大小
            file_size = file_path.stat().st_size
            max_size = self.config.get("max_file_size", 100 * 1024 * 1024)  # 100MB
            if file_size > max_size:
                logger.error(f"文件过大: {file_size} bytes > {max_size} bytes")
                return None
            
            # 检查文件权限
            if not os.access(file_path, os.R_OK):
                logger.error(f"文件不可读: {file_path}")
                return None
            
            return {
                "path": str(file_path),
                "format": self.supported_formats[file_extension],
                "size": file_size,
                "extension": file_extension,
                "name": file_path.stem
            }
            
        except Exception as e:
            logger.error(f"验证文档失败: {e}")
            return None
    
    async def parse_document(self, file_path: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """解析文档"""
        try:
            # 检查缓存
            cached_data = await self.get_cached_parse_result(file_path)
            if cached_data:
                logger.info(f"使用缓存的解析结果: {file_path}")
                return cached_data
            
            # 调用解析智能体
            parse_result = await self.parser_agent.parse_document(file_path)
            
            # 缓存结果
            await self.cache_parse_result(file_path, parse_result)
            
            return parse_result
            
        except Exception as e:
            logger.error(f"解析文档失败: {e}")
            raise
    
    async def create_novel_from_data(self, parsed_data: Dict[str, Any], **kwargs) -> Novel:
        """从解析数据创建小说对象"""
        try:
            # 从解析数据中提取信息
            title = parsed_data.get("title", "未知标题")
            author = parsed_data.get("author", "未知作者")
            genre = parsed_data.get("genre", "未知类型")
            language = parsed_data.get("detected_language", "zh")
            
            # 从kwargs中获取用户指定的信息
            title = kwargs.get("title", title)
            author = kwargs.get("author", author)
            genre = kwargs.get("genre", genre)
            language = kwargs.get("language", language)
            
            # 创建小说对象
            novel = Novel(
                title=title,
                author=author,
                genre=genre,
                language=LanguageCode(language),
                description=kwargs.get("description", ""),
                metadata={
                    "file_path": parsed_data.get("file_path", ""),
                    "file_format": parsed_data.get("format", ""),
                    "word_count": parsed_data.get("word_count", 0),
                    "char_count": parsed_data.get("char_count", 0),
                    "estimated_chapters": parsed_data.get("estimated_chapters", 0),
                    "processed_at": datetime.now().isoformat()
                }
            )
            
            return novel
            
        except Exception as e:
            logger.error(f"创建小说对象失败: {e}")
            raise
    
    async def split_chapters(self, novel: Novel, parsed_data: Dict[str, Any]) -> List[Chapter]:
        """切分章节"""
        try:
            # 调用解析智能体切分章节
            chapters = await self.parser_agent.split_chapters({
                "content": parsed_data.get("content", ""),
                "title": novel.title,
                "author": novel.author,
                "language": novel.language
            })
            
            # 更新小说的章节列表
            novel.chapters = chapters
            
            return chapters
            
        except Exception as e:
            logger.error(f"切分章节失败: {e}")
            raise
    
    async def analyze_content(self, novel: Novel, chapters: List[Chapter]) -> Dict[str, Any]:
        """分析内容"""
        try:
            logger.info(f"开始分析内容: {novel.title}")
            
            # 并行分析各个章节
            analysis_tasks = []
            for chapter in chapters:
                task = self.analyze_chapter(chapter)
                analysis_tasks.append(task)
            
            # 等待所有分析完成
            chapter_analyses = await asyncio.gather(*analysis_tasks)
            
            # 汇总分析结果
            overall_analysis = await self.summarize_analysis(novel, chapter_analyses)
            
            return overall_analysis
            
        except Exception as e:
            logger.error(f"分析内容失败: {e}")
            raise
    
    async def analyze_chapter(self, chapter: Chapter) -> Dict[str, Any]:
        """分析单个章节"""
        try:
            # 提取角色
            characters = await self.extract_characters(chapter.content)
            chapter.characters = characters
            
            # 提取地点
            locations = await self.extract_locations(chapter.content)
            chapter.locations = locations
            
            # 提取物品/术语
            items = await self.extract_items(chapter.content)
            chapter.items = items
            
            # 提取术语
            terminologies = await self.extract_terminologies(chapter.content)
            chapter.terminologies = terminologies
            
            # 生成摘要
            if not chapter.summary:
                chapter.summary = await self.generate_chapter_summary(chapter.content)
            
            return {
                "chapter_id": chapter.id,
                "characters": characters,
                "locations": locations,
                "items": items,
                "terminologies": terminologies,
                "summary": chapter.summary
            }
            
        except Exception as e:
            logger.error(f"分析章节失败: {e}")
            return {"chapter_id": chapter.id, "error": str(e)}
    
    async def extract_characters(self, content: str) -> List[str]:
        """提取角色"""
        try:
            # 这里可以使用NLP技术或正则表达式提取角色名
            # 暂时使用简单的实现
            characters = []
            
            # 使用一些常见的中文姓名模式
            import re
            
            # 简单的姓名模式
            name_patterns = [
                r'[\u4e00-\u9fff]{2,4}(?=说|道|想|问|答|笑|哭|叹|怒)',
                r'[\u4e00-\u9fff]{2,4}(?=的|是|在|有|没|会|能|要|去|来)',
            ]
            
            for pattern in name_patterns:
                matches = re.findall(pattern, content)
                characters.extend(matches)
            
            # 去重并过滤
            characters = list(set(characters))
            characters = [name for name in characters if len(name) >= 2 and len(name) <= 4]
            
            return characters[:20]  # 限制数量
            
        except Exception as e:
            logger.error(f"提取角色失败: {e}")
            return []
    
    async def extract_locations(self, content: str) -> List[str]:
        """提取地点"""
        try:
            # 使用正则表达式提取地点
            import re
            
            locations = []
            
            # 地点模式
            location_patterns = [
                r'[\u4e00-\u9fff]{2,6}(?=城|镇|村|山|河|湖|海|宫|殿|楼|阁|院|府|寺|庙)',
                r'[\u4e00-\u9fff]{2,6}(?=国|州|省|郡|县|区)',
            ]
            
            for pattern in location_patterns:
                matches = re.findall(pattern, content)
                locations.extend(matches)
            
            # 去重
            locations = list(set(locations))
            
            return locations[:15]  # 限制数量
            
        except Exception as e:
            logger.error(f"提取地点失败: {e}")
            return []
    
    async def extract_items(self, content: str) -> List[str]:
        """提取物品"""
        try:
            # 使用正则表达式提取物品
            import re
            
            items = []
            
            # 物品模式
            item_patterns = [
                r'[\u4e00-\u9fff]{2,6}(?=剑|刀|枪|矛|戟|弓|箭|盾|甲|袍|衣|帽|鞋|靴)',
                r'[\u4e00-\u9fff]{2,6}(?=丹|药|符|诀|经|书|卷|册|图|谱)',
            ]
            
            for pattern in item_patterns:
                matches = re.findall(pattern, content)
                items.extend(matches)
            
            # 去重
            items = list(set(items))
            
            return items[:10]  # 限制数量
            
        except Exception as e:
            logger.error(f"提取物品失败: {e}")
            return []
    
    async def extract_terminologies(self, content: str) -> List[str]:
        """提取术语"""
        try:
            # 使用正则表达式提取术语
            import re
            
            terminologies = []
            
            # 术语模式
            terminology_patterns = [
                r'[\u4e00-\u9fff]{2,6}(?=功|法|诀|术|道|心|式|掌|拳|腿|指|爪)',
                r'[\u4e00-\u9fff]{2,6}(?=境|界|级|阶|层|品|等|段)',
            ]
            
            for pattern in terminology_patterns:
                matches = re.findall(pattern, content)
                terminologies.extend(matches)
            
            # 去重
            terminologies = list(set(terminologies))
            
            return terminologies[:10]  # 限制数量
            
        except Exception as e:
            logger.error(f"提取术语失败: {e}")
            return []
    
    async def generate_chapter_summary(self, content: str) -> str:
        """生成章节摘要"""
        try:
            # 这里可以使用LLM生成摘要
            # 暂时使用简单的实现
            
            # 取前200字符作为摘要
            summary = content[:200].strip()
            if len(content) > 200:
                summary += "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"生成章节摘要失败: {e}")
            return "摘要生成失败"
    
    async def summarize_analysis(self, novel: Novel, chapter_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总分析结果"""
        try:
            all_characters = set()
            all_locations = set()
            all_items = set()
            all_terminologies = set()
            
            # 汇总所有章节的分析结果
            for analysis in chapter_analyses:
                if "error" not in analysis:
                    all_characters.update(analysis.get("characters", []))
                    all_locations.update(analysis.get("locations", []))
                    all_items.update(analysis.get("items", []))
                    all_terminologies.update(analysis.get("terminologies", []))
            
            # 更新小说的全局信息
            novel.metadata.update({
                "total_characters": len(all_characters),
                "total_locations": len(all_locations),
                "total_items": len(all_items),
                "total_terminologies": len(all_terminologies),
                "analysis_completed_at": datetime.now().isoformat()
            })
            
            return {
                "characters": list(all_characters),
                "locations": list(all_locations),
                "items": list(all_items),
                "terminologies": list(all_terminologies),
                "total_chapters": len(chapter_analyses),
                "successful_analyses": len([a for a in chapter_analyses if "error" not in a])
            }
            
        except Exception as e:
            logger.error(f"汇总分析结果失败: {e}")
            return {"error": str(e)}
    
    async def build_memory(self, novel: Novel, chapters: List[Chapter]) -> bool:
        """构建记忆"""
        try:
            logger.info(f"开始构建记忆: {novel.title}")
            
            # 存储角色信息
            for chapter in chapters:
                for char_name in chapter.characters:
                    await self.memory_agent.store_character_info({
                        "name": char_name,
                        "info": {"name": char_name, "novel": novel.title},
                        "chapter_context": chapter.content
                    })
            
            # 存储情节信息
            for chapter in chapters:
                await self.memory_agent.store_plot_info({
                    "chapter_id": chapter.id,
                    "summary": chapter.summary,
                    "details": {
                        "title": chapter.title,
                        "characters": chapter.characters,
                        "locations": chapter.locations,
                        "items": chapter.items
                    }
                })
            
            logger.info(f"记忆构建完成: {novel.title}")
            return True
            
        except Exception as e:
            logger.error(f"构建记忆失败: {e}")
            return False
    
    async def get_cached_parse_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取缓存的解析结果"""
        try:
            if not self.config.get("enable_cache", True):
                return None
            
            cache_dir = Path(self.config.get("cache_directory", "cache"))
            cache_file = cache_dir / f"{Path(file_path).stem}.json"
            
            if not cache_file.exists():
                return None
            
            # 检查缓存时间
            cache_ttl = self.config.get("cache_ttl", 86400)  # 24小时
            file_mtime = cache_file.stat().st_mtime
            if (datetime.now().timestamp() - file_mtime) > cache_ttl:
                return None
            
            # 读取缓存
            import json
            async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.loads(await f.read())
            
            logger.info(f"使用缓存的解析结果: {file_path}")
            return cache_data
            
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            return None
    
    async def cache_parse_result(self, file_path: str, parse_result: Dict[str, Any]) -> bool:
        """缓存解析结果"""
        try:
            if not self.config.get("enable_cache", True):
                return False
            
            cache_dir = Path(self.config.get("cache_directory", "cache"))
            cache_file = cache_dir / f"{Path(file_path).stem}.json"
            
            # 保存缓存
            import json
            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(parse_result, ensure_ascii=False, indent=2))
            
            logger.info(f"解析结果已缓存: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"缓存解析结果失败: {e}")
            return False
    
    async def export_novel(self, novel: Novel, export_format: str = "json") -> str:
        """导出小说"""
        try:
            output_dir = Path(self.config.get("output_directory", "output"))
            
            if export_format == "json":
                output_file = output_dir / f"{novel.title}.json"
                
                # 导出为JSON
                import json
                async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(novel.dict(), ensure_ascii=False, indent=2))
                
            elif export_format == "txt":
                output_file = output_dir / f"{novel.title}.txt"
                
                # 导出为TXT
                async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                    await f.write(f"标题: {novel.title}\n")
                    await f.write(f"作者: {novel.author}\n")
                    await f.write(f"类型: {novel.genre}\n\n")
                    
                    for chapter in novel.chapters:
                        await f.write(f"\n{chapter.title}\n")
                        await f.write("=" * 50 + "\n")
                        await f.write(chapter.content)
                        await f.write("\n\n")
            
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")
            
            logger.info(f"小说导出完成: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"导出小说失败: {e}")
            raise
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        try:
            total_processed = len(self.processing_history)
            successful_processed = len([h for h in self.processing_history if h.get("success", False)])
            failed_processed = total_processed - successful_processed
            
            return {
                "total_processed": total_processed,
                "successful_processed": successful_processed,
                "failed_processed": failed_processed,
                "success_rate": successful_processed / total_processed if total_processed > 0 else 0,
                "recent_processing": self.processing_history[-5:] if self.processing_history else []
            }
            
        except Exception as e:
            logger.error(f"获取处理统计信息失败: {e}")
            return {"error": str(e)} 