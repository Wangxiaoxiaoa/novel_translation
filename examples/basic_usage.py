#!/usr/bin/env python3
"""
小说翻译修改器使用示例
Novel Translation Modifier Usage Examples
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import NovelTranslationModifier


async def example_basic_translation():
    """基础翻译示例"""
    print("🚀 基础翻译示例")
    print("=" * 50)
    
    # 初始化翻译器
    translator = NovelTranslationModifier("config/default.yaml")
    
    try:
        # 初始化系统
        if not await translator.initialize():
            print("❌ 系统初始化失败")
            return False
        
        # 翻译小说
        success = await translator.translate_novel(
            input_file="ori_novel/《全职高手》（精校版全本）作者：蝴蝶蓝.txt",
            target_language="en",
            title="The King's Avatar",
            author="Butterfly Blue",
            genre="Gaming/Esports"
        )
        
        if success:
            print("✅ 翻译完成！")
        else:
            print("❌ 翻译失败")
        
        return success
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        return False
    
    finally:
        await translator.cleanup()


async def example_multiple_languages():
    """多语言翻译示例"""
    print("🌍 多语言翻译示例")
    print("=" * 50)
    
    # 要翻译的语言列表
    target_languages = [
        ("en", "English"),
        ("ja", "Japanese"),
        ("ko", "Korean")
    ]
    
    translator = NovelTranslationModifier("config/default.yaml")
    
    try:
        if not await translator.initialize():
            print("❌ 系统初始化失败")
            return False
        
        input_file = "ori_novel/《诡秘之主》作者：爱潜水的乌贼.txt"
        
        for lang_code, lang_name in target_languages:
            print(f"\n📖 开始翻译为{lang_name}...")
            
            success = await translator.translate_novel(
                input_file=input_file,
                target_language=lang_code,
                title="Lord of Mysteries" if lang_code == "en" else "诡秘之主",
                author="Cuttlefish That Loves Diving",
                genre="Fantasy/Mystery"
            )
            
            if success:
                print(f"✅ {lang_name}翻译完成！")
            else:
                print(f"❌ {lang_name}翻译失败")
        
        return True
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        return False
    
    finally:
        await translator.cleanup()


async def example_custom_config():
    """自定义配置示例"""
    print("⚙️ 自定义配置示例")
    print("=" * 50)
    
    # 创建自定义配置
    custom_config = {
        "openai": {
            "api_key": "your-api-key-here",
            "model": "gpt-4-turbo-preview",
            "temperature": 0.8,  # 更高的创造性
            "max_tokens": 4000
        },
        "translation": {
            "source_language": "zh",
            "target_language": "en",
            "cultural_adaptation": True,
            "preserve_formatting": True
        },
        "quality_control": {
            "consistency_check": True,
            "cultural_appropriateness_check": True,
            "plot_continuity_check": True,
            "character_consistency_check": True
        },
        "output": {
            "format": "chapters",
            "directory": "output/custom",
            "include_metadata": True
        }
    }
    
    # 保存自定义配置
    import yaml
    config_path = "config/custom_example.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(custom_config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"📝 自定义配置已保存到: {config_path}")
    
    # 使用自定义配置
    translator = NovelTranslationModifier(config_path)
    
    try:
        if not await translator.initialize():
            print("❌ 系统初始化失败")
            return False
        
        # 显示系统状态
        await translator.show_system_status()
        
        return True
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        return False
    
    finally:
        await translator.cleanup()


async def example_batch_processing():
    """批量处理示例"""
    print("📚 批量处理示例")
    print("=" * 50)
    
    # 要处理的文件列表
    novel_files = [
        {
            "file": "ori_novel/《全职高手》（精校版全本）作者：蝴蝶蓝.txt",
            "title": "The King's Avatar",
            "author": "Butterfly Blue",
            "genre": "Gaming"
        },
        {
            "file": "ori_novel/《诡秘之主》作者：爱潜水的乌贼.txt",
            "title": "Lord of Mysteries",
            "author": "Cuttlefish That Loves Diving",
            "genre": "Fantasy"
        },
        {
            "file": "ori_novel/《我给末世主角们发编制》作者：边鹿.txt",
            "title": "I Give Protagonists Official Positions",
            "author": "Bian Lu",
            "genre": "Apocalypse"
        }
    ]
    
    translator = NovelTranslationModifier("config/default.yaml")
    
    try:
        if not await translator.initialize():
            print("❌ 系统初始化失败")
            return False
        
        successful_translations = 0
        total_novels = len(novel_files)
        
        for i, novel_info in enumerate(novel_files, 1):
            print(f"\n📖 处理第{i}/{total_novels}本小说: {novel_info['title']}")
            
            # 检查文件是否存在
            if not Path(novel_info['file']).exists():
                print(f"⚠️ 文件不存在，跳过: {novel_info['file']}")
                continue
            
            success = await translator.translate_novel(
                input_file=novel_info['file'],
                target_language="en",
                title=novel_info['title'],
                author=novel_info['author'],
                genre=novel_info['genre']
            )
            
            if success:
                successful_translations += 1
                print(f"✅ 《{novel_info['title']}》翻译完成")
            else:
                print(f"❌ 《{novel_info['title']}》翻译失败")
        
        print(f"\n📊 批量处理完成:")
        print(f"   - 总计: {total_novels} 本小说")
        print(f"   - 成功: {successful_translations} 本")
        print(f"   - 失败: {total_novels - successful_translations} 本")
        print(f"   - 成功率: {successful_translations/total_novels*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        return False
    
    finally:
        await translator.cleanup()


async def example_quality_analysis():
    """质量分析示例"""
    print("🔍 质量分析示例")
    print("=" * 50)
    
    translator = NovelTranslationModifier("config/default.yaml")
    
    try:
        if not await translator.initialize():
            print("❌ 系统初始化失败")
            return False
        
        # 模拟质量分析过程
        print("📊 执行翻译质量分析...")
        
        # 显示系统状态
        await translator.show_system_status()
        
        # 这里可以添加实际的质量分析逻辑
        print("✅ 质量分析完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        return False
    
    finally:
        await translator.cleanup()


async def run_all_examples():
    """运行所有示例"""
    print("🎯 运行所有示例")
    print("=" * 80)
    
    examples = [
        ("基础翻译示例", example_basic_translation),
        ("自定义配置示例", example_custom_config),
        ("质量分析示例", example_quality_analysis),
        # 注意：批量处理和多语言示例可能耗时较长，仅在需要时运行
        # ("多语言翻译示例", example_multiple_languages),
        # ("批量处理示例", example_batch_processing),
    ]
    
    successful_examples = 0
    total_examples = len(examples)
    
    for name, example_func in examples:
        print(f"\n🚀 开始运行: {name}")
        print("-" * 50)
        
        try:
            success = await example_func()
            if success:
                successful_examples += 1
                print(f"✅ {name} 运行成功")
            else:
                print(f"❌ {name} 运行失败")
        except Exception as e:
            print(f"💥 {name} 运行异常: {e}")
        
        print("-" * 50)
    
    print(f"\n📊 示例运行总结:")
    print(f"   - 总计: {total_examples} 个示例")
    print(f"   - 成功: {successful_examples} 个")
    print(f"   - 失败: {total_examples - successful_examples} 个")
    print(f"   - 成功率: {successful_examples/total_examples*100:.1f}%")


async def main():
    """主函数"""
    print("🌟 小说翻译修改器使用示例")
    print("=" * 80)
    
    # 检查配置文件
    config_file = Path("config/default.yaml")
    if not config_file.exists():
        print("❌ 配置文件不存在，请先运行主程序进行初始化")
        return
    
    # 检查是否有小说文件
    novel_dir = Path("ori_novel")
    if not novel_dir.exists() or not list(novel_dir.glob("*.txt")):
        print("ℹ️ 小说文件目录为空，某些示例可能无法运行")
        print("   请将小说文件放入 ori_novel/ 目录中")
    
    try:
        # 运行选定的示例
        print("\n请选择要运行的示例:")
        print("1. 基础翻译示例")
        print("2. 多语言翻译示例")
        print("3. 自定义配置示例")
        print("4. 批量处理示例")
        print("5. 质量分析示例")
        print("6. 运行所有示例")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-6): ").strip()
        
        if choice == "1":
            await example_basic_translation()
        elif choice == "2":
            await example_multiple_languages()
        elif choice == "3":
            await example_custom_config()
        elif choice == "4":
            await example_batch_processing()
        elif choice == "5":
            await example_quality_analysis()
        elif choice == "6":
            await run_all_examples()
        elif choice == "0":
            print("👋 再见！")
        else:
            print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        print(f"❌ 运行失败: {e}")


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 运行示例
    asyncio.run(main()) 