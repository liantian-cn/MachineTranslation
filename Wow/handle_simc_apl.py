import os
import re
import json
from openai import OpenAI
from pathlib import Path
import requests
from typing import List, Dict, Generator, Any, Tuple
from extract_apl_terms import AplTermExtractor

LLM_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
LLM_API_URL = "https://api.siliconflow.cn/v1"
LLM_DEFAULT_MODEL = "Pro/deepseek-ai/DeepSeek-R1"

apl_term_extractor = AplTermExtractor()

simc_apl_manual = """
SimulationCraft APL 语法简明指南

=====================
一、基础结构
=====================
1. 动作组分类
   - actions.precombat: 战斗前执行的技能（仅非伤害技能）
   - actions: 常规循环技能组
   - 子动作组: 如 actions.ar（分支技能组）, actions.ar_cooldown（冷却技能组）

2. 动作定义
   - 初始化: actions.group_name=技能名（如 actions=auto_attack）
   - 追加动作: 用 += 
     例: actions+=/disrupt 

3. 调用子组
   - run_action_list,name=子组名: 跳转到子组（如切换天赋分支）
   - call_action_list: 调用子组后返回

=====================
二、条件逻辑
=====================
格式：if=条件1&条件2|条件3
1. 常用判断类型
   - Buff存在: buff.XXX.up
   - Debuff剩余时间: debuff.XXX.remains>5
   - 天赋启用: talent.XXX
   - 冷却状态: 
     - cooldown.XXX.up（可用）
     - cooldown.XXX.remains<5（剩余时间）

2. 逻辑运算符
   - &（与）: 同时满足
   - |（或）: 任意满足
   - !（非）: 取反

3. 特殊关键词
   - active_enemies>3: 当前目标数>3
   - fight_remains<30: 战斗剩余时间<30秒
   - fury>80: 资源（怒气）值>80

=====================
三、变量操作
=====================
定义与修改变量：
1. 初始化变量（带默认值）
   variable,name=XX,default=0,op=reset

2. 动态赋值
   variable,name=XX,op=set,value=表达式

示例:
variable,name=trinket1_steroids,value=trinket.1.has_stat.any_dps

=====================
四、关键参数
=====================
1. 非公共冷却技能
   use_off_gcd=1（不占GCD）
   例: /vengeful_retreat,use_off_gcd=1

2. 目标选择
   target_if=min:XX（优先选择最小XX的目标）
   例: target_if=min:debuff.burning_wound.remains

3. 冷却控制
   line_cd=1（技能调用间隔限制）
   charges=2（可用次数判定）

=====================
五、注释规范
=====================
以 # 开头，用于说明逻辑：
例:
# Spread Burning Wounds for uptime in multitarget scenarios
"""

pseudo_code_mannul = """

伪代码语法规则
1. 伪代码使用lua语法。
2. 伪代码存在多个内置函数，[]为可选参数。

 - CoolDown(技能名,[剩余冷却时间=0]) ：表明技能冷却状态
 - Prop(属性名,[属性参数1，属性参数1])： 返回某个属性的值。
 - Cast(技能名) ：表明释放技能
 - Goto(动作列表) ：表示跳转到到另一个函数

用例：
if Prop("应当打断", "focus") and Prop("与玩家敌对", "focus") then
    if CoolDown("心灵冰冻") and Prop("技能在施法距离", "心灵冰冻", "focus") then
        return Cast("心灵冰冻焦点")
    end
end
if Prop("玩家血量") < 50 then
    if Prop("玩家血量") < 80 then
        if Prop("符文能量") > 40 then
            return Cast("灵界打击")
        end
    end
end
if (Prop("Buff层数", "白骨之盾") < 5) or (Prop("Buff剩余时间", "白骨之盾") < 6) then

    if CoolDown("死神印记", 300) and (not Prop("玩家存在Buff", "破灭")) then
        return Cast("死神印记")
    end

    if Prop("技能在施法距离", "死神的抚摩", "target") and CoolDown("死神的抚摩", 300) then
        return Cast("死神的抚摩")
    end

    if Prop("玩家存在Buff", "破灭") and Prop("技能在施法距离", "精髓分裂", "target") and Prop("目标血量") > 10 then
        return Cast("精髓分裂")
    end

    if Prop("技能在施法距离", "精髓分裂", "target") then
        return Cast("精髓分裂")
    end
end

"""



def clean_json_string(raw_string):
    """
    清理被Markdown代码块包裹的JSON字符串
    示例输入： ```json ["arg1", "arg2"] ```
    示例输出： ["arg1", "arg2"]
    """
    # 匹配被 ```json 或 ``` 包裹的内容
    pattern = r'^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$'

    # 使用正则替换，保留分组1的内容（即实际JSON内容）
    cleaned = re.sub(pattern, r'\1', raw_string, flags=re.DOTALL)

    return cleaned.strip()



def chat_completion_stream(messages: List[Dict[str, str]],
                           api_key: str = LLM_API_KEY,
                           model: str = LLM_DEFAULT_MODEL,
                           api_url: str = LLM_API_URL,
                           timeout: int = 600,
                           max_tokens: int = 16384,
                           temperature: float = 0.7,
                           top_p: float = 0.7,
                           ) -> Tuple[str, str]:
    client = OpenAI(api_key=api_key, base_url=api_url)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    reasoning_content = []
    content = []

    for chunk in response:
        delta = chunk.choices[0].delta
        current_reason = getattr(delta, "reasoning_content", None)  # 安全获取属性
        current_content = getattr(delta, "content", None)

        if current_reason:
            print(current_reason, end="", flush=True)
            reasoning_content.append(current_reason)
        if current_content:  # 使用独立判断而非elif
            print(current_content, end="", flush=True)
            content.append(current_content)

    print("\n")
    return "".join(reasoning_content), "".join(content)


def split_into_paragraphs(text: str) -> List[str]:
    """将文本按空行分割成段落。

    Args:
        text: 需要分割的文本内容。

    Returns:
        List[str]: 每个元素为一个段落的内容。
    """
    lines = text.split('\n')
    paragraphs = []
    current_paragraph = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line == '':  # 空行
            if current_paragraph:
                paragraphs.append('\n'.join(current_paragraph))
                current_paragraph = []
        else:
            current_paragraph.append(line)

    if current_paragraph:  # 处理最后一段
        paragraphs.append('\n'.join(current_paragraph))

    return paragraphs


def split_paragraph_if_too_long(paragraph: str) -> List[str]:
    """如果段落过长，将其分割成多个小段落。

    Args:
        paragraph: 需要分割的段落内容。

    Returns:
        List[str]: 分割后的小段落列表。
    """
    lines = paragraph.split('\n')
    if len(lines) < 7:
        return [paragraph]

    # 计算每个小段落的行数
    num_lines = len(lines)
    num_splits = (num_lines + 5) // 6  # 每个小段落最多6行
    split_size = num_lines // num_splits
    remainder = num_lines % num_splits

    split_paragraphs = []
    start = 0
    for i in range(num_splits):
        end = start + split_size + (1 if i < remainder else 0)
        split_paragraphs.append('\n'.join(lines[start:end]))
        start = end

    return split_paragraphs


def split_file_into_paragraphs(file_path: Path) -> List[str]:
    """读取文件内容并按空行分割成段落。如果段落过长，进一步分割。

    Args:
        file_path: pathlib.Path 对象，文件路径。

    Returns:
        List[str]: 每个元素为一个段落的内容。

    Raises:
        FileNotFoundError: 如果文件不存在。
        IOError: 如果无法打开或读取文件。
    """
    with open(file_path, 'rt', encoding='utf-8') as f:
        content = f.read()

    paragraphs = split_into_paragraphs(content)
    final_paragraphs = []

    for paragraph in paragraphs:
        final_paragraphs.extend(split_paragraph_if_too_long(paragraph))

    return final_paragraphs


def extract_keywords_from_paragraph(paragraph_text: str) -> List[str]:
    term_strings = []

    messages = [
        {"role": "system", "content": "你是一个专业的翻译助理。你需要帮我找出一些术语，用于后续交给翻译大师翻译。"
                                      "关于文本：\n"
                                      "1. 这是电子游戏《魔兽世界》中的插件《SimulationCraft》的APL语法，这包含了一些战斗逻辑。"
                                      "你要做的：\n"
                                      "1. 找出明显的属于《魔兽世界》这个游戏游戏的技能、状态名称。\n"
                                      "2. 忽略gcd、cooldown、talent这样的很多游戏公用的游戏术语。\n"
                                      "3. 忽略诸如‘value’、‘name’、‘use’、‘set’这样的编程术语。\n"
                                      "4. 忽略所有长度小于等于3的单词。这一般是缩写。\n"
                                      "5. 根据APL语法手册，忽略那些显而易见的语法用词。"
                                      "注意：\n"
                                      "1. 下划线是代替空格存在的，下划线连接的，往往是一整个术语，当作整体来判断。\n、"
                                      "2. 像‘.’这样的符号，往往是多个术语的组合，需要分开判断。\n"
                                      "返回值要求：\n"
                                      "1. 返回一行一个术语、去掉特殊符号，小写\n"
                                      "2. 格式为：[\"word1\", \"word2\",..]\n"
                                      "3. 若文本完全可以理解，不存在专有名词，也要返回空列表。\n"
                                      f"APL语法手册：\n[APL语法手册开始]{simc_apl_manual}\n[APL语法手册结束]"},
        {"role": "user", "content": paragraph_text},
    ]
    think, result = chat_completion_stream(messages, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", temperature=0.3)

    words = json.loads(clean_json_string(result).strip())
    words = [word.strip().lower() for word in words]
    for word in words:
        results = apl_term_extractor.find_references(word, top_n=3)
        for idx, item in enumerate(results, 1):
            print(f"\n#{idx} [Score: {item.get('score', 0):.3f}]")
            print(f"{item.get('FullString_enUS', '')} => {item.get('FullString_zhCN', '')}")
            term_strings.append(f"{item.get('FullString_enUS', '')} => {item.get('FullString_zhCN', '')}")

    return sorted(list(set(term_strings)))


def clean_keywords_using_llm(term_strings: List[str]) -> str:
    """使用LLM清理关键词列表。
    """

    messages = [
        {"role": "system", "content": "你是一个专业的文本处理专家，擅长清理文本。\n"
                                      "首要工作：清理用户提交的文本。\n"
                                      "注意事项：\n"
                                      "1. 文本为游戏技能的翻译，已经是准确无误了，不需要任何翻译和润色。\n"
                                      "2. 提交的格式是：英文技能名称：英文技能说明   =>  中文技能名称：中文技能说明\n"
                                      "清理工作内容：\n"
                                      "1. 删掉各种奇怪的符号。\n"
                                      "2. 合并相同技能名称的翻译。\n"
                                      "3. 保持格式：英文技能名称：英文技能说明   =>  中文技能名称：中文技能说明。\n"
                                      "4. 去掉各种$代表的变量，按顺序用x、y、z、a、b、c这样的常见变量代替。\n"
                                      "5. 去掉|cxxxxxx这种颜色标识。\n"
                                      "6. 删除看起来是对其他文本引用的那些标记。\n"
                                      "7. 删除那些看起来是代表某些数字的引用，并用常见变量代替。\n"
                                      "返回值要求:"
                                      "1. 纯文本，不可使用markdown格式。"},
        {"role": "user", "content": "\n".join(term_strings)},
    ]
    think, result = chat_completion_stream(messages, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", temperature=0.01)
    return result


def pseudo_code_generator(original_code: str, translation_reference: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个编程大师。能看懂任何不常见的脚本语法。\n"
                                      "现在用户提交了一个叫做APL脚本语言的片段。你要为用户生成一个逻辑清晰的伪代码，伪代码是LUA脚本风格的。\n"
                                      "1. 已知APL脚本语言，是电子游戏《魔兽世界》中的模拟器《SimulationCraft》所使用的，是模拟游戏玩家的攻击技能逻辑。\n"
                                      "2. 因为存在大量游戏内的术语，所以提供了一些游戏内翻译作为参考。\n"
                                      "3. 请参考SimulationCraft APL 语法简明指南，但也需要要自行理解一些。"
                                      "4. 伪代码除了if else这样的程序逻辑外，函数名和 变量名使用中文，具体参考伪代码说明\n"
                                      "5. 每一行APL代码，对应一段伪代码。每行独立。\n"
                                      "6. 避免多个if条件写在一行，降低可读性。使用if嵌套if，可读性更好。"
                                      "输出：直接输出伪代码，不要多余的说明。\n"
                                      f"\n\n[翻译参考信息]{translation_reference}[/翻译参考信息]\n\n"
                                      f"\n\n[语法简明指南]{simc_apl_manual}[/语法简明指南]\n\n"
                                      f""},
        {"role": "user", "content": original_code},
    ]
    think, result = chat_completion_stream(messages, model="Pro/deepseek-ai/DeepSeek-R1", temperature=0.4)
    return result


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    apl_file_name = BASE_DIR.joinpath("simc").joinpath("ActionPriorityLists").joinpath("demonhunter_havoc.simc")

    all_pseudo_code = ""
    paragraphs = split_file_into_paragraphs(apl_file_name)

    for paragraph in paragraphs:
        print(paragraph)
        paragraph_keywords = extract_keywords_from_paragraph(paragraph)
        print("\n".join(paragraph_keywords))
        print("====================")
        print("====================")
        print("====================")
        paragraph_cleaned_keywords = clean_keywords_using_llm(paragraph_keywords)
        print(paragraph_cleaned_keywords)

        paragraph_pseudo_code = pseudo_code_generator(paragraph, paragraph_cleaned_keywords)
        print(paragraph_pseudo_code)
        all_pseudo_code += paragraph
        all_pseudo_code += paragraph_pseudo_code
    with open(BASE_DIR.joinpath("output.txt"), "w", encoding="utf-8") as f:
        f.write(all_pseudo_code)
