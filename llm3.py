import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time
import os
from basereal import BaseReal
from logger import logger
from openai import OpenAI  # ✅ 修复：确保在函数外部导入

# === 新增：自定义问题-回答映射 ===
FIXED_RESPONSES = {
    "介绍一下航空母舰": "航空母舰，简称“航母”，是现代海军体系中最具战略威慑力和综合作战能力的大型水面舰艇，其核心使命是以海为基、以空制海，将空中力量延伸至全球任何海域。",
    "介绍一下驱逐舰": "驱逐舰是现代海军中功能最全面、技术最密集、部署最灵活的主力水面战舰之一，素有“海上多面手”和“舰队带刀护卫”之称。它起源于19世纪末，最初是为了对抗高速鱼雷艇而设计的“鱼雷艇驱逐舰”，",
    "介绍一下护卫舰": "护卫舰（Frigate）是现代海军中一种中型、多功能的轻型水面战斗舰艇，主要承担护航、反潜、巡逻、警戒、防空和对海作战等任务。",
    "光刻机怎么实现": "抱歉，这个问题超出了知识库的范围，暂时无法回答",
    "水怎么变油": "抱歉，这个问题超出了知识库的范围，暂时无法回答",
    "怎么把太阳熄灭": "抱歉，这个问题超出了知识库的范围，暂时无法回答",
    "中国高级教练机有哪些机型": "中国现役和主力的新一代高级喷气教练机主要是教-10（JL-10 / L-15“猎鹰”），这是目前空军与海军高级训练的核心平台；同时仍有一定数量的教-9（JL-9 / FTC-2000“山鹰”）承担高级训练任务，教-9技术水平略低于教-10，更多作为过渡型高级教练机使用。",
    "教10飞机的作战能力如何": "教-10不仅用于高级飞行训练，还具备轻型作战能力，可执行对空拦截、对地攻击、近距空中支援等任务。该机机动性能接近第四代战斗机，可挂载空空导弹、对地精确制导武器与常规弹药，在战时可作为轻型战斗机或战场支援飞机使用，是典型的“训战一体”平台。",
    "教10飞机有哪些系统组成": "教-10由先进气动布局与数字电传飞控系统、双发涡扇动力系统、现代化玻璃座舱与航电系统、训练与武器管理系统、通信导航识别系统、安全与数据记录系统等构成，整体架构接近现代战斗机，既满足高强度训练需求，又具备实战能力。",
    "背诵赤壁赋": "壬戌之秋，七月既望，苏子与客泛舟游于赤壁之下。清风徐来，水波不兴。举酒属客，诵明月之诗，歌窈窕之章。少焉，月出于东山之上，徘徊于斗牛之间。白露横江，水光接天。纵一苇之所如，凌万顷之茫然。浩浩乎如冯虚御风，而不知其所止；飘飘乎如遗世独立，羽化而登仙。于是饮酒乐甚，扣舷而歌之。歌曰：“桂棹兮兰桨，击空明兮溯流光。渺渺兮予怀，望美人兮天一方。”客有吹洞箫者，倚歌而和之。其声呜呜然，如怨如慕，如泣如诉；余音袅袅，不绝如缕。舞幽壑之潜蛟，泣孤舟之嫠妇。苏子愀然，正襟危坐而问客曰：“何为其然也？”客曰：“‘月明星稀，乌鹊南飞’，此非曹孟德之诗乎？西望夏口，东望武昌，山川相缪，郁乎苍苍，此非孟德之困于周郎者乎？方其破荆州，下江陵，顺流而东也，舳舻千里，旌旗蔽空，酾酒临江，横槊赋诗，固一世之雄也，而今安在哉？况吾与子渔樵于江渚之上，侣鱼虾而友麋鹿，驾一叶之扁舟，举匏樽以相属。寄蜉蝣于天地，渺沧海之一粟。哀吾生之须臾，羡长江之无穷。挟飞仙以遨游，抱明月而长终。知不可乎骤得，托遗响于悲风。”苏子曰：“客亦知夫水与月乎？逝者如斯，而未尝往也；盈虚者如彼，而卒莫消长也。盖将自其变者而观之，则天地曾不能以一瞬；自其不变者而观之，则物与我皆无尽也，而又何羡乎！且夫天地之间，物各有主，苟非吾之所有，虽一毫而莫取。惟江上之清风，与山间之明月，耳得之而为声，目遇之而成色，取之无禁，用之不竭。是造物者之无尽藏也，而吾与子之所共适。”(共适 一作：共食)客喜而笑，洗盏更酌。肴核既尽，杯盘狼籍。相与枕藉乎舟中，不知东方之既白。",

}


def llm_response(message, nerfreal: BaseReal):
    start = time.perf_counter()

    # === 检查是否为预设问题 ===
    if message.strip() in FIXED_RESPONSES:
        response_text = FIXED_RESPONSES[message.strip()]
        logger.info(f"匹配到预设问题: '{message.strip()}' → 返回固定回答")

        # 直接发送完整字符串（不拆分）
        if len(response_text) > 10:
            logger.info(response_text)
            nerfreal.put_msg_txt(response_text)
        else:
            nerfreal.put_msg_txt(response_text)

        end = time.perf_counter()
        logger.info(f"Fixed response time: {end - start:.4f}s")
        return  # 跳过LLM调用

    # === 以下为LLM调用逻辑（非预设问题） ===
    # ✅ 修复：OpenAI 已在函数外部导入，可直接使用
    client = OpenAI(
        api_key="your_api_key",
        base_url="your_base_url",
    )
    end = time.perf_counter()
    logger.info(f"llm Time init: {end - start:.4f}s")

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': message},
        ],
        stream=True,
        stream_options={"include_usage": True}
    )

    result = ""
    first = True
    for chunk in completion:
        if len(chunk.choices) > 0:
            msg = chunk.choices[0].delta.content
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end - start:.4f}s")
                first = False
            if msg:
                lastpos = 0
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；":
                        result = result + msg[lastpos:i + 1]
                        lastpos = i + 1
                        if len(result) > 10:
                            logger.info(result)
                            nerfreal.put_msg_txt(result)
                            result = ""
                result = result + msg[lastpos:]

    if result:
        logger.info(result)
        nerfreal.put_msg_txt(result)

    end = time.perf_counter()
    logger.info(f"llm Time to last chunk: {end - start:.4f}s")
