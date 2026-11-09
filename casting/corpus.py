# -*- coding: utf-8 -*-
"""正文解析层：章节切分、多引号抽取、人名挖掘、台词归因、性别/年龄证据。

设计约束：
- 不依赖任何单本书的人工表（书名、人名、别名、丑角表都不写死）。
- 书名/人名/别名全部从正文统计里长出来；合并动作输出到 name_merges，供人工复核。
- 引号样式按书自动探测，识别不出的段落计入 coverage，不静默丢弃。
- 性别证据分层计分；证据不足标 unknown 并上报，不默认丢给旁白。
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CJK = r"\u4e00-\u9fa5"

# --------------------------------------------------------------------------- 章节
CH_HEAD_RE = re.compile(
    r"^\s*(第[0-9零〇一二三四五六七八九十百千万两]{1,10}[章回节篇]|"
    r"楔子|序章|序言|引子|尾声|后记|番外[一二三四五六七八九十0-9]*|"
    r"Chapter\s+[0-9IVXLCivxlc]+)\s*(.*)$"
)
VOL_HEAD_RE = re.compile(r"^\s*(第[0-9零〇一二三四五六七八九十百千万两]{1,8}卷)\s*(.*)$")

# --------------------------------------------------------------------------- 引号样式
QUOTE_STYLES = {
    "cn": ("“", "”"),
    "jp": ("「", "」"),
    "jp2": ("『", "』"),
    "ascii": ('"', '"'),
}
STYLE_RE = {
    "cn": re.compile(r"“([^”]{1,1500})”"),
    "jp": re.compile(r"「([^」]{1,1500})」"),
    "jp2": re.compile(r"『([^』]{1,1500})』"),
    "ascii": re.compile(r'"([^"]{1,1500})"'),
}

# --------------------------------------------------------------------------- 言说动词
SPEECH = (
    "说道", "笑道", "问道", "答道", "叫道", "嚷道", "喊道", "骂道", "叹道", "喝道",
    "沉声道", "低声道", "高声道", "冷笑道", "开口道", "接口道", "喃喃道", "自语道",
    "回答道", "追问道", "反问道", "吩咐道", "解释道", "劝道", "回道", "应道", "喝道",
    "道", "说", "问", "答", "嚷", "喊", "叫", "吼", "骂", "叹", "哼", "念", "劝",
    "回答", "追问", "反问", "吩咐", "解释", "补充", "提醒", "低语", "嘀咕",
)
SPEECH_ALT = "|".join(sorted(SPEECH, key=len, reverse=True))
VERB_RE = re.compile("(?:" + SPEECH_ALT + ")")

FUNC_CHARS = set(
    "的了着过是在不我你他她它这那有和就都也又很吗呢吧啊么们个把被让给对从向与及或"
    "但因为所以如果虽然可以没有将已还未曾才是只更最再哪谁什么怎样如此"
)
STOP_NAMES = {
    "众人", "大家", "有人", "自己", "对方", "此人", "那人", "一人", "二人", "三人",
    "所有人", "其他人", "别人", "旁人", "两人", "几个人", "一个人", "这些人", "学生们",
    "同学们", "仆从", "随从", "侍卫", "士兵", "军士", "百姓", "母亲", "父亲", "儿子",
    "女儿", "妻子", "丈夫", "兄弟", "哥哥", "弟弟", "姐姐", "妹妹", "先生", "大人",
    "公子", "姑娘", "夫人", "女子", "男子", "老人", "少年", "青年", "孩子", "小孩",
    "皇帝", "将军", "大王", "陛下", "殿下", "微臣", "老夫", "老臣", "本王", "寡人",
    "朕", "臣妾", "妾身", "小人", "在下", "属下", "声音", "一个声音", "轻柔声音",
    "立刻", "之后", "一愣", "一边", "面前", "出来", "厉声", "实话", "传言", "头目",
    "一声", "门口", "进来", "口气", "身边", "诧异", "胡说", "一眼", "肯定", "终于",
    "必须", "句话", "走来", "时候", "前去", "片刻", "打断", "谁知", "能够", "上来",
    "扭头", "笑道", "消息", "兵卒", "正要", "传来", "妇人", "坐下", "事情", "刚刚",
    "勉强", "外面", "咬牙", "人来", "军官", "拒绝", "止步", "一次", "哈哈", "通报",
    "下去", "小心", "挥手", "一下", "无法", "身后", "心中", "脸上", "手中", "脚下",
    "回头", "抬头", "低头", "点头", "摇头", "转身", "起身", "上前", "后退", "跪下",
    "站起", "走出", "走进", "低声", "高声", "大声", "小声", "轻声", "沉声", "冷声",
    "笑着", "冷笑", "微笑", "苦笑", "大笑", "连忙", "急忙", "赶紧", "果然", "竟然",
    "忽然", "突然", "随即", "随后", "于是", "因此", "但是", "可是", "不过", "而且",
    "并且", "然后", "接着", "继续", "仍然", "依然", "已经", "曾经", "正在", "将要",
    "可以", "应该", "知道", "觉得", "认为", "看到", "听到", "想到", "说道", "问道",
    "答道", "叫道", "嚷道", "喊道", "骂道", "叹道", "喝道", "沉声道", "低声道",
    "冷笑道", "开口道", "愿意", "办法", "抱怨", "缤纷", "主意", "事情", "资格",
    "计划", "命令", "沉默", "意外", "敷衍", "当面", "同时", "许多", "之前", "反而",
    "外人", "前者", "后者", "此时", "这时", "那时", "一时", "无人", "一些", "一点",
    "一直", "一起", "一样", "进城", "出城", "你们", "我们", "他们", "她们", "不是",
    "就是", "还是", "如实", "照实", "直接", "干脆", "抬手", "摆手", "拱手", "抱拳",
    "欠身", "低低", "轻轻", "淡淡", "冷冷", "缓缓", "慢慢", "默默", "悄悄", "高声",
    "正色", "哼道", "喊道", "问道", "答道", "叫道", "笑意", "神色", "语气", "目光",
    "任何", "明明", "连连", "哈哈", "胡说", "实话", "明知", "严肃", "投奔",
    "明白", "明知故", "能否", "时间", "房间", "单独", "云游", "相迎", "仰头",
    "胡乱", "别乱", "公子请", "王请", "宦者", "师父", "师傅", "师弟", "师兄", "师姐",
    "师妹", "士兵", "糊里糊涂", "自言自语", "何人",
}
BAD_END = set(
    "知觉得看听想要会能来去出进上下过到问道答笑叫喊说着了是的在很么呢吧啊"
    "实诚话事时处边面前后中里外开起住掉给让被把对向从为以及或但却就都也又更最"
    "再才只已曾将要可该点摇转回抬低拱抱欠挥摆站坐跪走跑停死活好坏多少大小高低"
    "真假对错难易新旧快慢远近强弱冷热轻重急缓怒惊喜悲愁苦甜香臭亮暗妻交法怨纷"
    "意样声地得着过们么"
)
PLACE_SUFFIX = (
    "城", "州", "县", "郡", "国", "京", "都", "关", "山", "河", "江", "湖", "府",
    "宫", "殿", "营", "军", "镇", "村", "庄", "门", "桥", "谷", "岭", "原", "洲",
    "阳", "阴", "津", "渡", "塞", "堡", "台", "寺", "庙", "观", "楼", "阁",
)
FEMALE_TITLES = (
    "太后", "公主", "郡主", "夫人", "娘娘", "姑娘", "小姐", "娘子", "婆婆", "婶婶",
    "姐姐", "妹妹", "女", "娘", "姑", "婆", "婶", "姐", "妹", "氏", "妃", "后",
)
MALE_TITLES = (
    "先生", "公子", "将军", "大人", "陛下", "殿下", "郎君", "少爷", "老爷", "掌柜",
    "师父", "师傅", "师兄", "师弟", "兄", "爷", "王", "帝", "侯", "郎", "僧", "侠",
    "伯", "仆", "卫", "兵", "将", "官",
)
TITLES = tuple(sorted(set(FEMALE_TITLES + MALE_TITLES), key=len, reverse=True))
TITLE_ONLY = {"天王", "太后", "天子", "皇帝", "陛下", "殿下", "主公", "大王", "王爷",
              "长公主", "大长公主", "单于", "可汗", "大汗", "国主", "储君", "太子"}
NAME_PREFIX = {"老", "小", "大", "阿", "长"}
STATE_PREFIX = {"蜀", "吴", "魏", "赵", "燕", "齐", "楚", "韩", "秦", "晋", "梁",
                "宁", "湘", "济", "鲁", "宋", "陈", "越", "周", "汉", "唐", "明", "清"}
# 只剥离“明显不是姓”的前导噪声；单字姓（于/向/马/连/冷/何/胡/明等）不剥
LEAD_STRIP = tuple(sorted(set(list(SPEECH) + [
    "胡说", "实话", "明知", "任何", "明明", "连连", "哈哈", "严肃", "投奔",
    "请", "跟", "听", "看", "走", "来", "去", "人", "位", "名", "个", "众", "诸",
    "两", "一", "声", "手", "头", "情", "能", "经", "称", "翻", "劳", "剩", "投",
    "奔", "笑", "哈", "与", "及", "或", "为", "对", "把", "被", "将", "已", "还",
    "就", "都", "也", "又", "才", "只", "便", "即", "而", "且", "则", "如", "若",
    "似", "像", "比", "较", "很", "太", "更", "最", "颇", "甚", "极", "略", "稍",
    "微", "轻", "慢", "缓", "淡", "忽", "猛", "顿", "赶", "急", "忙", "频", "仍",
    "犹", "自", "亲", "互", "相", "彼", "此", "各", "每", "所", "有", "无", "非",
    "未", "不", "没", "勿", "谁", "哪", "怎", "这", "那", "其", "之", "但", "却",
    "因", "由", "然", "故", "实", "愿", "办", "抱", "缤", "随", "等",
    "路", "可", "可能", "干", "里", "外", "内",
]), key=len, reverse=True))

SURNAMES = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水"
    "窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕"
    "郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏"
    "成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆"
    "高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉"
    "龚程嵇邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车"
    "侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白"
    "怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍却璩"
    "桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终"
    "暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简"
    "饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公"
)
AGE_WORDS = {
    "少年": ("少年", "孩童", "孩子", "十几岁", "十六七", "十七八", "十五六", "少年郎"),
    "青年": ("青年", "年轻", "二十", "十八", "十九", "二十多", "不到三十", "后生"),
    "中年": ("中年", "四十", "五十", "四十多", "五十多", "壮年"),
    "老年": ("老年", "老者", "老人", "花甲", "年迈", "白发", "六十", "七十", "八十", "老翁"),
}
SELF_FEMALE = ("妾身", "臣妾", "奴家", "小女子", "老身", "奴家", "妾")
SELF_MALE = ("老夫", "本王", "朕", "微臣", "在下", "小人", "属下", "本官", "本将")
KIN_FEMALE = ("母亲", "娘亲", "女儿", "妻子", "夫人", "姐姐", "妹妹", "嫂子", "婆婆", "祖母")
KIN_MALE = ("父亲", "爹爹", "儿子", "丈夫", "哥哥", "弟弟", "叔叔", "祖父", "伯父")
NAME_CHAR_FEMALE = set("娟芳娜敏静丽燕霞倩婷雪梅兰菊娘姑妹姐媛妍婉秀珍琳莉萍红")
NAME_CHAR_MALE = set("强伟军刚勇峰磊鹏杰斌龙虎山海江河石铁雷霆坤浩宇豪")
# 语气/状态 overlay：只从引号附近的叙述提示词判断，不由年龄推导
OVERLAYS = {
    "耳语": ("低声", "轻声", "耳语", "压低声", "小声道", "窃窃", "附耳", "低低地"),
    "醉": ("醉", "酒气", "微醺", "举杯", "醉意", "醺"),
    "怒": ("怒", "愤", "厉声", "吼道", "拍案", "咬牙", "沉着脸", "厉喝"),
    "怯": ("怯", "颤声", "怯生生", "结巴", "嗫嚅", "不敢", "畏缩"),
    "虚弱": ("虚弱", "无力", "气若游丝", "喘息", "吃力", "沙哑", "奄奄"),
    "回忆": ("回忆", "想起", "记得", "当年", "往事", "原来", "恍惚"),
    "高喊": ("高喊", "大喊", "喊道", "叫道", "厉喝", "咆哮", "嘶吼", "扬声"),
}


def detect_overlay(near: str) -> str | None:
    """返回引号附近最明显的一个状态标签；没有则 None。"""
    best, best_n = None, 0
    for label, words in OVERLAYS.items():
        n = sum(near.count(w) for w in words)
        if n > best_n:
            best, best_n = label, n
    return best


ADVERBS = tuple(sorted({
    "拱手", "小声", "大声", "开口", "喃喃", "继续", "冷冷", "微笑", "起身", "老实",
    "直接", "点头", "高声", "正色", "随即", "低声", "急忙", "轻声", "摇头", "叹息",
    "插口", "上前", "转身", "提醒", "惊讶", "冷淡", "补充", "马上", "冷笑", "一听",
    "说来", "大笑", "抬头", "良久", "改口", "颤声", "一会", "感慨", "解释", "疑惑",
    "赔笑", "皱眉", "急切", "脱口", "大致", "回来", "平淡", "恳切", "激动", "一直",
    "挺身", "实话", "苦笑", "淡淡", "缓缓", "慢慢", "轻轻", "笑吟吟", "笑眯眯",
    "含糊", "迟疑", "犹豫", "沉默", "诧异",
    "笑", "冷", "轻", "沉", "低", "高", "大", "小", "正", "急", "颤", "淡", "缓",
    "慢", "微", "苦", "怒", "惊", "讶", "疑", "叹", "哼", "念", "劝",
}, key=len, reverse=True))


def _strip_adverbs(window: str) -> str:
    changed = True
    while changed:
        changed = False
        for adv in ADVERBS:
            if window.endswith(adv) and len(window) > len(adv):
                window = window[:-len(adv)]
                changed = True
                break
    return window


def _strip_leading(window: str) -> str:
    """剥掉“说道吴王”“请徐公子”这类前导噪声，留下真正的人名后缀。"""
    changed = True
    while changed and len(window) >= 2:
        changed = False
        for tok in LEAD_STRIP:
            if window.startswith(tok) and len(window) > len(tok):
                window = window[len(tok):]
                changed = True
                break
    return window


def _clean_line(line: str) -> str:
    return line.rstrip("\n").strip("\u3000 \t\r\ufeff")


def parse_book(path: str | Path, limit: int | None = None) -> dict:
    """流式读取正文，按章节标题切分；无标题时按长度分块并标记 synthetic。"""
    path = Path(path)
    lines: list[str] = []
    with path.open(encoding="utf-8-sig", errors="ignore") as fh:
        for line in fh:
            lines.append(_clean_line(line))
    chapters: list[dict] = []
    cur = None
    vols: list[str] = []
    for line in lines:
        m = CH_HEAD_RE.match(line)
        if m and len(line) <= 60 and not line.endswith("。"):
            if cur is not None:
                chapters.append(cur)
            cur = {"idx": len(chapters), "title": line.strip(), "paras": [], "vol": vols[-1] if vols else ""}
            continue
        vm = VOL_HEAD_RE.match(line)
        if vm and len(line) <= 40:
            vols.append(line.strip())
            continue
        if cur is None:
            continue
        cur["paras"].append(line)
    if cur is not None:
        chapters.append(cur)

    synthetic = False
    if len(chapters) < 3:
        synthetic = True
        chapters = _chunk_paragraphs(lines, 3200)

    if limit:
        chapters = chapters[:limit]
        for i, ch in enumerate(chapters):
            ch["idx"] = i

    style_counts: Counter = Counter()
    para_total = 0
    for ch in chapters:
        for text in ch["paras"]:
            if text:
                para_total += 1
                for style in ("cn", "jp", "jp2", "ascii"):
                    o, c = QUOTE_STYLES[style]
                    style_counts[style] += min(text.count(o), text.count(c))
    primary = max(style_counts, key=style_counts.get) if style_counts else "cn"
    if style_counts.get(primary, 0) == 0:
        primary = "cn"

    return {
        "path": str(path),
        "chapters": chapters,
        "chapter_mode": "synthetic_chunks" if synthetic else "headings",
        "primary_style": primary,
        "style_counts": dict(style_counts),
        "paragraphs": para_total,
    }


def _chunk_paragraphs(lines: list[str], target: int) -> list[dict]:
    chapters, cur, size = [], [], 0
    for line in lines:
        cur.append(line)
        size += len(line)
        if size >= target:
            chapters.append({"idx": len(chapters), "title": f"chunk {len(chapters)}",
                             "paras": cur, "vol": ""})
            cur, size = [], 0
    if cur:
        chapters.append({"idx": len(chapters), "title": f"chunk {len(chapters)}",
                         "paras": cur, "vol": ""})
    return chapters


# --------------------------------------------------------------------------- 引号
def quote_spans(text: str, style: str) -> list[tuple[int, int, str, str]]:
    """返回 [(start, end, inner_text, style)]；行首 —— 也作为对白。"""
    spans = []
    for m in STYLE_RE[style].finditer(text):
        spans.append((m.start(), m.end(), m.group(1), style))
    stripped = text.lstrip()
    if not spans and stripped.startswith(("——", "—", "–")):
        off = len(text) - len(stripped)
        k = 0
        while k < len(stripped) and stripped[k] in "—–-":
            k += 1
        body = stripped[k:].strip()
        if body:
            start = off + k
            spans.append((start, len(text), body, "dash"))
    return spans


def count_quotes(chapters, style: str) -> int:
    return sum(len(quote_spans(t, style)) for ch in chapters
               for t in ch["paras"] if t)


# --------------------------------------------------------------------------- 人名挖掘
def _valid_shape(cand: str) -> bool:
    if not (2 <= len(cand) <= 4):
        return False
    if any(not ("\u4e00" <= c <= "\u9fa5") for c in cand):
        return False
    if any(c in FUNC_CHARS for c in cand):
        return False
    if cand in STOP_NAMES:
        return False
    if cand[-1] in BAD_END:
        return False
    if cand.endswith(("地", "的", "得", "着", "了", "吗", "呢", "吧", "啊", "们")):
        return False
    if cand[0] in "第每某各本上下前后左右这那":
        return False
    if cand[0] == cand[1]:
        return False
    if any(cand.startswith(p) and len(set(cand[:2])) == 1 for p in ("微微", "轻轻", "慢慢",
                                                                    "默默", "悄悄", "缓缓",
                                                                    "淡淡", "冷冷", "渐渐")):
        return False
    if cand[-1] in PLACE_SUFFIX:
        return False
    return True


def _title_stem_ok(cand: str) -> bool:
    for t in TITLES:
        if cand.endswith(t) and len(cand) > len(t):
            stem = cand[:-len(t)]
            if stem[0] in SURNAMES or stem in STATE_PREFIX or stem in NAME_PREFIX:
                return True
            return len(stem) >= 2 and stem[0] not in FUNC_CHARS and \
                stem[0] not in "道说问答叫喊笑骂叹哼"
    return False


def _name_like(cand: str, count: int, quote_total: int) -> bool:
    if cand in TITLE_ONLY:
        return True
    if cand[0] in SURNAMES:
        return True
    if cand.endswith(TITLES) and _title_stem_ok(cand):
        return True
    return count >= max(30, int(quote_total * 0.001)) and len(cand) <= 3


# ---- 1.1 人名证据分级（普通词 vs 角色）
TAG_GAP = set("笑冷淡轻怒叹哼微苦忙急沉缓低高大小森然地着了")
TAG_VERB_RE = re.compile(
    "[" + re.escape("".join(sorted(TAG_GAP))) + "]{0,3}"
    "(?:道|说|问|答|喊|叫|笑|骂|叹|哼|嚷|吼|开口|回答)"
    "(?![话说得出过着法服明])")
# 宽松版：只看“词后紧跟言说动词”，用于判断引导语前是不是修饰语（的/个/位…）
TAG_VERB_LOOSE = re.compile(
    "[" + re.escape("".join(sorted(TAG_GAP))) + "]{0,3}"
    "(?:道|说|问|答|喊|叫|笑|骂|叹|哼|嚷|吼|开口|回答)"
    "(?![话说得出过着法服明])")
MOD_PRE = set("的个种名位群些这那每某几两半用以换更最很太挺真好一二三四五六七八九十百千万")
CLAUSE_PRE = set("，。！？；：、“”「」『』\n ")
# 言说动词后必须是停顿/引号/行尾，否则是“空间通道/味道/知道”这类复合词
VERB_END = set("，。！？；：、“”「」『』…—~～ \t\n\u3000")
TIME_PREFIX = ("时候", "时间", "现在", "然后", "之后", "当时", "此时",
               "那时", "平时", "顿时", "时候")
HONORIFIC_ADDR = ("先生", "公子", "姑娘", "小姐", "大人", "将军", "掌柜",
                  "兄", "爷", "师父", "师傅", "郎君")
GROUP_WORDS = ("他们", "她们", "两人", "二人", "众人", "大家", "所有人",
               "一群", "几个", "旁人")


def _has_identity(name: str) -> bool:
    return (name in TITLE_ONLY or name.endswith(TITLES)
            or name.endswith(HONORIFIC_ADDR))


def _speech_evidence(chapters, names, style: str) -> dict[str, dict]:
    """统计每个候选的“对话归属”证据，用于把人名和普通词分开。

    tag    : 紧贴言说动词（可隔 0-3 个修饰字）
    mod    : 上述 tag 里，前面是“的/个/种/位/这/那…”的（名词短语，不是主语）
    start  : tag 且位于句首/标点后（主语位置）
    vocative: 出现在引号内、后面直接跟逗号/感叹号等（被称呼）
    """
    info = {n: {"tag": 0, "mod": 0, "start": 0, "vocative": 0, "loose": 0}
            for n in names}
    order = sorted(names, key=lambda x: (-len(x), x))
    if not order:
        return info
    pat = re.compile("|".join(re.escape(n) for n in order))
    voc = re.compile(r"(?:啊|呀|哪)?[，,！!？?：:、]")
    for ch in chapters:
        for text in ch["paras"]:
            if not text:
                continue
            spans = quote_spans(text, style)
            for m in pat.finditer(text):
                name = m.group(0)
                rest = text[m.end():m.end() + 6]
                mo = TAG_VERB_RE.match(rest)
                if not mo:
                    continue
                info[name]["loose"] += 1
                prev = text[m.start() - 1] if m.start() > 0 else ""
                if prev in MOD_PRE:
                    info[name]["mod"] += 1
                nxt = rest[mo.end():mo.end() + 1]
                if nxt and nxt not in VERB_END:
                    continue
                info[name]["tag"] += 1
                if prev == "" or prev in CLAUSE_PRE:
                    info[name]["start"] += 1
                if any(s <= m.start() < e for s, e, _, _ in spans) \
                        and voc.match(text[m.end():m.end() + 3]):
                    info[name]["vocative"] += 1
    return info


def mine_names(chapters, primary_style: str) -> dict:
    """从言说动词前后缀统计候选人名，做通用过滤 + 别名合并。"""
    quote_total = count_quotes(chapters, primary_style)
    min_count = max(3, int(quote_total * 0.00015))
    raw: Counter = Counter()
    raw_chapters: dict[str, set] = defaultdict(set)
    deg: Counter = Counter()
    di: Counter = Counter()
    DEGREE = ("很", "十分", "非常", "有些", "有点", "略显", "极其", "特别")
    for ch in chapters:
        idx = ch["idx"]
        for text in ch["paras"]:
            if not text:
                continue
            for d in DEGREE:
                start = 0
                while True:
                    p = text.find(d, start)
                    if p < 0:
                        break
                    tail = text[p + len(d):p + len(d) + 4]
                    tail = re.sub(r"[^\u4e00-\u9fa5]", "", tail)
                    for L in (2, 3, 4):
                        if len(tail) >= L:
                            deg[tail[:L]] += 1
                    start = p + len(d)
            # X地：能带“地”的多是副词/形容词，不是人名
            for p, ch2 in enumerate(text):
                if ch2 != "地":
                    continue
                tail = re.sub(r"[^\u4e00-\u9fa5]", "", text[max(0, p - 4):p])
                for L in (2, 3, 4):
                    if len(tail) >= L:
                        di[tail[-L:]] += 1
            for m in VERB_RE.finditer(text):
                window = text[max(0, m.start() - 10):m.start()]
                window = re.sub(r"[^\u4e00-\u9fa5]", "", window)
                if len(window) < 2:
                    continue
                window = _strip_leading(_strip_adverbs(window))
                for L in (2, 3, 4):
                    if len(window) >= L:
                        cand = window[-L:]
                        if _valid_shape(cand):
                            raw[cand] += 1
                            raw_chapters[cand].add(idx)
            # 引号后置归属（「……」赵玖道）：中文书名里大量用这种结构
            close = QUOTE_STYLES[primary_style][1]
            for m in re.finditer(re.escape(close), text):
                tail = re.sub(r"[^\u4e00-\u9fa5]", "", text[m.end():m.end() + 8])
                tail = _strip_leading(tail)
                if len(tail) < 2 or tail[0] not in SURNAMES:
                    continue
                for L in (2, 3):
                    cand = tail[:L]
                    if len(tail) >= L and _valid_shape(cand) \
                            and cand[-1] not in BAD_END \
                            and cand[-1] not in "手头脸眼身心目口脚嘴眉背腰上下里外中前后边面" \
                            and cand[-1] not in "一二三四五六七八九十的了着过是在都也又就":
                        raw[cand] += 1
                        raw_chapters[cand].add(idx)

    # 长名支配短后缀（郭时风 vs 时风）
    cands = {c: n for c, n in raw.items()
             if n >= min_count and _name_like(c, n, quote_total)}

    # 形容词/状态词误判：常被“很/十分/非常”修饰、或能带“地”的候选不是人名
    for c in list(cands):
        if c[0] not in SURNAMES and deg.get(c, 0) >= 3 and deg[c] >= 0.08 * cands[c]:
            cands.pop(c, None)
        elif di.get(c, 0) >= 2 and not c.endswith(TITLES):
            cands.pop(c, None)
    # 叠字副词后缀（乐呵呵/笑眯眯/气冲冲）不是人名
    REDUP = ("呵呵", "嘻嘻", "哈哈", "嘿嘿", "悠悠", "腾腾", "冲冲", "洋洋", "乎乎",
             "溜溜", "哒哒", "滋滋", "咪咪", "咚咚", "嗡嗡")
    for c in list(cands):
        if len(c) >= 3 and c[-2:] in REDUP:
            cands.pop(c, None)
        elif len(c) == 2 and raw.get(c + c[-1], 0) >= 0.5 * cands[c]:
            # “乐呵”背后是“乐呵呵道”，保留 2 字只会留下副词骨架
            cands.pop(c, None)
    # 前缀污染：已知人名 + 形容词/状态词（吕树好奇）
    for c in list(cands):
        for s in list(cands):
            if s != c and len(s) >= 2 and len(c) > len(s) and c.startswith(s):
                rest = c[len(s):]
                if (rest in STOP_NAMES or deg.get(rest, 0) >= 3) and cands[c] < 0.5 * cands[s] \
                        and not c.endswith(TITLES):
                    cands.pop(c, None)
                    break
    # 短前缀被长名支配（薛金 vs 薛金摇）：同源切片产物，保留长名
    for c in list(cands):
        for s in list(cands):
            if s != c and len(s) > len(c) and s.startswith(c) \
                    and cands[s] >= 0.6 * cands[c] \
                    and raw_chapters[c] <= raw_chapters[s] \
                    and not c.endswith(TITLES):
                cands.pop(c, None)
                break
    # 反过来：已知人名 + 多切的一字（徐础一/马维脸），丢弃长尾
    for c in list(cands):
        for s in list(cands):
            if s != c and len(c) == len(s) + 1 and c.startswith(s) \
                    and cands[s] >= 3 * cands[c] and not c.endswith(TITLES):
                cands.pop(c, None)
                break

    # 去掉“已知人名 + 副词/动词”污染（徐础猜、徐础立刻）和“前导噪声 + 人名”（路徐础）
    tail_noise = {"猜", "长", "请"} | set(ADVERBS) | STOP_NAMES
    for c in list(cands):
        if cands[c] >= 30:
            continue
        for s in list(cands):
            if s == c or len(s) < 2:
                continue
            if c.startswith(s) and len(c) > len(s):
                r = c[len(s):]
                if r in tail_noise or r in BAD_END or any(r.startswith(a) for a in ADVERBS):
                    cands.pop(c, None)
                    break
            if c.endswith(s) and len(c) > len(s):
                p = c[:-len(s)]
                if p in LEAD_STRIP or p in STOP_NAMES:
                    cands.pop(c, None)
                    break
        if c in cands and cands[c] < 30 and any(
                c.startswith(tok) and len(c) > len(tok) for tok in LEAD_STRIP):
            cands.pop(c, None)
    drop = set()
    ordered = sorted(cands, key=lambda x: (-len(x), x))
    for short in list(cands):
        for long in ordered:
            if len(long) > len(short) and long.endswith(short) and \
                    cands[long] >= cands[short] * 0.5:
                drop.add(short)
                break
    cands = {c: n for c, n in cands.items() if c not in drop}

    # ---- 1.1 证据分级：普通词不占角色位，但只有“负证据”足够明确时才拒，
    # 避免把低频真名（高岛平、刘旋宇这类）误删。每个被拒词都记录理由。
    ev = _speech_evidence(chapters, list(cands), primary_style)
    rejected: list[dict] = []
    rejected_names: set[str] = set()
    for c in sorted(cands, key=lambda x: (-cands[x], x)):
        e = ev[c]
        ident = _has_identity(c)
        deg_n = deg.get(c, 0)
        reasons = []
        if not ident:
            if deg_n >= 3 and deg_n >= 0.05 * cands[c]:
                reasons.append(f"常被程度副词修饰x{deg_n}")
            if e["mod"] >= 2 and e["mod"] / max(e["loose"], 1) >= 0.6:
                reasons.append(f"引导语前均为修饰语x{e['mod']}/{e['loose']}")
            if cands[c] <= 10 and c.startswith(TIME_PREFIX):
                reasons.append("时间/连接词前缀")
        if reasons:
            rejected.append({
                "term": c, "count": cands[c],
                "chapters": len(raw_chapters[c]),
                "evidence": {"tag": e["tag"], "modifier": e["mod"],
                             "subject": e["start"], "vocative": e["vocative"],
                             "identity": ident, "degree": deg_n,
                             "adverbial": di.get(c, 0)},
                "reason": "；".join(reasons),
            })
            rejected_names.add(c)

    # 前后缀关系 + 称谓别名
    alias: dict[str, str] = {}
    for name in sorted(cands, key=lambda x: (len(x), x)):
        # 前缀扩展且带称谓：欢颜 -> 欢颜郡主
        for long in cands:
            if long != name and long.startswith(name) and long.endswith(TITLES) and \
                    cands[long] >= cands[name] * 0.5:
                alias[name] = long
                break
    honorific = ("先生", "公子", "姑娘", "小姐", "大人", "将军", "掌柜", "兄", "爷",
                 "师父", "师傅", "郎君")
    for name in list(cands):
        if cands[name] < 5:
            continue
        for t in honorific:
            if name.endswith(t) and len(name) > len(t):
                stem = name[:-len(t)]
                if not (1 <= len(stem) <= 2):
                    break
                pool = [x for x in cands if x != name and x.startswith(stem)
                        and not x.endswith(honorific) and cands[x] >= cands[name]]
                if len(pool) == 1:
                    alias[name] = pool[0]
                break

    # 同尾字改名（徐础 / 楼础）：同段不共现 + 章区间重叠 + 非称谓名
    para_sets = _paragraph_name_sets(chapters, cands)
    cooc: Counter = Counter()
    for names in para_sets:
        ns = sorted(names)
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                cooc[(ns[i], ns[j])] += 1
    merges = []
    by_last: dict[str, list[str]] = defaultdict(list)
    for n in cands:
        if len(n) >= 2 and not n.endswith(TITLES) and n[-1] not in PLACE_SUFFIX:
            by_last[n[-1]].append(n)
    for last, group in by_last.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if len(a) != len(b) or a[0] == b[0]:
                    continue
                cooc_n = cooc.get((a, b), 0) + cooc.get((b, a), 0)
                if cooc_n > max(10, int(0.01 * min(cands[a], cands[b]))):
                    continue
                # 低频名字不参与改名合并，避免把路人并进主角
                if min(cands[a], cands[b]) < 20:
                    continue
                ca, cb = raw_chapters[a], raw_chapters[b]
                inter = len(ca & cb)
                union = len(ca | cb) or 1
                overlap = inter / union
                # 2 字名允许“前后期改名”（区间近乎不重叠），3 字名要求区间重叠
                if len(a) == 2:
                    if overlap >= 0.2 or (inter == 0 and min(cands[a], cands[b]) >= 8):
                        pass
                    else:
                        continue
                elif overlap < 0.2:
                    continue
                if min(cands[a], cands[b]) < 8:
                    continue
                lo, hi = (a, b) if cands[a] >= cands[b] else (b, a)
                alias[hi] = alias.get(hi, lo)
                merges.append({"from": hi, "to": alias[hi], "shared": last,
                               "counts": [cands[a], cands[b]],
                               "chapter_overlap": round(inter / union, 2)})

    # 解析 canonical（处理链式别名）
    def canon(n: str) -> str:
        seen = set()
        while n in alias and n not in seen:
            seen.add(n)
            n = alias[n]
        return n

    gaz: dict[str, dict] = {}
    for c, n in cands.items():
        gaz[c] = {"count": n, "canonical": canon(c), "chapters": sorted(raw_chapters[c])}
    # 别名指向的 canonical 也进 gazetteer
    for c in list(alias):
        if c not in gaz:
            gaz[c] = {"count": cands.get(c, 0), "canonical": canon(c),
                      "chapters": sorted(raw_chapters.get(c, set()))}
    # 只保留 canonical 有足够台词的；别名保留
    final = {}
    canon_counts: Counter = Counter()
    for c, info in gaz.items():
        canon_counts[info["canonical"]] += info["count"]
    for c, info in gaz.items():
        if canon_counts[info["canonical"]] >= min_count:
            final[c] = info
    # 证据不足的词只标成候选：保留在词表里继续参与台词归因（否则短章里
    # 其他真角色会因为同章可归因台词变少而无法进入窗口、整角色消失），
    # 但不再生成角色、不占声线位。
    if rejected_names:
        for k, info in final.items():
            if info["canonical"] in rejected_names:
                info["candidate"] = True
    return {
        "gazetteer": final,
        "merges": merges,
        "candidate_terms": rejected,
        "quote_total": quote_total,
        "raw_candidates": len(raw),
        "kept": len(final),
    }


def _paragraph_name_sets(chapters, names) -> list[set]:
    pat = _names_alt(names)
    out = []
    for ch in chapters:
        for text in ch["paras"]:
            if text:
                found = {m.group(0) for m in pat.finditer(text)}
                if len(found) > 1:
                    out.append(found)
    return out


def _names_alt(gaz) -> re.Pattern:
    names = sorted(gaz, key=lambda x: (-len(x), x))
    return re.compile("|".join(re.escape(n) for n in names)) if names else re.compile(r"$^")


def _find_names(text: str, pat: re.Pattern) -> list[tuple[int, str]]:
    return [(m.start(), m.group(0)) for m in pat.finditer(text)]


# --------------------------------------------------------------------------- 归因
def _clause_window(text: str, pos: int, name_len: int) -> str:
    left = 0
    for c in "。！？；，、“”「」『』":
        p = text.rfind(c, 0, pos)
        if p > left:
            left = p + 1
    left = max(left, pos - 24)
    right_candidates = [p for p in (text.find(c, pos + name_len) for c in "。！？；，、")
                        if p != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left:min(right, pos + name_len + 24)]


def attribute_paragraph(text: str, pat: re.Pattern, gaz: dict,
                        style: str) -> list[dict]:
    spans = quote_spans(text, style)
    out = []
    cursor = 0
    for start, end, inner, qstyle in spans:
        prefix = text[cursor:start]
        suffix = text[end:end + 40]
        cursor = end
        speaker, conf = None, "unknown"
        found = _find_names(prefix, pat)
        # 1) 冒号前“人名 + 副词 + 言说动词”
        for pos, name in reversed(found):
            tail = prefix[pos + len(name):]
            if len(tail) <= 10 and re.search(r"(?:" + SPEECH_ALT + r")[：:]?$", tail):
                speaker, conf = name, "strong"
                break
        # 2) 人名紧贴引号
        if not speaker and found:
            pos, name = found[-1]
            tail = prefix[pos + len(name):]
            if len(tail) <= 5 and len(found) == 1:
                speaker, conf = name, "weak"
        # 2b) 人名在引号前 12 字内且无人称代词
        if not speaker and found:
            pos, name = found[-1]
            tail = prefix[pos + len(name):]
            if len(tail) <= 12 and len(found) == 1 and not re.search(r"[她他]", tail):
                speaker, conf = name, "weak"
        # 3) 引号后紧跟人名（+言说动词）
        if not speaker:
            head = suffix[:14]
            after = _find_names(head, pat)
            if after:
                pos, name = after[0]
                if pos <= 4:
                    rest = head[pos + len(name):pos + len(name) + 5]
                    conf = "strong" if re.search(r"(?:" + SPEECH_ALT + r")", rest) else "weak"
                    speaker = name
        # 4) 整段只有一个人名
        if not speaker:
            all_names = {n for _, n in _find_names(text, pat)}
            if len(all_names) == 1:
                speaker, conf = next(iter(all_names)), "weak"
        cid = gaz.get(speaker, {}).get("canonical") if speaker else None
        near = text[max(0, start - 24):min(len(text), end + 24)]
        out.append({"start": start, "end": end, "text": inner, "speaker": speaker,
                    "canonical": cid, "conf": conf, "style": qstyle,
                    "overlay": detect_overlay(near)})
    return out


PENDING_BUDGET = 1000
PENDING_RADII = (1, 3, 5, 7, 9, 11, 13, 15)


def _context_indices(items, pi, radius):
    """取同一场景块内、pi 两侧各 radius 段的段落下标；空行视为场景边界。"""
    left = []
    for j in range(pi - 1, max(-1, pi - radius - 1), -1):
        if items[j] is None:
            break
        left.append(j)
    right = []
    for j in range(pi + 1, min(len(items), pi + radius + 1)):
        if items[j] is None:
            break
        right.append(j)
    return [pi] + left + right


def _context_resolve(items, pi, pat, gaz, budget=PENDING_BUDGET):
    """逐级扩窗：±1 → ±3 → …，总上下文不超过 budget 字。

    只有在扩窗里出现唯一“有言说动作的人名”，或唯一候选名时才归因；
    否则返回 None，保持 pending。返回 (cid, radius, chars)。
    """
    last_span = 0
    for radius in PENDING_RADII:
        idxs = _context_indices(items, pi, radius)
        texts = [items[j][0] for j in sorted(idxs)]
        span = sum(len(t) for t in texts)
        if span > budget:
            return None, radius, last_span
        last_span = span
        names, cue_names = set(), set()
        for t in texts:
            for pos, name in _find_names(t, pat):
                cid = gaz.get(name, {}).get("canonical")
                if not cid:
                    continue
                names.add(cid)
                win = t[max(0, pos - 6):pos + len(name) + 10]
                if re.search(SPEECH_ALT, win):
                    cue_names.add(cid)
        if len(cue_names) == 1:
            return next(iter(cue_names)), radius, span
        if not cue_names and len(names) == 1:
            return next(iter(names)), radius, span
    return None, PENDING_RADII[-1], last_span


def _context_text(items, pi, budget=PENDING_BUDGET):
    """给二级通道的上下文：尽量取满但不超 budget 字。"""
    idxs = _context_indices(items, pi, PENDING_RADII[-1])
    idxs.sort()
    parts, total = [], 0
    for j in idxs:
        t = items[j][0]
        if total + len(t) + 1 > budget:
            room = budget - total - 1
            if room > 0:
                parts.append(t[:room])
            break
        parts.append(t)
        total += len(t) + 1
    return "\n".join(parts)


GENDER_F_RE = re.compile(r"她(?!们)")
GENDER_M_RE = re.compile(r"(?<!其)他(?!们|人|乡|日|处|物|杀|年|国|方)")
GROUP_RE = re.compile("|".join(GROUP_WORDS))
GENDER_TITLE_RE = re.compile(
    "姑娘|小姐|夫人|太太|婆婆|公子|少爷|先生|大人|将军|老汉|老者|少女|妇人")


def _window_gender(window: str) -> tuple[int, int]:
    """小句里只取明确的单数“她/他”。

    “他们/她们/两人/众人”是群像，不能把性别算到某一个人头上；
    “其他/他人/他乡”里的“他”不是代词，旧口径会误记成男。
    """
    if GROUP_RE.search(window):
        return 0, 0
    return len(GENDER_F_RE.findall(window)), len(GENDER_M_RE.findall(window))


def _gender_category(window: str) -> str:
    if GROUP_RE.search(window):
        return "group"
    if GENDER_TITLE_RE.search(window):
        return "title"
    return "clause"


def attribute_book(book: dict, gaz: dict, pending_resolver=None) -> dict:
    chapters = book["chapters"]
    style = book["primary_style"]
    pat = _names_alt(gaz)
    chars: dict[str, dict] = {}
    conf_counter: Counter = Counter()
    style_counter: Counter = Counter()
    total_quotes = 0
    dash_quotes = 0
    mixed_chapters = 0
    unquoted_suspected = 0
    unquoted_samples: list[dict] = []
    pending_unquoted: list[dict] = []
    pending_items: list[dict] = []
    pending_stats = Counter(resolved_pm1=0, resolved_pm3=0, resolved_far=0,
                            still_pending=0, unquoted=0, llm_resolved=0,
                            llm_failed=0, llm_skipped=0)
    pending_levels: Counter = Counter()
    long_quotes = 0
    unattributed: list[dict] = []
    char_quotes: dict[str, list[str]] = defaultdict(list)

    def touch(cid):
        if cid not in chars:
            chars[cid] = {"turns": 0, "chars": 0, "chapters": [], "mentions": {},
                          "gender_votes": Counter(), "age_votes": Counter(),
                          "gender_by_chapter": defaultdict(Counter),
                          "gender_ctx": Counter(),
                          "kin_f": 0, "kin_m": 0,
                          "states": Counter(), "state_chapters": defaultdict(Counter),
                          "address_titles": Counter(), "self_refs": Counter()}
        return chars[cid]

    chapter_rows = []
    for ch in chapters:
        idx = ch["idx"]
        blocks, cur_block = [], []
        mentions = set()
        last = None
        items = []
        used_other_style = False
        for pi, text in enumerate(ch["paras"]):
            if not text:
                items.append(None)
                continue
            spans = quote_spans(text, style)
            if not spans:
                # 本章主样式抽不到时尝试其他样式，并标记混合
                for alt in ("jp", "cn", "jp2", "ascii"):
                    if alt == style:
                        continue
                    if quote_spans(text, alt):
                        spans = quote_spans(text, alt)
                        used_other_style = True
                        break
            if not spans and re.search(r"(?:" + SPEECH_ALT + r")[：:]", text):
                unquoted_suspected += 1
                pending_stats["unquoted"] += 1
                item = {"id": f"ch{idx}p{pi}u", "chapter": idx, "para": pi,
                        "kind": "unquoted", "text": text[:300],
                        "reason": "有言说动词但没有引号样式命中"}
                if len(pending_unquoted) < 500:
                    pending_unquoted.append(item)
                if len(unquoted_samples) < 30:
                    unquoted_samples.append({"ch": idx, "text": text[:80]})
            quotes = attribute_paragraph(text, pat, gaz, style)
            found = _find_names(text, pat)
            items.append([text, found, quotes])
            for pos, name in found:
                cid = gaz[name]["canonical"]
                mentions.add(cid)
                st = touch(cid)
                st["mentions"][name] = st["mentions"].get(name, 0) + 1
                window = _clause_window(text, pos, len(name))
                if len(_find_names(window, pat)) == 1:
                    f_n, m_n = _window_gender(window)
                    raw_m = window.count("他")
                    if raw_m > m_n:
                        st["gender_ctx"][("skip", "other_word")] += raw_m - m_n
                    if GROUP_RE.search(window):
                        st["gender_ctx"][("skip", "group")] += 1
                    if f_n or m_n:
                        cat = _gender_category(window)
                        if f_n:
                            st["gender_votes"]["女"] += f_n
                            st["gender_by_chapter"][idx]["女"] += f_n
                            st["gender_ctx"][("女", cat)] += f_n
                        if m_n:
                            st["gender_votes"]["男"] += m_n
                            st["gender_by_chapter"][idx]["男"] += m_n
                            st["gender_ctx"][("男", cat)] += m_n
                    # 只有紧贴人名之前的亲属称谓才算（“妹妹吕小鱼”），
                    # “陈祖安的妹妹”归妹妹，不归陈祖安
                    pre = text[max(0, pos - 3):pos]
                    st["kin_f"] += sum(1 for w in
                                       ("妹妹", "姐姐", "女儿", "妻子", "嫂子", "娘亲",
                                        "姑姑", "祖母", "婶婶", "母亲") if pre.endswith(w))
                    st["kin_m"] += sum(1 for w in
                                       ("弟弟", "哥哥", "儿子", "丈夫", "父亲", "爹爹",
                                        "叔叔", "伯父", "祖父") if pre.endswith(w))
                for bucket, words in AGE_WORDS.items():
                    if any(w in window for w in words):
                        st["age_votes"][bucket] += 1
        if used_other_style:
            mixed_chapters += 1

        counts = Counter()
        for it in items:
            if not it:
                continue
            for q in it[2]:
                if q["canonical"] and q["conf"] in ("strong", "weak"):
                    counts[q["canonical"]] += 1
        alt_pair = [c for c, _ in counts.most_common(2)]
        use_alt = len(alt_pair) == 2 and counts[alt_pair[1]] >= 4

        # 第一遍：段落内规则 → 上下文扩窗（±1 → ±3 → …，1000 字封顶）
        for pi2, it in enumerate(items):
            if it is None:
                last = None
                continue
            text, found, quotes = it
            if not quotes:
                continue
            para = next((q["canonical"] for q in quotes if q["canonical"]), None)
            for q in quotes:
                if q["canonical"]:
                    continue
                if para:
                    q["canonical"], q["conf"] = para, "para"
                elif last and not found:
                    if use_alt and last in alt_pair:
                        other = alt_pair[0] if last == alt_pair[1] else alt_pair[1]
                        q["canonical"], q["conf"] = other, "alt"
                    else:
                        q["canonical"], q["conf"] = last, "cont"
            ctx_hit = None
            for q in quotes:
                if q["canonical"]:
                    continue
                if ctx_hit is None:
                    ctx_hit = _context_resolve(items, pi2, pat, gaz) or (None, 0, 0)
                cid, radius, span = ctx_hit
                if cid:
                    q["canonical"], q["conf"] = cid, "ctx"
                    q["ctx_level"], q["ctx_chars"] = radius, span
                    if radius <= 1:
                        pending_stats["resolved_pm1"] += 1
                    elif radius <= 3:
                        pending_stats["resolved_pm3"] += 1
                    else:
                        pending_stats["resolved_far"] += 1
                    pending_levels[radius] += 1
            resolved = [q["canonical"] for q in quotes if q["canonical"]]
            if resolved:
                last = resolved[-1]

        # 二级通道调用点：仍在 pending 的引号整章一次交给 resolver。
        # resolver=None（默认）时不调用，pending 原样保留。
        unresolved = []
        for pi2, it in enumerate(items):
            if not it:
                continue
            for qi, q in enumerate(it[2]):
                if q["canonical"]:
                    continue
                ctx = _context_text(items, pi2)
                unresolved.append({"id": f"ch{idx}p{pi2}q{qi}", "pi": pi2, "q": q,
                                   "context": ctx})
        if unresolved and pending_resolver is not None:
            cands = sorted(chars, key=lambda c: -chars[c]["turns"])[:30]
            paras = [it[0] if it else None for it in items]
            by_para: dict[int, list] = defaultdict(list)
            for u in unresolved:
                by_para[u["pi"]].append(u["q"])
            for pidx, qs in by_para.items():
                resp = None
                try:
                    resp = pending_resolver(paras, pidx, cands,
                                            lang=book.get("lang", "zh"),
                                            max_context_chars=PENDING_BUDGET,
                                            min_confidence=0.7)
                except Exception:
                    pending_stats["llm_failed"] += len(qs)
                    resp = None
                speaker = resp.get("speaker") if isinstance(resp, dict) else None
                conf = float(resp.get("confidence", 0) or 0) if isinstance(resp, dict) else 0
                if speaker and speaker in chars and conf >= 0.7:
                    for q in qs:
                        q["canonical"], q["conf"] = speaker, "llm"
                        q["llm_confidence"] = conf
                        pending_stats["llm_resolved"] += 1
                else:
                    pending_stats["llm_skipped"] += len(qs)

        # 记账：仍然 pending 的引号，保留状态并输出给人工/二级通道
        ctx_by_id = {u["id"]: u["context"] for u in unresolved}
        for pi2, it in enumerate(items):
            if not it:
                continue
            for qi, q in enumerate(it[2]):
                if q["canonical"]:
                    continue
                pending_stats["still_pending"] += 1
                if len(pending_items) < 1000:
                    qid = f"ch{idx}p{pi2}q{qi}"
                    ctx = ctx_by_id.get(qid) or _context_text(items, pi2)
                    pending_items.append({
                        "id": qid, "chapter": idx, "para": pi2,
                        "kind": "quote", "quote": q["text"][:300],
                        "context": ctx, "context_chars": len(ctx),
                        "reason": "扩窗至 1000 字仍无唯一候选",
                    })

        # 第二遍：只负责生成 turns，不再改归属
        for pi2, it in enumerate(items):
            if it is None:
                if cur_block:
                    blocks.append(cur_block)
                    cur_block = []
                last = None
                continue
            text, found, quotes = it
            if not quotes:
                continue
            para_named = any(q["conf"] in ("strong", "weak", "para") for q in quotes)
            for q in quotes:
                total_quotes += 1
                conf_counter[q["conf"]] += 1
                style_counter[q["style"]] += 1
                if q["style"] == "dash":
                    dash_quotes += 1
                if len(q["text"]) > 800:
                    long_quotes += 1
                cid = q["canonical"]
                if not cid:
                    if len(unattributed) < 40:
                        unattributed.append({"ch": idx, "text": text[:60],
                                             "quote": q["text"][:40]})
                    continue
                st = touch(cid)
                st["turns"] += 1
                st["chars"] += len(q["text"])
                st["chapters"].append(idx)
                char_quotes[cid].append(q["text"])
                if q.get("overlay"):
                    st["states"][q["overlay"]] += 1
                    st["state_chapters"][idx][q["overlay"]] += 1
                addr = 1 if (_find_names(q["text"], pat) or "你" in q["text"]
                             or "您" in q["text"]) else 0
                cur_block.append((cid, len(q["text"]), addr, 1 if para_named else 0,
                                  idx, q["start"], q["text"], pi2))

        if cur_block:
            blocks.append(cur_block)

        for b in blocks:
            for t in b:
                touch(t[0])["mentions"]["_blocks"] = \
                    touch(t[0])["mentions"].get("_blocks", 0) + 1
        chapter_rows.append({"idx": idx, "title": ch["title"],
                             "mentions": sorted(mentions), "blocks": [b for b in blocks if b]})

    char_rows = {}
    for cid, st in chars.items():
        chs = sorted(set(st["chapters"]))
        char_rows[cid] = {
            "id": cid, "canonical": cid, "turns": st["turns"], "chars": st["chars"],
            "chapters": chs, "chapter_count": len(chs),
            "first": chs[0] if chs else None, "last": chs[-1] if chs else None,
            "span": (chs[-1] - chs[0] + 1) if chs else 0,
            "gender_votes": dict(st["gender_votes"]),
            "gender_by_chapter": {str(k): dict(v)
                                  for k, v in sorted(st["gender_by_chapter"].items())},
            "gender_ctx": {f"{g}:{cat}": n
                           for (g, cat), n in sorted(st["gender_ctx"].items())},
            "kin_f": st.get("kin_f", 0), "kin_m": st.get("kin_m", 0),
            "age_votes": dict(st["age_votes"]),
            "states": dict(st["states"]),
            "state_chapters": {str(k): dict(v) for k, v in st["state_chapters"].items()},
            "mentions": {k: v for k, v in st["mentions"].items() if k != "_blocks"},
        }

    return {
        "chapters": chapter_rows,
        "characters": char_rows,
        "char_quotes": char_quotes,
        "coverage": {
            "chapters": len(chapter_rows),
            "chapter_mode": book["chapter_mode"],
            "primary_style": style,
            "style_counts": book["style_counts"],
            "mixed_style_chapters": mixed_chapters,
            "quotes": total_quotes,
            "dash_quotes": dash_quotes,
            "long_quotes": long_quotes,
            "unquoted_suspected": unquoted_suspected,
            "unquoted_samples": unquoted_samples,
            "pending": {
                **{k: pending_stats[k] for k in
                   ("resolved_pm1", "resolved_pm3", "resolved_far", "still_pending",
                    "unquoted", "llm_resolved", "llm_failed", "llm_skipped")},
                "levels": dict(sorted(pending_levels.items())),
                "budget_chars": PENDING_BUDGET,
            },
            "attribution": dict(conf_counter),
            "strong_pct": round(100 * conf_counter["strong"] / max(total_quotes, 1), 1),
            "any_pct": round(100 * (total_quotes - conf_counter["unknown"])
                             / max(total_quotes, 1), 1),
            "characters": len(char_rows),
            "name_merges": len(gaz),
        },
        "unattributed_samples": unattributed,
        "pending_items": pending_items,
        "pending_unquoted": pending_unquoted,
    }


# --------------------------------------------------------------------------- 性别 / 年龄
def resolve_identities(data: dict, gaz: dict) -> dict:
    """分层证据判定性别/年龄；证据不足标 unknown，不静默默认。"""
    chars = data["characters"]
    char_quotes = data["char_quotes"]
    pat = _names_alt(gaz)
    # 称谓证据：对话里“X姑娘 / X公子”
    addr_ev: dict[str, Counter] = defaultdict(Counter)
    for cid, texts in char_quotes.items():
        for text in texts:
            for m in pat.finditer(text):
                tail = text[m.end():m.end() + 4]
                for t in TITLES:
                    if tail.startswith(t):
                        gender = "女" if t in FEMALE_TITLES else "男"
                        addr_ev[gaz[m.group(0)]["canonical"]][gender] += 1
                        break
    out = {}
    unknown = []
    audit = []
    cand_ids = {cid for cid, info in gaz.items() if info.get("candidate")}
    for cid, st in chars.items():
        is_role = cid not in cand_ids
        fs = ms = 0
        why = []
        name = cid
        # 1) 称谓后缀 / 纯称谓：最强证据，独立成条，不被代词淹没
        title_gender = None
        if name in TITLE_ONLY:
            title_gender = "女" if name in ("太后", "长公主", "大长公主") else "男"
            why.append(f"固定称谓:{name}")
        else:
            for t in FEMALE_TITLES:
                if name.endswith(t) and len(name) > len(t):
                    title_gender = "女"
                    why.append(f"称呼后缀:{t}")
                    break
            if not title_gender:
                for t in MALE_TITLES:
                    if name.endswith(t) and len(name) > len(t):
                        title_gender = "男"
                        why.append(f"称呼后缀:{t}")
                        break
        if title_gender == "女":
            fs += 8
        elif title_gender == "男":
            ms += 8
        # 2) 被称谓语：他人对话里“X姑娘 / X公子”，需≥2 次才算数
        ae = addr_ev.get(cid, Counter())
        if ae:
            nf, nm = ae.get("女", 0), ae.get("男", 0)
            if nf >= 2:
                fs += min(3 * nf, 6)
                why.append(f"被称姑娘等x{nf}")
            if nm >= 2:
                ms += min(3 * nm, 6)
                why.append(f"被称公子等x{nm}")
        # 3) 代词：旁白小句窗口投票，各自封顶，避免长角色被刷成多数派
        gv = st["gender_votes"]
        if gv.get("女"):
            fs += min(gv["女"], 6)
            why.append(f"她x{gv['女']}")
        if gv.get("男"):
            ms += min(gv["男"], 6)
            why.append(f"他x{gv['男']}")
        # 亲属称谓：旁白写“妹妹/女儿”比通用“他/她”可靠
        kf = min(6, 2 * st.get("kin_f", 0))
        km = min(6, 2 * st.get("kin_m", 0))
        if kf:
            fs += kf
            why.append(f"旁白亲属词女x{kf // 2}")
        if km:
            ms += km
            why.append(f"旁白亲属词男x{km // 2}")
        # 4) 自称：按词条计，不同词条不叠加超过 cap
        texts = char_quotes.get(cid, [])
        self_f = min(6, 2 * sum(1 for w in SELF_FEMALE if any(w in t for t in texts)))
        self_m = min(6, 2 * sum(1 for w in SELF_MALE if any(w in t for t in texts)))
        if self_f:
            fs += self_f
            why.append(f"自称女性词x{self_f // 2}")
        if self_m:
            ms += self_m
            why.append(f"自称男性词x{self_m // 2}")
        # 5) 亲属关系 / 名用字：弱证据
        for w in KIN_FEMALE:
            if sum(1 for t in texts if w in t) >= 2:
                fs += 2
                why.append(f"亲属词{w}")
                break
        for w in KIN_MALE:
            if sum(1 for t in texts if w in t) >= 2:
                ms += 2
                why.append(f"亲属词{w}")
                break
        if name[-1] in NAME_CHAR_FEMALE:
            fs += 1
            why.append("名用字偏女")
        if name[-1] in NAME_CHAR_MALE:
            ms += 1
            why.append("名用字偏男")

        if title_gender:
            opp = fs if title_gender == "男" else ms
            same = ms if title_gender == "男" else fs
            if opp >= 8 and opp > same * 1.5:
                gender, conf = "未知", "conflict"
                why.append("称谓与正文证据冲突")
            else:
                gender, conf = title_gender, "high"
        elif kf >= 4 and kf > km * 2:
            gender, conf = "女", "kin"
        elif km >= 4 and km > kf * 2:
            gender, conf = "男", "kin"
        elif fs >= 5 and fs >= ms * 1.2:
            gender, conf = "女", "high" if fs >= 9 else "medium"
        elif ms >= 6 and ms >= fs * 1.2:
            gender, conf = "男", "high" if ms >= 10 else "medium"
        else:
            gender, conf = "未知", "unknown"
        if gender == "未知" and is_role:
            unknown.append({"cid": cid, "turns": st["turns"],
                            "chapters": st["chapter_count"], "conf": conf,
                            "evidence": {"女": fs, "男": ms, "notes": why[:6]}})

        # 1.1 性别审计：全书锁定一次，逐章只做“证据层”核对，不参与判定
        series = []
        for ch_s, v in sorted((st.get("gender_by_chapter") or {}).items()):
            f_n, m_n = v.get("女", 0), v.get("男", 0)
            if f_n >= 2 and f_n > m_n:
                series.append((int(ch_s), "女"))
            elif m_n >= 2 and m_n > f_n:
                series.append((int(ch_s), "男"))
        flips = sum(1 for a, b in zip(series, series[1:]) if a[1] != b[1])
        opposite = {"女": "男", "男": "女"}.get(gender)
        opp_votes = total_votes = 0
        for v in (st.get("gender_by_chapter") or {}).values():
            total_votes += v.get("女", 0) + v.get("男", 0)
            if opposite:
                opp_votes += v.get(opposite, 0)
        opp_chapters = [ch_s for ch_s, g in series if opposite and g == opposite]
        causes = {}
        for k, n in (st.get("gender_ctx") or {}).items():
            g, cat = k.split(":", 1)
            if opposite and g == opposite:
                causes[cat] = causes.get(cat, 0) + n
        if conf == "conflict":
            causes["title_conflict"] = causes.get("title_conflict", 0) + 1
        if is_role:
            audit.append({
                "cid": cid, "turns": st["turns"], "chapters": st["chapter_count"],
                "gender": gender, "confidence": conf, "locked": gender in ("男", "女"),
                "evidence_types": why[:8],
                "votes": {"女": int(st["gender_votes"].get("女", 0)),
                          "男": int(st["gender_votes"].get("男", 0))},
                "opposite_votes": opp_votes, "total_votes": total_votes,
                "consistency": round(1 - opp_votes / total_votes, 3) if total_votes else 1.0,
                "opposite_chapters": opp_chapters,
                "opposite_chapter_count": len(opp_chapters),
                "drift_episodes": flips,
                "root_causes": causes,
                "revision": None,
            })

        av = Counter(st.get("age_votes") or {})
        av_total = sum(av.values())
        age_order = ("少年", "青年", "中年", "老年")
        age, age_src = None, None
        if av_total >= 5:
            top = av.most_common()
            if top[0][1] / av_total >= 0.6:
                age, age_src = top[0][0], "evidence"
            elif len(top) > 1 and abs(age_order.index(top[1][0]) - age_order.index(top[0][0])) == 1 \
                    and (top[0][1] + top[1][1]) / av_total >= 0.7:
                # 相邻两档混淆（少年/青年）时取后一档：长篇角色会随书成长
                age, age_src = max((top[0][0], top[1][0]), key=age_order.index), "evidence_adjacent"
        if age is None:
            if any(w in name for w in ("老", "翁", "婆")) or name in ("太后", "老仆"):
                age, age_src = "老年", "name"
            elif any(w in "".join(texts[:20]) for w in ("老夫", "老身", "老朽")):
                age, age_src = "老年", "self_ref"
            elif any(w in "".join(texts[:20]) for w in ("小女子", "在下", "晚辈")):
                age, age_src = "青年", "self_ref"
            elif name.endswith(("公子", "少爷", "姑娘", "小姐", "郡主")) or \
                    name.startswith(("小", "阿")):
                age, age_src = "青年", "title_word"
            else:
                age, age_src = ("青年" if gender == "女" else "中年"), "default"
        out[cid] = {**st, "gender": gender, "gender_conf": conf,
                    "gender_score": {"女": fs, "男": ms},
                    "gender_notes": why[:10],
                    "age": age, "age_source": age_src,
                    "role": is_role}

    audit.sort(key=lambda r: (-r["turns"], r["cid"]))
    root = Counter()
    skipped = Counter()
    for cid, st in chars.items():
        for k, n in (st.get("gender_ctx") or {}).items():
            g, cat = k.split(":", 1)
            if g == "skip":
                skipped[cat] += n
    for r in audit:
        if r["locked"]:
            for cat, n in r["root_causes"].items():
                root[cat] += n
    audit_summary = {
        "roles": len(audit),
        "locked": sum(1 for r in audit if r["locked"]),
        "unknown": sum(1 for r in audit if not r["locked"]),
        "drift_episodes": sum(r["drift_episodes"] for r in audit if r["locked"]),
        "voice_drifts": 0,
        "roles_with_opposite": sum(1 for r in audit if r["locked"]
                                   and r["opposite_votes"] > 0),
        "opposite_votes": sum(r["opposite_votes"] for r in audit if r["locked"]),
        "total_votes": sum(r["total_votes"] for r in audit if r["locked"]),
        "root_causes": dict(sorted(root.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    return {"characters": out, "unknown_gender": unknown,
            "gender_audit": {"summary": audit_summary, "roles": audit}}


def apply_gender_overrides(chars: dict, overrides: dict | None) -> list[dict]:
    """显式改判路径：只改性别，并记录旧值、理由和受影响章节。"""
    revisions = []
    for cid in sorted(overrides or {}):
        spec = overrides[cid]
        if not isinstance(spec, dict) or cid not in chars:
            continue
        new = spec.get("gender")
        if new not in ("男", "女", "未知"):
            continue
        old = chars[cid]["gender"]
        chapters = sorted(set(chars[cid].get("chapters") or []))
        chars[cid]["gender"] = new
        chars[cid]["gender_conf"] = "override"
        chars[cid]["gender_notes"] = list(chars[cid].get("gender_notes") or [])[:9] \
            + [f"人工改判:{spec.get('reason', '')}"]
        revisions.append({
            "cid": cid, "from": old, "to": new,
            "reason": spec.get("reason", ""),
            "source": spec.get("source", "manual"),
            "chapters": chapters, "chapter_count": len(chapters),
        })
    return revisions


DISGUISE_RULES = (
    ("女扮男装", "男", "disguise"),
    ("男扮女装", "女", "disguise"),
    ("女扮男", "男", "disguise"),
    ("男扮女", "女", "disguise"),
    ("扮作", None, "disguise"),
    ("扮成", None, "disguise"),
    ("假扮", None, "disguise"),
    ("伪装成", None, "disguise"),
    ("易容", None, "disguise"),
    ("化名", None, "alias"),
    ("附身", None, "possession"),
    ("夺舍", None, "possession"),
    ("魂穿", None, "possession"),
)


def _presentation_gender(tail: str) -> str:
    if re.search(r"男|公子|少爷|王爷|男子|少年|汉子", tail):
        return "男"
    if re.search(r"女|姑娘|小姐|女子|少女|妇人|夫人", tail):
        return "女"
    return "未知"


def presentation_events(chapters, gaz: dict, style: str) -> list[dict]:
    """剧情内伪装/化名/附身：只认原文明确写出的证据，不猜。

    与“判定错误”分开：底层角色不变，这里只记录呈现身份和证据章节。
    """
    pat = _names_alt(gaz)
    neg = re.compile(r"不|没|未|非|别|无|并非|不是|不像|没有|以为|误以为|难道|是否|"
                     r"假如|如果|如同|好像|似乎|传闻")
    events = []
    seen = set()
    for ch in chapters:
        for text in ch["paras"]:
            if not text:
                continue
            found = _find_names(text, pat)
            if not found:
                continue
            for kw, gender, kind in DISGUISE_RULES:
                start = text.find(kw)
                while start >= 0:
                    end = start + len(kw)
                    if neg.search(text[max(0, start - 12):start]):
                        start = text.find(kw, end)
                        continue
                    if kind == "disguise" and text[end:end + 1] in ("的", "之"):
                        start = text.find(kw, end)
                        continue
                    # 附身/夺舍的受害者（被X附身 / 附身于X）不是呈现身份的人
                    if kind == "possession" and (
                            re.search(r"被|遭|受", text[max(0, start - 3):start])
                            or text[end:end + 1] == "于"):
                        start = text.find(kw, end)
                        continue
                    after = [p for p, n in found if p >= end]
                    before = [p for p, n in found if p < start]
                    # 伪装/附身的施动者必须在动词之前；动词后的人名是目标（假扮X）或受害者
                    pos = before[-1] if before else None
                    limit = 8 if kind == "possession" else 16
                    if kind == "possession" and text[end:end + 1] == "、":
                        pos = None
                    if pos is not None and abs(start - pos) <= limit:
                        name = next(n for p, n in found if p == pos)
                        # “贺荣兵卒假扮”= 贺荣的部下，不是贺荣本人
                        if text[pos + len(name):pos + len(name) + 1] in "兵军卒众家府队":
                            start = text.find(kw, end)
                            continue
                        cid = gaz[name]["canonical"]
                        # 施动者和动词之间若有句读，说明是另一分句（“杀了霍华德，然后扮成…”）
                        if kind == "disguise" and re.search(
                                r"[，。；！？]", text[pos + len(name):start]):
                            start = text.find(kw, end)
                            continue
                        shown = gender or _presentation_gender(text[end:end + 10])
                        target = None
                        if kind == "disguise" and after and "扮" in kw:
                            tp = after[0]
                            if tp - end <= 8:
                                target = gaz[next(n for p, n in found if p == tp)]["canonical"]
                        key = (cid, ch["idx"], kind, shown)
                        if key not in seen:
                            seen.add(key)
                            events.append({
                                "cid": cid, "chapter": ch["idx"],
                                "base_gender": None, "presentation": shown,
                                "kind": kind, "target": target,
                                "evidence": text[max(0, start - 12):end + 14],
                            })
                    start = text.find(kw, end)
    events.sort(key=lambda e: (e["chapter"], e["cid"], e["kind"], e["presentation"]))
    return events
