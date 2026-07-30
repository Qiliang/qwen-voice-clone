"""中文音色名编解码。

当前方案（codec）：
- 字表 1295 = 数字 10 + `_` `-` + 汉字 1283（常用字优先，兼顾人名用字）
- 正文全小写 base36（0-9a-z），API 强制小写也不丢信息
- qwen-audio prefix≤10 → 保证 5 字；qwen preferred_name≤16 → 保证 8 字

旧方案（codec_legacy）：base62 + GB2312 一级，仅用于解码历史音色。
"""

from __future__ import annotations


# 1283 汉字：人名常用字优先，其余按《现代汉语常用字表》等补齐
_HANZI_1283 = (
'平实质朴音电台感温柔亲和直爽利落惊奇讶异活泼灵动紧张不安悚恐怖朗亢奋激昂内敛含蓄满犀自然诵稳陈述标准播呆萌软糯专业解说客观冷静鼓舞励振坚韧科普讲带货轻松闲聊愤懑沉娇嗲甜腻细婉成熟委屈哀怨急躁焦灼着撒蜜有力情澎湃浑厚戏腔耐心傲俏皮智能助手二次元漫风格叙事浸天搞怪逗趣屑倨信笃定欢乐明伤低回切服新闻联期待向往忧郁好循善诱铿锵深可爱刻薄认谆劝导惋惜叹愉悦麻沮丧雅端庄气从容烦中充沛喜意外憨青春朝真童大润磁性角色模仿邻家御姐干练老慵懒调少年机知绿茶压抑快侠骨暖男武威严领阴强势霸狠毒辣雍宫廷怯弱卑人方清神秘术士贴正凛诙谐卡通楚俊纯稚嫩鬼马指挥美闺秀女重硬阳刚光开将热忱推荐笑慈祥祖母游运喘息闷凌厉剑血慧随在文书卷教腹黑亮宠拟声太变司仪睿尖穿透邃复古报子潮流理告健顾问启蒙师诚邀请恳响应包克制礼貌樱豪欣楠萱婷涵瑶梓诗思梦雨若曦诺浩轩墨雪萝莉叔南枝北辰东篱西窗夏秋冬花月星海山川小哥宝贝姑娘伙一乙十丁厂七卜入八九几儿了乃刀又三于亏工土才寸下丈与万上口巾千乞亿个勺久凡及夕丸么广亡门义之尸弓己已卫也飞刃习叉乡丰王井夫无云扎艺木五支厅犬区历尤友匹车巨牙屯比互瓦止日冈水见午牛毛升长仁什片仆化仇币仍仅斤爪反介父今凶分乏公仓氏勿欠丹匀乌凤勾六火为斗忆订计户尺引丑巴孔队办以允予双幻玉刊示末未击打巧扑扒功扔去甘世节本丙左右石布龙灭轧占旧帅归且旦目叶甲申叮号田由史只央兄叼叫另叨四生失禾丘付仗代仙们白仔他斥瓜乎丛令用甩印句匆册犯处鸟务饥主市立闪兰半汁汇头汉宁穴它讨写让训必议讯记永尼民出辽奶奴加召边发孕圣对矛纠幼丝式刑扛寺吉扣考托执巩圾扩扫地扬场耳共芒亚芝朽权过臣再协厌百存而页匠夸夺灰达列死夹轨邪划迈毕至此贞尘劣当早吐吓虫曲团同吊吃因吸吗屿帆岁岂则肉网朱先丢舌竹迁乔伟传乒乓休伍伏优伐延件任价份华仰伪似后行舟全会杀合兆企众爷伞创肌朵杂危旬旨负各名多争壮冲冰庆亦刘齐交衣产决妄闭闯羊并关米灯州汗污江池汤忙兴宇守宅字军许论农讽设访寻那迅尽孙阵收阶防奸如妇她妈羽买红纤级约纪驰巡寿弄麦形进戒吞远违扶抚坛技坏扰拒找批扯址走抄坝贡攻赤折抓扮抢孝均抛投坟抗坑坊抖护壳志扭块把却劫芽芹芬苍芳芦劳苏杆杠杜材村杏极李杨求更束豆两丽医否还歼来连步旱盯呈时吴县里园旷围呀吨足邮困吵串员听吩吹呜吧吼别岗帐财针钉我乱秃私每兵估体何但伸作伯伶佣你住位伴身皂佛近彻役返余希坐谷妥岔肝肚肠龟免狂犹删条卵岛迎饭饮系言冻状亩况床库疗这序辛弃冶忘间判灶灿弟汪沙汽沃泛沟没沈怀完宋宏牢究穷灾良证评补初社识诉诊词译君即层尿尾迟局改忌际陆阿阻附妙妖妨努忍劲鸡驱纱纳纲驳纵纷纸纹纺驴纽奉玩环责现表规抹拢拔拣担坦押抽拐拖拍者顶拆拥抵拘抱垃拉拦拌幸招坡披拨择抬其取苦茂苹苗英范茄茎茅林杯柜析板枪构杰枕或画卧刺枣卖矿码厕奔态欧垄妻轰顷转斩轮到非肯齿些虎虏肾贤尚旺具果味昆国昌畅易典固忠咐呼鸣咏呢岸岩帖罗帜岭凯败贩购图钓垂牧物乖刮秆季佳侍供使例版侄侦侧凭侨佩依的迫征爬彼径所舍金命斧爸采受乳贪念贫肤肺肢肿胀朋股肥胁周昏鱼兔狐忽狗备饰饱饲京享店夜庙府底剂郊废净盲放育闸闹郑券单炒炊炕炎炉沫浅法泄河沾泪油泊沿泡注泻泳泥沸波泽治怕怜学宗宜审宙官空帘'
)


class CnNameCodec:
    """汉字/数字/`_`/`-` ↔ 全小写 base36。"""

    BODY_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
    BODY_BASE = len(BODY_CHARS)
    CN_SIZE = 1295  # 10 digits + _- + 1283 hanzi

    def __init__(self) -> None:
        self.cn_chars = "0123456789_-" + _HANZI_1283
        if len(self.cn_chars) != self.CN_SIZE:
            raise RuntimeError(
                f"字表大小异常: {len(self.cn_chars)} != {self.CN_SIZE}"
            )
        self.cn_index = {ch: i for i, ch in enumerate(self.cn_chars)}
        self.cn_base = len(self.cn_chars)

    def _to_body(self, n: int) -> str:
        if n == 0:
            return self.BODY_CHARS[0]
        digits = []
        while n > 0:
            n, rem = divmod(n, self.BODY_BASE)
            digits.append(self.BODY_CHARS[rem])
        return "".join(reversed(digits))

    def _from_body(self, code: str) -> int:
        big = 0
        for c in code:
            big = big * self.BODY_BASE + self.BODY_CHARS.index(c)
        return big

    def plain_for_encode(self, text: str, *, alnum_only: bool = False) -> str:
        """兼容旧接口；新字表中 `_`/`-` 直接参与压码。"""
        return text or ""

    def encode(self, text: str, max_len: int = 10, *, alnum_only: bool = False) -> str:
        """中文名压成全小写 base36。

        ``alnum_only`` 保留兼容，新编码输出始终为 [a-z0-9]。
        """
        del alnum_only  # 接口兼容
        if not text:
            return self.BODY_CHARS[0]

        for ch in text:
            if ch not in self.cn_index:
                raise ValueError(
                    f"字符不在允许字表中（常用汉字/数字/_/-）: {ch!r}"
                )

        big_num = 0
        for ch in text:
            big_num = big_num * self.cn_base + self.cn_index[ch]

        result = self._to_body(big_num)
        if len(result) > max_len:
            raise ValueError(f"编码后长度{len(result)}超过{max_len}，字数太多装不下")
        return result

    def decode(self, code: str, char_count: int) -> str:
        if char_count == 0:
            return ""
        if not code or any(c not in self.BODY_CHARS for c in code):
            raise ValueError("无法解码：正文含非法字符")

        big_num = self._from_body(code)
        chars = []
        for _ in range(char_count):
            big_num, rem = divmod(big_num, self.cn_base)
            chars.append(self.cn_chars[rem])
        if big_num != 0:
            raise ValueError("无法解码：code 或 char_count 不正确")
        return "".join(reversed(chars))

    def decode_auto(self, code: str, max_chars: int = 16) -> str:
        """无 char_count 时取能解尽的最短结果（不含前导字表 '0' 填充）。"""
        if not code:
            return ""
        if any(c not in self.BODY_CHARS for c in code):
            raise ValueError("无法解码：含非 base36 字符")
        for n in range(1, max_chars + 1):
            try:
                text = self.decode(code, n)
            except ValueError:
                continue
            if text.startswith("0") and len(text) > 1:
                continue
            limit = max(len(code), max_chars)
            if self.encode(text, max_len=limit) == code:
                return text
        raise ValueError("无法解码：不是合法的 encode 结果")


class CnNameCodecLegacy:
    """历史：汉字/数字 ↔ base62 + 可选 `_…` 后缀（GB2312 一级字表）。"""

    CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
    BODY_CHARS = CHARSET[:-1]
    BODY_BASE = len(BODY_CHARS)
    ASCII_TAIL = "0123456789_"

    def __init__(self) -> None:
        self.cn_chars = "0123456789" + self._build_gb2312_l1()
        self.cn_index = {ch: i for i, ch in enumerate(self.cn_chars)}
        self.cn_base = len(self.cn_chars)

    @staticmethod
    def _build_gb2312_l1() -> str:
        chars = []
        for qu in range(16, 56):
            for wei in range(1, 95):
                try:
                    chars.append(bytes([0xA0 + qu, 0xA0 + wei]).decode("gb2312"))
                except UnicodeDecodeError:
                    pass
        return "".join(chars)

    def _to_body(self, n: int) -> str:
        if n == 0:
            return self.BODY_CHARS[0]
        digits = []
        while n > 0:
            n, rem = divmod(n, self.BODY_BASE)
            digits.append(self.BODY_CHARS[rem])
        return "".join(reversed(digits))

    def _from_body(self, code: str) -> int:
        big = 0
        for c in code:
            big = big * self.BODY_BASE + self.BODY_CHARS.index(c)
        return big

    def _split_underscore_tail(self, text: str) -> tuple[str, str]:
        i = len(text)
        while i > 0 and text[i - 1] in self.ASCII_TAIL:
            i -= 1
        head, tail = text[:i], text[i:]
        if "_" not in tail:
            return text, ""
        return head, tail

    def plain_for_encode(self, text: str, *, alnum_only: bool = False) -> str:
        if not text:
            return text
        if not alnum_only:
            return text
        head, u_tail = self._split_underscore_tail(text)
        if u_tail:
            return head + u_tail.replace("_", "")
        return text

    def encode(self, text: str, max_len: int = 10, *, alnum_only: bool = False) -> str:
        if not text:
            return self.BODY_CHARS[0]
        if alnum_only:
            head, tail = self.plain_for_encode(text, alnum_only=True), ""
        else:
            head, tail = self._split_underscore_tail(text)
        for ch in head:
            if ch not in self.cn_index:
                raise ValueError(f"字符不在允许字表中（数字或 GB2312 一级）: {ch!r}")
        big_num = 0
        for ch in head:
            big_num = big_num * self.cn_base + self.cn_index[ch]
        body = self._to_body(big_num) if head else ""
        result = body + tail
        if not result:
            result = self.BODY_CHARS[0]
        if len(result) > max_len:
            raise ValueError(f"编码后长度{len(result)}超过{max_len}，字数太多装不下")
        return result

    def decode(self, code: str, char_count: int) -> str:
        if char_count == 0:
            return ""
        if "_" in code:
            body, after = code.split("_", 1)
            tail = "_" + after
        else:
            body, tail = code, ""
        head_len = char_count - len(tail)
        if head_len < 0:
            raise ValueError("无法解码：char_count 小于后缀长度")
        if head_len == 0:
            if body:
                raise ValueError("无法解码：纯后缀但正文非空")
            return tail
        if not body or any(c not in self.BODY_CHARS for c in body):
            raise ValueError("无法解码：正文含非法字符")
        big_num = self._from_body(body)
        chars = []
        for _ in range(head_len):
            big_num, rem = divmod(big_num, self.cn_base)
            chars.append(self.cn_chars[rem])
        if big_num != 0:
            raise ValueError("无法解码：code 或 char_count 不正确")
        return "".join(reversed(chars)) + tail

    def decode_auto(self, code: str, max_chars: int = 16) -> str:
        if not code:
            return ""
        if "_" in code:
            _body, after = code.split("_", 1)
            tail_len = len("_" + after)
        else:
            tail_len = 0
        for n in range(tail_len, max_chars + 1):
            try:
                text = self.decode(code, n)
            except ValueError:
                continue
            head = text[: len(text) - tail_len] if tail_len else text
            if head.startswith("0") and len(head) > 1:
                continue
            limit = max(len(code), max_chars)
            if self.encode(text, max_len=limit) == code:
                return text
            if self.encode(text, max_len=limit, alnum_only=True) == code:
                return text
        raise ValueError("无法解码：不是合法的 encode 结果")


codec = CnNameCodec()


if __name__ == "__main__":
    # 阿琛（低沉自然男声）
    # 晓楠（理性平稳女声）
    # 小樱（温柔积极女声）
    samples = [
        ("小樱", 10),
        ("子豪", 10),
        ("温柔小姐姐", 10),
        ("南枝-1", 10),
        ("阿伟", 10),
        ("小楠", 10),
        ("低沉自然", 10),
        # ("温柔积极", 10),
        ("理性平稳", 10),
        # ("温柔积极", 10),
        ("理性阳光", 10),
    ]
    for name, max_len in samples:
        enc = codec.encode(name, max_len=max_len)
        dec = codec.decode_auto(enc)
        print(f"{name} -> {enc} ({len(enc)}) -> {dec}")
    print(f"字表大小: {codec.cn_base}；正文 base36 全小写")
