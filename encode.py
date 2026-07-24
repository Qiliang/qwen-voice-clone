class CnNameCodec:
    """汉字/数字音色名 ↔ base62（正文）+ 可选 `_…` 后缀。"""

    CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
    BODY_CHARS = CHARSET[:-1]  # base62，不含 _；正文永不出现 _，故首个 _ 可唯一切开后缀
    BODY_BASE = len(BODY_CHARS)
    ASCII_TAIL = "0123456789_"

    def __init__(self) -> None:
        # 中间/末尾纯数字走字表；带 _ 的末尾（如 _2）默认可原样追加
        self.cn_chars = "0123456789" + self._build_gb2312_l1()
        self.cn_index = {ch: i for i, ch in enumerate(self.cn_chars)}
        self.cn_base = len(self.cn_chars)

    @staticmethod
    def _build_gb2312_l1() -> str:
        """GB2312 一级常用汉字（16–55 区，3755 字）。"""
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
        """只切「含 _ 的末尾」：灵动欣欣_2 → (灵动欣欣, _2)；灵动欣欣2 不切。"""
        i = len(text)
        while i > 0 and text[i - 1] in self.ASCII_TAIL:
            i -= 1
        head, tail = text[:i], text[i:]
        if "_" not in tail:
            return text, ""
        return head, tail

    def plain_for_encode(self, text: str, *, alnum_only: bool = False) -> str:
        """编码前实际参与压码的字符串（决定 decode 的 char_count）。"""
        if not text:
            return text
        if not alnum_only:
            return text
        head, u_tail = self._split_underscore_tail(text)
        if u_tail:
            return head + u_tail.replace("_", "")
        return text

    def encode(self, text: str, max_len: int = 10, *, alnum_only: bool = False) -> str:
        """汉字/数字压成 base62。

        alnum_only=False（默认）：末尾 `_…` 原样追加（适合 preferred_name）。
        alnum_only=True：`_…` 折叠进正文（`欣欣_2`→按`欣欣2`编），输出仅 [A-Za-z0-9]
        （适合 CosyVoice / qwen-audio prefix）。完整原名请另存 display_name。
        """
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
        """无 char_count 时取能解尽的最短结果（不含前导字表 '0' 填充）。"""
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


# 默认单例，供业务直接使用
codec = CnNameCodec()


if __name__ == "__main__":
    name = "温柔小姐姐"
    enc = codec.encode(name)
    dec = codec.decode(enc, len(name))
    print(f"{name} -> {enc} (长度{len(enc)}) -> {dec}")

    name2 = "灵动欣欣_2"
    enc2 = codec.encode(name2, max_len=10, alnum_only=True)
    # alnum 折叠后按「灵动欣欣2」解
    print(f"{name2} -> {enc2} (alnum, 长度{len(enc2)}) -> {codec.decode(enc2, 5)}")
    print(f"字表大小: {codec.cn_base}；正文 base62，末尾 _… 原样追加或折叠")
