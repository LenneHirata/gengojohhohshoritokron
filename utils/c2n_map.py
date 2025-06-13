class C2NMap:
    def __init__(self, c2n: dict[str, int]):
        self.c2n = c2n
        self.n2c = {i: [] for i in range(10)}
        for key, value in self.c2n.items():
            self.n2c[value].append(key)

    def __encode(self, text: str) -> str:
        if len(text) != 1:
            raise ValueError(f"文字は1文字である必要があります: {text}")

        if text == " ":
            return " "

        if text not in self.c2n:
            raise ValueError(f"文字はマッピングに含まれていません: {text}")

        return str(self.c2n[text])

    def __call__(self, text: str) -> str:
        return "".join(map(self.__encode, text))
