from pydantic import BaseModel, Field, field_validator


class C2NMap(BaseModel):
    c2n: dict[str, int] = Field(description="文字から数字へのマッピング")

    @field_validator("c2n")
    @classmethod
    def validate_c2n(cls, v: dict[str, int]) -> dict[str, int]:
        for key, value in v.items():
            if len(key) != 1:
                raise ValueError(f"キーは1文字である必要があります: {key}")
            if not 0 <= value <= 9:
                raise ValueError(f"値は0から9の間である必要があります: {value}")
        return v

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
