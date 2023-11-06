import re
from typing import Dict
import requests
from functools import lru_cache
from llama_hub.confluence.base import ConfluenceReader as BaseConfluenceReader

class ConfluenceReader(BaseConfluenceReader):
    def __init__(self, base_url: str = None, oauth2: Dict | None = None, cloud: bool = True) -> None:
        super().__init__(base_url, oauth2, cloud)

    def process_pdf(self, link):
        try:
            import pytesseract  # type: ignore
            from pdf2image import convert_from_bytes  # type: ignore
        except ImportError:
            raise ImportError(
                "`pytesseract` or `pdf2image` package not found, please run `pip"
                " install pytesseract pdf2image`"
            )

        import pytesseract  # type: ignore
        from pdf2image import convert_from_bytes  # type: ignore
        
        response = self.confluence.request(path=link, absolute=True)
        text = ""
        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text
        try:
            images = convert_from_bytes(response.content)
        except ValueError:
            return text

        for i, image in enumerate(images):
            image_text = pytesseract.image_to_string(image)
            text += f"Page {i + 1}:\n{image_text}\n\n"

        return text

    def process_image(self, link):
        try:
            from io import BytesIO  # type: ignore

            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError:
            raise ImportError(
                "`pytesseract` or `Pillow` package not found, please run `pip install"
                " pytesseract Pillow`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""
        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text
        try:
            image = Image.open(BytesIO(response.content))
        except OSError:
            return text

        return pytesseract.image_to_string(image)

    def process_doc(self, link):
        try:
            from io import BytesIO  # type: ignore

            import docx2txt  # type: ignore
        except ImportError:
            raise ImportError(
                "`docx2txt` package not found, please run `pip install docx2txt`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text
        file_data = BytesIO(response.content)

        return docx2txt.process(file_data)

    def process_xls(self, link):
        try:
            import xlrd  # type: ignore
        except ImportError:
            raise ImportError("`xlrd` package not found, please run `pip install xlrd`")

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text

        workbook = xlrd.open_workbook(file_contents=response.content)
        for sheet in workbook.sheets():
            text += f"{sheet.name}:\n"
            for row in range(sheet.nrows):
                for col in range(sheet.ncols):
                    text += f"{sheet.cell_value(row, col)}\t"
                text += "\n"
            text += "\n"

        return text

    def process_svg(self, link):
        try:
            from io import BytesIO  # type: ignore

            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
            from reportlab.graphics import renderPM  # type: ignore
            from svglib.svglib import svg2rlg  # type: ignore
        except ImportError:
            raise ImportError(
                "`pytesseract`, `Pillow`, or `svglib` package not found, please run"
                " `pip install pytesseract Pillow svglib`"
            )

        response = self.confluence.request(path=link, absolute=True)
        text = ""

        if (
            response.status_code != 200
            or response.content == b""
            or response.content is None
        ):
            return text

        drawing = svg2rlg(BytesIO(response.content))

        img_data = BytesIO()
        renderPM.drawToFile(drawing, img_data, fmt="PNG")
        img_data.seek(0)
        image = Image.open(img_data)

        return pytesseract.image_to_string(image)



def cookie_request(path, absolute):
    assert absolute
    return requests.get(path, cookies=parse_cookie())

@lru_cache
def parse_cookie(cookie_file="./cookies.txt"):
    cookies = {}
    with open (cookie_file, 'r') as fp:
        for line in fp:
            if not re.match(r'^\#', line):
                line_fields = line.strip().split('\t')
                cookies[line_fields[5]] = line_fields[6]
    return cookies
