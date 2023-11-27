import os
import re
import base64
from typing import Dict
import requests
from functools import lru_cache
from llama_hub.confluence.base import ConfluenceReader as BaseConfluenceReader
from llama_index.indices.postprocessor import (
    SentenceTransformerRerank as BaseSentenceTransformerRerank,
)
from typing import List, Optional, Literal
from llama_index.callbacks import CBEventType, EventPayload
from llama_index.schema import MetadataMode, NodeWithScore, QueryBundle
from dataclasses import dataclass

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document

@dataclass
class File:
    content: str
    file_name: str
    url: str

class SentenceTransformerRerank(BaseSentenceTransformerRerank):
    def __init__(
        self,
        top_n: int = 2,
        model: str = "cross-encoder/stsb-distilroberta-base",
    ):
        super().__init__(top_n=top_n, model=model)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")

        query_and_nodes = [
            (
                query_bundle.query_str,
                node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            for node in nodes
        ]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            scores = self._model.predict(query_and_nodes)

            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores):
                node.score = score.item()

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                : self.top_n
            ]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes


class ConfluenceReader(BaseConfluenceReader):
    def __init__(
        self, base_url: str = None, oauth2: Dict | None = None, cloud: bool = True
    ) -> None:
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
    with open(cookie_file, "r") as fp:
        for line in fp:
            if not re.match(r"^\#", line):
                line_fields = line.strip().split("\t")
                cookies[line_fields[5]] = line_fields[6]
    return cookies


class BitbucketReader(BaseReader):
    """Bitbucket reader.

    Reads the content of files in Bitbucket repositories.

    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        project_key: Optional[str] = None,
        branch: Optional[str] = "refs/heads/develop",
        extensions_to_skip: Optional[List] = [],
    ) -> None:
        """Initialize with parameters."""
        if os.getenv("BITBUCKET_USERNAME") is None:
            raise ValueError("Could not find a Bitbucket username.")
        if os.getenv("BITBUCKET_API_KEY") is None:
            raise ValueError("Could not find a Bitbucket api key.")
        if base_url is None:
            raise ValueError("You must provide a base url for Bitbucket.")
        if project_key is None:
            raise ValueError("You must provide a project key for Bitbucket repository.")
        self.base_url = base_url
        self.project_key = project_key
        self.branch = branch
        self.extensions_to_skip = extensions_to_skip

    def get_headers(self):
        username = os.getenv("BITBUCKET_USERNAME")
        api_token = os.getenv("BITBUCKET_API_KEY")
        auth = base64.b64encode(f"{username}:{api_token}".encode()).decode()
        return {"Authorization": f"Basic {auth}"}

    def get_slugs(self) -> List:
        """
        Get slugs of the specific project.
        """
        repos_url = (
            f"{self.base_url}/rest/api/latest/projects/{self.project_key}/repos/"
        )
        headers = self.get_headers()
        slugs = []
        response = requests.get(repos_url, headers=headers)

        if response.status_code == 200:
            repositories = response.json()["values"]
            for repo in repositories:
                repo_slug = repo["slug"]
                slugs.append(repo_slug)
        return slugs

    def load_all_file_paths(self, slug, branch, directory_path="", paths=[]):
        """
        Go inside every file that is present in the repository and get the paths for each file
        """
        content_url = f"{self.base_url}/rest/api/latest/projects/{self.project_key}/repos/{slug}/browse/{directory_path}"
        query_params = {
            "at": branch,
        }
        headers = self.get_headers()
        response = requests.get(content_url, headers=headers, params=query_params)
        response = response.json()
        if "errors" in response:
            raise ValueError(response["errors"])
        children = response["children"]
        for value in children["values"]:
            if value["type"] == "FILE":
                if (
                    value["path"].get("extension")
                    and value["path"]["extension"] not in self.extensions_to_skip
                ):
                    paths.append(
                        {
                            "slug": slug,
                            "path": f'{directory_path}/{value["path"]["toString"]}',
                        }
                    )
            elif value["type"] == "DIRECTORY":
                self.load_all_file_paths(
                    slug=slug,
                    branch=branch,
                    directory_path=f'{value["path"]["toString"]}'
                    if directory_path == ""
                    else f'{directory_path}/{value["path"]["toString"]}',
                    paths=paths,
                )

    def load_text_by_paths(self, slug, file_path, branch) -> List:
        """
        Go inside every file that is present in the repository and get the paths for each file
        """
        if not file_path.startswith("/"):
            file_path = "/" + file_path
        content_url = f"{self.base_url}/rest/api/latest/projects/{self.project_key}/repos/{slug}/browse{file_path}"

        query_params = {
            "at": branch,
        }
        headers = self.get_headers()
        # print(content_url)  # DEBUG
        response = requests.get(content_url, headers=headers, params=query_params)
        # print(file_path)  # DEBUG
        children = response.json()
        if "errors" in children:
            # raise ValueError(children["errors"]) # DEBUG
            return []
        if "lines" in children:
            return children["lines"]
        return []

    def load_text(self, paths) -> List[File]:
        text_dict = []
        for path in paths:
            lines_list = self.load_text_by_paths(
                slug=path["slug"], file_path=path["path"], branch=self.branch
            )
            concatenated_string = ""

            for line_dict in lines_list:
                text = line_dict.get("text", "")
                concatenated_string = concatenated_string + " " + text

            file_path = (
                path["path"] if not path["path"].startswith("/") else path["path"][1:]
            )

            text_dict.append(
                File(
                    content=concatenated_string,
                    file_name=path["path"].split("/")[-1],
                    url=f"{self.base_url}/projects/{self.project_key}/repos/{path['slug']}/browse/{file_path}",
                )
            )
        return text_dict

    def load_data(self) -> List[Document]:
        """Return a list of Document made of each file in Bitbucket."""
        slugs = self.get_slugs()
        paths = []
        for slug in slugs:
            if slug in ["dsbt"]:
                continue
            self.load_all_file_paths(
                slug=slug, branch=self.branch, directory_path="", paths=paths
            )
        texts = self.load_text(paths)
        print(len(texts))
        return [
            Document(
                text=text.content,
                extra_info={"file_name": text.file_name, "url": text.url},
            )
            for text in texts
        ]


def test_bitbucketReader():
    loader = BitbucketReader(
        project_key="MOBDATA",
        base_url="https://bitbucket.valtech.de",
        branch="master",
        extensions_to_skip=[
            ".VIN-decoding",
            "URL-generalization",
            "scraping",
            "FizzBuzz",
            "Driver-Behaviour",
            "VIN-OCR",
            "Sensor-Log",
            ".png",
            ".jpg",
        ],
    )

    documents = loader.load_data()
    print(len(documents))


if __name__ == "__main__":
    test_bitbucketReader()
