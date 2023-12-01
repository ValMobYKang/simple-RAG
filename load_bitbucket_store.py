import os
import base64
import requests
from typing import List, Optional
from dataclasses import dataclass

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
)

@dataclass
class File:
    content: str
    file_name: str
    url: str


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
        try:
            response = requests.get(
                content_url, headers=headers, params=query_params, timeout=5
            )
        except Exception as e:
            print(f"[Connection Error]: {e}")
            return []
        children = response.json()
        if "errors" in children:
            # raise ValueError(children["errors"]) # DEBUG
            print(f"[ValueError]: {content_url}")
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

        print(f"Load slugs done! A total of {len(paths)}")
        texts = self.load_text(paths)
        return [
            Document(
                text=text.content,
                extra_info={"file_name": text.file_name, "url": text.url},
            )
            for text in texts
        ]


if __name__ == "__main__":
    print("Start to load bitbucket store ...")
    for param in [
            "BITBUCKET_PROJECT",
            "BITBUCKET_URL",
            "BITBUCKET_USERNAME",
            "BITBUCKET_API_KEY",
        ]: 
        assert os.environ.get(param) != None, f"{param} environment variables is not defined"
 
    index = VectorStoreIndex.from_documents(
        documents=BitbucketReader(
            project_key=os.environ["BITBUCKET_PROJECT"],
            base_url=os.environ["BITBUCKET_URL"],
            branch="master",
            extensions_to_skip=[
                ".VIN-decoding",
                "URL-generalization",
                "scraping",
                "FizzBuzz",
                "Driver-Behaviour",
                "VIN-OCR",
                "Sensor-Log",
                "png",
                "jpg",
                "ppm",
            ],
        ).load_data(),
        service_context=ServiceContext.from_defaults(
            llm=None,
            chunk_size=512,
            chunk_overlap=20,
            embed_model=HuggingFaceEmbedding(model_name="/embedding_model" if os.path.exists("/embedding_model") else "BAAI/bge-small-en-v1.5"), 
        ),
        show_progress=True,
    )

    index.storage_context.persist(persist_dir="store")
