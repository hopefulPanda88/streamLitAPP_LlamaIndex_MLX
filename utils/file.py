import json
import os.path
from typing import AnyStr, List, Dict
from pathlib import Path

import PyPDF2


def check_file_type(file_path: AnyStr, type_list: List[AnyStr]) -> AnyStr:
    """
    Checks if the file type is valid
    :param file_path: file path
    :param type_list: valid file types
    :return: the type of the extension.
    """
    path = Path(file_path)
    file_extension = path.suffix.lower()[1:]
    if file_extension not in type_list:
        raise TypeError(f"the supported file type is {', '.join(type_list)}")
    else:
        return file_extension


def read_file(file_path: AnyStr) -> AnyStr:
    """
    Reads the file based on its extension(currently only txt and pdf files are supported).
    However, there still coule be issues not settled: for example, if the file
    size is too big for the standard functions such as file.read() or extractText() to handle.
    :param file_path: path of the file to be read
    :return: The content of the file
    """
    if not os.path.exists(file_path):
        raise FileExistsError(f" The file path doesn't exists: {file_path}")

    path = Path(file_path)
    extension = path.suffix.lower()[1:]
    if extension not in ["txt", "pdf"]:
        raise TypeError(f"The file type : {extension} has not been supported yet")

    content = ""
    if extension == "txt":
        with open(file_path, "r") as file:
            content = file.read()

    if extension == "pdf":
        reader = PyPDF2.PdfReader(file_path)
        num_pages = len(reader.pages)

        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            content += text

    return content


def read_json_file(file_path: str) -> Dict[str, str]:
    """
    Read the json file and return the data dictionary
    Args:
        file_path: file path of the json file to be read

    Returns:
        The data dictionary
    """
    if os.path.isfile(file_path):
        if check_file_type(file_path, ["json"]):
            with open(file_path, "r") as file:
                data = json.load(file)
            return data
        else:
            raise TypeError(f"the type of the file: {file_path} to be read is not supported!")
    else:
        raise FileNotFoundError(f"{file_path} is not valid!")
