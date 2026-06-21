from pathlib import Path

from documents.document import Document
from utils.image_utils import scale_image, image_to_base64
from PIL import Image
import settings


def generate_id_from_doc(doc: Document):
    data_type = doc.metadata["type"]

    doc_name = doc.metadata["doc"]
    page = doc.metadata["page"]

    match data_type:
        case "image":
            return tuple([doc_name, page, *doc.metadata["path"]])
        case "text":
            return tuple([doc_name, page, doc.metadata["text"]])
        case _:
            raise NotImplementedError()


def image_doc_to_image_path(doc: Document) -> Path:
    path_parts = list(doc.metadata["path"])
    path_parts[-1] = path_parts[-1] + ".jpg"
    return settings.VISUAL_DOC_STORE_DIR.joinpath(*path_parts)


def prepare_image_docs(
    docs: list[Document],
    min_pixels: int = 256 * 256,
    max_pixels: int = 784 * 1024,
) -> list[dict]:
    content = []

    prepare_description_template = "[Visual document {n}]\nFrom domain: {domain}, document: {doc_name}, page: {page}"

    for n, doc in enumerate(docs, start=1):
        text = prepare_description_template.format(
            n=n,
            domain=doc.metadata["domain"],
            doc_name=doc.metadata["doc"],
            page=doc.metadata["page"] + 1,
        )
        content.append(dict(type="text", text=text))

        image = Image.open(image_doc_to_image_path(doc))
        image = scale_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        base64_image = image_to_base64(image)
        content.append(
            dict(
                type="image_url",
                image_url=dict(url=f"data:image/jpeg;base64,{base64_image}"),
            )
        )

    return content


def prepate_text_docs(docs: list[Document]) -> list[dict]:
    content = []

    prepare_description_template = "[Text document {n}]\nFrom domain: {domain}, document: {doc_name}, page: {page}\nContent:{content}"

    for n, doc in enumerate(docs, start=1):
        text = prepare_description_template.format(
            n=n,
            domain=doc.metadata["domain"],
            doc_name=doc.metadata["doc"],
            page=doc.metadata["page"] + 1,
            content=doc.metadata["text"],
        )
        content.append(dict(type="text", text=text))

    return content


def convert_docs_to_content_repr(docs: dict[str, list[Document]]) -> list[dict]:
    image_docs = docs["image_docs"]
    text_docs = docs["text_docs"]

    content = []
    content.extend(prepare_image_docs(image_docs))
    content.extend(prepate_text_docs(text_docs))

    return content
