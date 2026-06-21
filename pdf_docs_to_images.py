import asyncio
from argparse import ArgumentParser
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Literal, Optional

import pymupdf
import tqdm


def pdf_file_to_images(
    file: Path,
    dpi: int = 180,
    target_size: Optional[tuple[int, int]] = None,
) -> list[dict]:
    file = Path(file)
    doc = pymupdf.open(file)

    images = []
    for page in doc.pages():
        image = page.get_pixmap(dpi=dpi).pil_image()

        if target_size is not None:
            image = image.resize(target_size)

        images.append(
            {
                "doc": file.stem,
                "page": page.number,
                "image": image,
                "type": "page",
            }
        )

    return images


@dataclass
class Args:
    input_dir: Path
    output_dir: Path
    dpi: int
    target_size: Optional[tuple[int, int]] = None


def parse_args() -> Args:
    parser = ArgumentParser()

    parser.add_argument("--input", "-i", type=Path, required=True, dest="input_dir")
    parser.add_argument("--output", "-o", type=Path, required=True, dest="output_dir")
    parser.add_argument("--dpi", type=int, default=180, required=False, dest="dpi")
    parser.add_argument(
        "--target-size",
        "-s",
        type=int,
        nargs=2,
        default=None,
        required=False,
        dest="target_size",
    )

    return Args(**vars(parser.parse_args()))


def main():
    args = parse_args()

    if args.output_dir.exists():
        raise Exception(f"The directory {args.output_dir} already exists.")

    getter = itemgetter("image", "doc", "page")

    pdf_files = list(args.input_dir.glob("*.pdf"))
    for pdf_file in tqdm.tqdm(pdf_files, ncols=100):
        pdf_data = pdf_file_to_images(
            pdf_file,
            dpi=args.dpi,
            target_size=None,
        )

        save_dir = args.outout_dir.joinpath(pdf_file.stem)
        save_dir.mkdir(parents=True, exist_ok=False)

        for image_data in pdf_data:
            image, doc, page = getter(image_data)

            file = save_dir.joinpath(f"{page:06d}.jpg")
            image.save(file)


if __name__ == "__main__":
    main()
