from pathlib import Path
import tarfile
from huggingface_hub import hf_hub_download


def main():
    domains = [
        # "healthcare",
        "education",
        # "construction",
        # "crm",
        # "energy",
        # "finance",
        # "commerce_manufacturing",
        # "legal",
    ]

    for domain in domains:
        archive_path = hf_hub_download(
            repo_id="Salesforce/UniDoc-Bench",
            filename=f"{domain}_pdfs.tar.gz",
            repo_type="dataset",
            local_dir="./data/unidoc_bench",
        )

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall("./data/pdfs/")

        it = Path(f"./data/pdfs/{domain}").glob("._*")
        for file in it:
            file.unlink()


if __name__ == "__main__":
    main()
