import io
import zipfile
import requests
import frontmatter


def read_repo_data(repo_owner: str, repo_name: str) -> list[dict]:
    """
    Download and parse all markdown files from a GitHub repository.

    Args:
        repo_owner: GitHub username or organization
        repo_name: Repository name

    Returns:
        List of dictionaries containing file content and metadata
    """
    prefix = 'https://codeload.github.com'
    url = f'{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)

    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename = file_info.filename
        filename_lower = filename.lower()

        if not (filename_lower.endswith('.md')
            or filename_lower.endswith('.mdx')):
            continue

        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode('utf-8', errors='ignore')
                post = frontmatter.loads(content)
                data = post.to_dict()
                data['filename'] = filename
                repository_data.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    zf.close()
    return repository_data


def print_example(docs: list[dict], index: int = 0):
    """Print a single document example."""
    if not docs:
        print("No documents found")
        return
    doc = docs[index]
    print(f"Filename: {doc.get('filename', 'N/A')}")
    print("-" * 50)
    for key, value in doc.items():
        if key != 'filename':
            print(f"{key}: {value[:200] if isinstance(value, str) and len(value) > 200 else value}")
    print("-" * 50)


def main(show_example: bool = False):
    print("Ingesting repository data...")

    dtc_faq = read_repo_data('DataTalksClub', 'faq')
    evidently_docs = read_repo_data('evidentlyai', 'docs')

    print(f"FAQ documents: {len(dtc_faq)}")
    print(f"Evidently documents: {len(evidently_docs)}")

    if show_example:
        print("\n=== Example FAQ document ===")
        print_example(dtc_faq)

    return dtc_faq, evidently_docs


if __name__ == "__main__":
    import sys
    show_example = "--example" in sys.argv
    main(show_example=show_example)
