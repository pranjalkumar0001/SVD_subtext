from pathlib import Path

def load_txt_documents(folder_path: str | Path) -> dict[str, str]:
    """
    Loads all .txt documents from the specified folder and returns a dictionary
    where keys are filenames (with extension) and values are the corresponding
    document contents.

    Args:
        folder_path (str | Path): The path to the folder containing .txt files."""

    folder =  Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"The provided path '{folder_path}' is not a valid directory.")
    if not folder.is_dir():
        raise NotADirectoryError(f"The provided path '{folder_path}' is not a valid directory.")

    files = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        ),
        key=lambda path: path.name)
    if not files:
        raise ValueError(f"No .txt files found in the directory '{folder_path}'.")

    doc_names = [file.name for file in files]
    doc_contents = [file.read_text(encoding="utf-8-sig") for file in files]
    return dict(zip(doc_names, doc_contents))

