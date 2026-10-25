import pytest

from src.data_loader import load_txt_documents


def test_load_documents_in_filename_order(tmp_path):
    (tmp_path / "b.txt").write_text("Second document.", encoding="utf-8")
    (tmp_path / "a.TXT").write_text("First document.", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("Ignore this.", encoding="utf-8")
    (tmp_path / "folder.txt").mkdir()

    documents = load_txt_documents(tmp_path)

    assert documents == {
        "a.TXT": "First document.",
        "b.txt": "Second document.",
    }

    # Dictionary equality checks contents, not insertion order.
    assert list(documents) == ["a.TXT", "b.txt"]


def test_load_utf8_with_bom(tmp_path):
    (tmp_path / "example.txt").write_text(
        "Café robot.", encoding="utf-8-sig"
    )

    documents = load_txt_documents(tmp_path)

    assert documents == {"example.txt": "Café robot."}


def test_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_txt_documents(tmp_path / "missing")


def test_file_instead_of_folder(tmp_path):
    path = tmp_path / "example.txt"
    path.write_text("Robot.", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        load_txt_documents(path)


def test_folder_without_txt_files(tmp_path):
    (tmp_path / "notes.csv").write_text("Ignore this.", encoding="utf-8")

    with pytest.raises(ValueError):
        load_txt_documents(tmp_path)