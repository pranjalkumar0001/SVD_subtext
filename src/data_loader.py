import os


def load_txt_documents(file_path):
        """
        Load text documents from a .txt file.
    
        Args:
            file_path (str): Path to a folder containing .txt files.
        """
        documents = []
        for filename in os.listdir(file_path):
            if filename.endswith(".txt"):
                with open(os.path.join(file_path, filename), "r") as f:
                    documents.append(f.read())
        return documents