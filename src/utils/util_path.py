from pathlib import Path


class UtilPath:
    @staticmethod
    def get_root_path():
        return str(Path(__file__).resolve().parent.parent.parent)