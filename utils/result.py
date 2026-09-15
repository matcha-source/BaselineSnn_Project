import json
from pathlib import Path


def save_history(history, file_path):
    """
    Save training history to a JSON file.
    """

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with file_path.open("w") as file:
        json.dump(
            history,
            file,
            indent=4
        )