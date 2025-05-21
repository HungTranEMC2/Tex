import json


def return_form1040(path: str):
    with open(path, "rb") as file:
        form = json.load(file)

    return form
