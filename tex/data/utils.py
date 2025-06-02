import json
from typing import Any, Dict, List

from tex.data.constants import DATA_PATH


def get_form_lines(
    year: int,
    form_name: str,
) -> List[str, Any]:
    with open(
        DATA_PATH.format(
            year=year,
            doc_type="forms",
            form_name=form_name,
        ),
        "r",
    ) as file:
        lines = json.load(file)["lines"]
    return lines


def get_instruction(year: int, instruction_name: str) -> Dict[str, Any]:
    with open(
        DATA_PATH.format(
            year=year,
            doc_type="instructions",
            instruction_name=instruction_name,
        ),
        "r",
    ) as file:
        return json.load(file)
