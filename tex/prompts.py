from tex.utils.data import return_form1040

form1040 = return_form1040(
    path="tex/data/2024/forms/1040.json",
)


def return_global_instruction() -> str:
    return """
    You are a tax expert. You will look at provided tax-filing instructiions \
        and associated forms (e.g., W2 or 1099). Based on the given \
            information, you must respond the instruction with numbers.
    """


def return_instructions() -> str:
    prompt_instruction = """
    <|begin_of_text|>
    <|start_header_id|>{role}<|end_header_id|>
    """
    return prompt_instruction
