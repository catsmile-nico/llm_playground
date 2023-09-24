import os, tiktoken
def write_csv_line(csv_path: str, values: dict):
    """Writes csv line(values) into csv (csv_path)

    Args:
        csv_path (str): path of csv file to be written
        values (dict): keys are headers, values are values
    """
    if not os.path.exists(csv_path): 
        with open(csv_path, "w") as file: 
            line = '","'.join([str(v) for v in values.keys()])
            file.write(f'"{line}"\n')

    with open(csv_path, "a") as file:
        line = '","'.join([str(v).replace('"', "'") for v in values.values()])
        file.write(f'"{line}"\n')

def count_tokens(text):
    ENCODING = tiktoken.encoding_for_model("gpt-4")
    return len(ENCODING.encode(text))

def limit_tokens(text, max_tokens):
    ENCODING = tiktoken.encoding_for_model("gpt-4")
    tokens = ENCODING.encode(text)
    return ENCODING.decode(tokens[:max_tokens])