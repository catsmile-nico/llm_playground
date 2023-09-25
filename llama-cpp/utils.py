import os, re, tiktoken

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

def count_tokens(text, model="gpt-4"):
    """Count number of tokens the text has

    Args:
        text (str): Text to count
        model (str, optional): Tokenizer model to use. Defaults to "gpt-4".

    Returns:
        int: The token count of the text
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def limit_tokens(text, max_tokens, model="gpt-4"):
    """Limits the text to meet max_token limit

    Args:
        text (str): Text to shrink
        max_tokens (int): Maximum token allowed for text
        model (str, optional): Tokenizer model to use. Defaults to "gpt-4".

    Returns:
        str: The text shrunk to the token limit
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])

def extract_json_category(text):
    """From a text, extract all the json string pair category and sub-category

    Args:
        text (str): String to be parsed

    Returns:
        list: list of json strings
    """

    # match the JSON string with dynamic values
    pattern = r'{\s*"CATEGORY":\s*"(.*?)",\s*"SUB-CATEGORY":\s*\["(.*?)"\]\s*}'
    matches = re.findall(pattern, text)

    # Extract matched strings
    json_strings = []
    for match in matches:
        category = match[0]
        sub_category = match[1]
        json_string = f'{{ "CATEGORY": "{category}", "SUB-CATEGORY": ["{sub_category}"] }}'
        json_strings.append(json_string)

    return json_strings