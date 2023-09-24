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
        line = '","'.join([str(v) for v in values.values()])
        file.write(f'"{line}"\n')