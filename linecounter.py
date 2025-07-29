import nbformat


def count_lines_in_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    line_count = 0
    for cell in nb.cells:
        if cell.cell_type == "code":
            lines = cell.source.splitlines()
            line_count += len(lines)
    return line_count


# Example usage
notebook_path = "liv_vadankhan_analyser\ITH.ipynb"
print(f"Total code lines: {count_lines_in_notebook(notebook_path)}")
