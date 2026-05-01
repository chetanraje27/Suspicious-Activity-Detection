import nbformat

notebook_path = r"d:/Downloads/SHAR_Complete_Project/SHAR/notebooks/01_EDA_and_Preprocessing.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

changed = False
for cell in nb.cells:
    if cell.cell_type == 'code':
        if "from tqdm.notebook import tqdm" in cell.source:
            cell.source = cell.source.replace("from tqdm.notebook import tqdm", "from tqdm.auto import tqdm")
            changed = True
        elif "tqdm.notebook" in cell.source:
             cell.source = cell.source.replace("tqdm.notebook", "tqdm.auto")
             changed = True

if changed:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook updated: Switched tqdm.notebook to tqdm.auto for better compatibility.")
else:
    print("No tqdm.notebook occurrences found.")
