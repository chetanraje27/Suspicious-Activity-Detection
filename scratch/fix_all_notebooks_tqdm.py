import nbformat
import os
from pathlib import Path

notebooks_dir = Path(r"d:/Downloads/SHAR_Complete_Project/SHAR/notebooks")
notebook_files = list(notebooks_dir.glob("*.ipynb"))

print(f"Found {len(notebook_files)} notebooks to check.")

for nb_path in notebook_files:
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
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
            with open(nb_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            print(f"Fixed tqdm in: {nb_path.name}")
        else:
            print(f"No tqdm.notebook found in: {nb_path.name}")
            
    except Exception as e:
        print(f"Error processing {nb_path.name}: {e}")

print("Batch update complete.")
