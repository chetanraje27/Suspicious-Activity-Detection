import nbformat
import os
from pathlib import Path

notebooks_dir = Path(r"d:/Downloads/SHAR_Complete_Project/SHAR/notebooks")
notebook_files = list(notebooks_dir.glob("*.ipynb"))

for nb_path in notebook_files:
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if "D:\\Downloads" in cell.source:
                    print(f"Found absolute path in: {nb_path.name}")
                    print(f"Cell source snippet: {cell.source[:100]}...")
            
    except Exception as e:
        pass
