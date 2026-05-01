import nbformat

notebook_path = r"d:/Downloads/SHAR_Complete_Project/SHAR/notebooks/05_Phase6_Explainability.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

changed = False
for cell in nb.cells:
    if cell.cell_type == 'code':
        # Update Grad-CAM hook to use full backward hook
        if "target_layer.register_backward_hook" in cell.source:
            cell.source = cell.source.replace("target_layer.register_backward_hook", "target_layer.register_full_backward_hook")
            changed = True
        
        # Update _save_gradients signature for full_backward_hook (it gets grad_input, grad_output instead of module, grad_in, grad_out)
        # Actually register_full_backward_hook(self, module, grad_input, grad_output)
        # Wait, the signature is same: (module, grad_input, grad_output)
        # but the content of grad_output is a tuple.
        
if changed:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook updated: Switched to register_full_backward_hook.")
else:
    print("No changes needed in notebook.")
