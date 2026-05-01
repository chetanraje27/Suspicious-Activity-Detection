import nbformat

notebook_path = r"d:/Downloads/SHAR_Complete_Project/SHAR/notebooks/05_Phase6_Explainability.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Find the GradCAM class cell and add instantiation after it if not present
instantiation_code = """# Hook into last conv layer of ResNet50 for Grad-CAM
target_layer = list(model.cnn.children())[-2][-1].conv3
grad_cam = GradCAM(model, target_layer)
print("✅ Grad-CAM initialized on last ResNet50 conv block.")"""

gradcam_class_index = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "class GradCAM:" in cell.source:
        gradcam_class_index = i
        break

if gradcam_class_index != -1:
    # Check if next cell is already instantiation
    found = False
    if gradcam_class_index + 1 < len(nb.cells):
        if "grad_cam =" in nb.cells[gradcam_class_index + 1].source:
            found = True
            print("Instantiation cell already exists.")
    
    if not found:
        print("Inserting instantiation cell.")
        new_cell = nbformat.v4.new_code_cell(source=instantiation_code)
        nb.cells.insert(gradcam_class_index + 1, new_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook 05 fixed: Grad-CAM instantiation restored.")
