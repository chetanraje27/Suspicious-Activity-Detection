import nbformat

notebook_path = r"d:/Downloads/SHAR_Complete_Project/SHAR/notebooks/05_Phase6_Explainability.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# 1. Add Reload cell at the beginning (after setup)
reload_cell = nbformat.v4.new_code_cell(source="""import sys
import importlib
if 'model' in sys.modules:
    importlib.reload(sys.modules['model'])
from model import CNNLSTM
print("✅ Model module reloaded with latest fixes (float32 forced for LSTM).")""")

# Insert after the first setup cell
nb.cells.insert(2, reload_cell)

# 2. Update GradCAM class in its cell
for cell in nb.cells:
    if cell.cell_type == 'code' and "class GradCAM" in cell.source:
        cell.source = """class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        # Use full backward hook for modern PyTorch compatibility
        target_layer.register_full_backward_hook(self._save_gradients)
        target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        # Force float32 and disable autocast locally to ensure LSTM stability
        input_tensor = input_tensor.to(torch.float32)
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(1).item()
            
            self.model.zero_grad()
            # Target the specific class for backprop
            score = output[0, target_class]
            score.backward()
            
        grads = self.gradients.mean(dim=[0, 2, 3])
        act = self.activations[0]
        cam = (grads[:, None, None] * act).sum(0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam /= cam.max()
            
        return cam.cpu().numpy(), CLASS_NAMES[target_class]"""
        print("Updated GradCAM class in notebook.")

# 3. Fix Section 1.4 visualization loop if it uses old GradCAM
for cell in nb.cells:
    if cell.cell_type == 'code' and "grad_cam.generate(tensor)" in cell.source:
        # Just ensure tensor is float32 here too for good measure
        if "tensor = " in cell.source and ".float()" not in cell.source:
             cell.source = cell.source.replace("to(device)", "to(device).float()")
             print("Updated tensor to float32 in visualization cell.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook 05 fully updated with robust XAI logic.")
