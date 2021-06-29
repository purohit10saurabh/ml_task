from utils import *

model = torch.load('../cpu_model.pt')
model.eval()
example = torch.rand(1, 4, 84, 84)
with torch.no_grad():
	traced_script_module = torch.jit.trace(model, example)
	output = traced_script_module(torch.ones(1, 4, 84, 84))
print(output)

sm = torch.jit.script(traced_script_module)
traced_script_module.save("../traced_model.pt")

