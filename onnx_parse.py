import onnx
from collections import Counter

model = onnx.load("yolo11n.onnx")
print("IR version", model.ir_version)

for opset in model.opset_import:
    print("Opset version", opset.version, "for domain", opset.domain)

print("\nInputs:")

for value in model.graph.input:
    print(value.name)

print("\nOutputs:")
for value in model.graph.output:
    print(value.name)

op_counts = Counter(node.op_type for node in model.graph.node)

print("\nOperators:")
for op_type, count in op_counts.items():
    print(f"{op_type}: {count}")

onnx.checker.check_model(model)
print("\nModel is valid ONNX")