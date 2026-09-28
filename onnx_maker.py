import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

x = helper.make_tensor_value_info(
    "x", TensorProto.FLOAT, ["batch", 4]
)

y = helper.make_tensor_value_info(
    "y", TensorProto.FLOAT, ["batch", 2]
)

w_data = np.random.randn(4, 2).astype(np.float32)
b_data = np.random.randn(2).astype(np.float32)

w = numpy_helper.from_array(w_data, name="w")
b = numpy_helper.from_array(b_data, name="b")

matmul = helper.make_node(
    "MatMul",
    inputs = ["x", "w"],
    outputs=["xw"],
)

add = helper.make_node(
    "Add",
    inputs=["xw", "b"],
    outputs=["y"],
)

graph = helper.make_graph(
    [matmul, add],
    "linear_graph",
    [x],
    [y],
    [w, b],
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 18)],
)

onnx.checker.check_model(model)
onnx.save(model, "linear.onnx")
