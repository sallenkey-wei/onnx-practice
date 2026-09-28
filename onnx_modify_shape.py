# 此脚本目的是修改 输入shape [1, 3, 640, 640] 为 [1, 3, 1280, 1280]，并重新执行 shape inference
# 但是由于reshape算子中常量shape仍然是[1, 128, 20, 20]，所以导致add算子[1, 128, 20, 20] + [1, 128, 40, 40]不兼容，导致失败
import onnx

model = onnx.load("yolo11n.onnx")

input_name = "images"
new_shape = [1, 3, 1280, 1280]

for model_input in model.graph.input:
    if model_input.name == input_name:
        tensor_shape = model_input.type.tensor_type.shape

        tensor_shape.ClearField("dim")

        for value in new_shape:
            dim = tensor_shape.dim.add()
            dim.dim_value = value
        break
else:
    raise ValueError(f"找不到模型输入：{input_name}")

# 2. 清除中间 Tensor 的旧 Shape
# 保留 ValueInfo 和元素类型
for value_info in model.graph.value_info:
    tensor_type = value_info.type.tensor_type

    if tensor_type.HasField("shape"):
        tensor_type.ClearField("shape")


# 3. 清除模型输出的旧 Shape
# 保留输出名称和数据类型，只删除维度信息
for model_output in model.graph.output:
    tensor_type = model_output.type.tensor_type

    if tensor_type.HasField("shape"):
        #tensor_type.shape.ClearField("dim")
        tensor_type.ClearField("shape")


# 4. 重新执行 Shape Inference
model = onnx.shape_inference.infer_shapes(
    model,
    check_type=True,
    strict_mode=False,
    data_prop=True,
)


onnx.checker.check_model(model)
onnx.save(model, "model_new_shape.onnx")