import copy
import numpy as np
import onnx

from onnx import helper, numpy_helper, TensorProto


#增加dequatize算子到onnx模型的输入前
def add_uint8_dequantize_input(
    input_model_path: str,
    output_model_path: str,
    input_name: str | None = None,
    scale: float = 1.0 / 255.0,
    zero_point: int = 0,
):
    """
    在 float32 ONNX 模型输入前插入 DequantizeLinear：

        uint8_input
             |
        DequantizeLinear
             |
        original_float_input
             |
        原始模型

    参数：
        input_model_path:
            原始 FP32 ONNX 模型路径

        output_model_path:
            修改后的模型保存路径

        input_name:
            要修改的模型输入名称。
            如果模型只有一个输入，可以不指定。

        scale:
            DequantizeLinear 的 scale，默认 1/255

        zero_point:
            uint8 零点，默认 0
    """

    model = onnx.load(input_model_path)
    graph = model.graph

    # ---------------------------------------------------------
    # 1. 查找需要修改的模型输入
    # ---------------------------------------------------------
    initializer_names = {
        initializer.name for initializer in graph.initializer
    }

    # 排除某些旧模型中同时出现在 graph.input 里的 initializer
    real_inputs = [
        graph_input
        for graph_input in graph.input
        if graph_input.name not in initializer_names
    ]

    if input_name is None:
        if len(real_inputs) != 1:
            names = [value.name for value in real_inputs]
            raise ValueError(
                f"模型有 {len(real_inputs)} 个输入，请通过 input_name 指定。"
                f"可用输入：{names}"
            )
        original_input = real_inputs[0]
    else:
        original_input = next(
            (
                value
                for value in real_inputs
                if value.name == input_name
            ),
            None,
        )

        if original_input is None:
            names = [value.name for value in real_inputs]
            raise ValueError(
                f"找不到输入 {input_name!r}，可用输入：{names}"
            )

    if original_input.type.tensor_type.elem_type != TensorProto.FLOAT:
        current_type = TensorProto.DataType.Name(
            original_input.type.tensor_type.elem_type
        )
        raise TypeError(
            f"原输入 {original_input.name!r} 不是 FLOAT，"
            f"而是 {current_type}"
        )

    original_input_name = original_input.name
    uint8_input_name = f"{original_input_name}_uint8"

    scale_name = f"{original_input_name}_dequant_scale"
    zero_point_name = f"{original_input_name}_dequant_zero_point"
    node_name = f"{original_input_name}_DequantizeLinear"

    existing_names = {
        value.name for value in graph.input
    } | {
        value.name for value in graph.output
    } | {
        initializer.name for initializer in graph.initializer
    }

    for name in (
        uint8_input_name,
        scale_name,
        zero_point_name,
    ):
        if name in existing_names:
            raise ValueError(f"模型中已经存在同名张量：{name}")

    # ---------------------------------------------------------
    # 2. 将模型外部输入改名并修改为 UINT8
    #
    # 原来：
    #   graph input: images(float32)
    #
    # 修改后：
    #   graph input: images_uint8(uint8)
    #   DequantizeLinear output: images(float32)
    #
    # 原模型所有节点仍然使用 images，无需逐个修改。
    # ---------------------------------------------------------
    original_input.name = uint8_input_name
    original_input.type.tensor_type.elem_type = TensorProto.UINT8

    # ---------------------------------------------------------
    # 3. 创建 scale 和 zero_point
    # ---------------------------------------------------------
    scale_initializer = numpy_helper.from_array(
        np.array(scale, dtype=np.float32),
        name=scale_name,
    )

    zero_point_initializer = numpy_helper.from_array(
        np.array(zero_point, dtype=np.uint8),
        name=zero_point_name,
    )

    graph.initializer.extend([
        scale_initializer,
        zero_point_initializer,
    ])

    # ---------------------------------------------------------
    # 4. 创建 DequantizeLinear 节点
    # ---------------------------------------------------------
    dequant_node = helper.make_node(
        op_type="DequantizeLinear",
        inputs=[
            uint8_input_name,
            scale_name,
            zero_point_name,
        ],
        outputs=[original_input_name],
        name=node_name,
    )

    # 放在原始节点之前
    graph.node.insert(0, dequant_node)

    # ---------------------------------------------------------
    # 5. 检查 opset
    # DequantizeLinear 从 opset 10 开始支持
    # ---------------------------------------------------------
    default_opset = next(
        (
            opset.version
            for opset in model.opset_import
            if opset.domain in ("", "ai.onnx")
        ),
        None,
    )

    if default_opset is None:
        raise ValueError("模型没有默认 ONNX opset")

    if default_opset < 10:
        raise ValueError(
            f"当前模型 opset={default_opset}，"
            "DequantizeLinear 至少需要 opset 10。"
        )

    # ---------------------------------------------------------
    # 6. 检查并进行 shape inference
    # ---------------------------------------------------------
    onnx.checker.check_model(model)

    model = onnx.shape_inference.infer_shapes(
        model,
        strict_mode=True,
    )

    onnx.checker.check_model(model)
    onnx.save(model, output_model_path)

    print("修改完成")
    print(f"原始输入：{original_input_name}, FLOAT")
    print(f"新的输入：{uint8_input_name}, UINT8")
    print(f"scale：{np.float32(scale)}")
    print(f"zero_point：{zero_point}")
    print(f"输出模型：{output_model_path}")


if __name__ == "__main__":
    add_uint8_dequantize_input(
        input_model_path="yolo11n.onnx",
        output_model_path="yolo11n_uint8_input.onnx",

        # 如果只有一个模型输入，可以不填
        input_name="images",

        scale=1.0 / 255.0,
        zero_point=0,
    )
