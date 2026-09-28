import onnx

import numpy as np
import onnx
from onnx import numpy_helper


def change_model_batch(
    input_path,
    output_path,
    new_batch=4,
):
    model = onnx.load(input_path)

    # 只修改真正的模型输入，不修改 initializer 权重
    initializer_names = {
        initializer.name
        for initializer in model.graph.initializer
    }

    for model_input in model.graph.input:
        if model_input.name in initializer_names:
            continue

        tensor_type = model_input.type.tensor_type

        if not tensor_type.HasField("shape"):
            continue

        dims = tensor_type.shape.dim

        if len(dims) == 0:
            continue

        old_batch = dims[0]

        if old_batch.HasField("dim_value"):
            print(
                f"修改输入 {model_input.name}: "
                f"batch {old_batch.dim_value} -> {new_batch}"
            )
        elif old_batch.HasField("dim_param"):
            print(
                f"修改输入 {model_input.name}: "
                f"batch {old_batch.dim_param} -> {new_batch}"
            )

        # 清除原来的 dim_value 或 dim_param
        old_batch.Clear()

        # 设置新的固定 batch
        old_batch.dim_value = new_batch

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
        strict_mode=True,
        data_prop=True,
    )

    onnx.checker.check_model(model)
    onnx.save(model, output_path)


'''
change_model_batch(
    input_path="yolo11n.onnx",
    output_path="model_batch4.onnx",
    new_batch=4,
)
'''


def change_reshape_n(
    input_path,
    output_path,
    new_n=4,
    only_old_n=None,
    run_shape_inference=True,
):
    """
    修改所有静态 Reshape 节点目标 shape 的第 0 维。

    参数：
        input_path:
            输入 ONNX 模型路径。

        output_path:
            输出 ONNX 模型路径。

        new_n:
            新的第 0 维，例如 4。

        only_old_n:
            只修改第 0 维等于指定值的 Reshape。
            例如 only_old_n=1 表示只把 1 改成 4。
            设置为 None 表示无条件修改所有静态 Reshape。

        run_shape_inference:
            是否在修改后重新执行 shape inference。
    """
    model = onnx.load(input_path)
    graph = model.graph

    # initializer 名称 -> numpy 数组
    initializer_map = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }

    # Constant 节点输出名称 -> numpy 数组
    constant_map = {}

    for node in graph.node:
        if node.op_type != "Constant" or not node.output:
            continue

        for attr in node.attribute:
            if attr.name == "value":
                constant_map[node.output[0]] = numpy_helper.to_array(
                    attr.t
                )
                break

    static_shape_map = {
        **initializer_map,
        **constant_map,
    }

    changed_count = 0
    skipped_count = 0

    for index, node in enumerate(graph.node):
        if node.op_type != "Reshape":
            continue

        node_name = node.name or f"Reshape_{index}"

        if len(node.input) < 2:
            print(f"跳过 {node_name}: 没有 shape 输入")
            skipped_count += 1
            continue

        shape_input_name = node.input[1]

        if shape_input_name not in static_shape_map:
            print(
                f"跳过 {node_name}: shape '{shape_input_name}' "
                f"是动态计算得到的"
            )
            skipped_count += 1
            continue

        old_shape = np.asarray(
            static_shape_map[shape_input_name]
        ).copy()

        if old_shape.ndim != 1 or old_shape.size == 0:
            print(
                f"跳过 {node_name}: 非法目标 shape {old_shape}"
            )
            skipped_count += 1
            continue

        old_n = int(old_shape[0])

        if only_old_n is not None and old_n != only_old_n:
            print(
                f"跳过 {node_name}: "
                f"第 0 维为 {old_n}，不是 {only_old_n}"
            )
            skipped_count += 1
            continue

        new_shape = old_shape.copy()
        new_shape[0] = new_n

        # Reshape 的 shape 输入必须是整数类型，通常是 int64。
        new_shape = new_shape.astype(np.int64)

        # 每个 Reshape 使用独立的新 initializer，
        # 避免原 shape 常量还被其他节点使用。
        safe_node_name = (
            node_name
            .replace("/", "_")
            .replace(":", "_")
            .replace(".", "_")
        )

        new_shape_name = (
            f"{safe_node_name}_batch{new_n}_shape_{index}"
        )

        new_initializer = numpy_helper.from_array(
            new_shape,
            name=new_shape_name,
        )

        graph.initializer.append(new_initializer)

        # 让当前 Reshape 使用新的 shape
        node.input[1] = new_shape_name

        print(
            f"修改 {node_name}: "
            f"{old_shape.tolist()} -> {new_shape.tolist()}"
        )

        changed_count += 1

    # 删除旧的中间张量 shape 信息
    del graph.value_info[:]

    if run_shape_inference:
        # 删除旧输出 shape，防止与新的推导结果冲突
        for output in graph.output:
            tensor_type = output.type.tensor_type

            if tensor_type.HasField("shape"):
                tensor_type.ClearField("shape")

        try:
            model = onnx.shape_inference.infer_shapes(
                model,
                check_type=True,
                strict_mode=True,
                data_prop=True,
            )
            print("Shape inference 成功")
        except onnx.shape_inference.InferenceError as error:
            print("警告：Shape inference 失败")
            print(error)
            print("仍然保存修改后的模型，请用 ONNX Runtime 验证")

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    print()
    print(f"修改完成: {changed_count} 个 Reshape")
    print(f"跳过数量: {skipped_count}")
    print(f"输出模型: {output_path}")


change_reshape_n(
    input_path="yolo11n.onnx",
    output_path="model_reshape_n4.onnx",
    new_n=4,

    # 推荐只把原来 N=1 的 Reshape 改成 N=4
    only_old_n=1,

    run_shape_inference=True,
)