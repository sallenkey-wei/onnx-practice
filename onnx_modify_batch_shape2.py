import copy
import os
from functools import lru_cache

import numpy as np
import onnx
from onnx import numpy_helper


def get_tensor_shapes(model):
    """
    收集模型中已经声明或推导出的所有张量 shape。

    返回：
        {
            tensor_name: [dim0, dim1, ...]
        }

    未知维度使用 None 表示。
    """
    shape_map = {}

    value_infos = (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    )

    for value_info in value_infos:
        tensor_type = value_info.type.tensor_type

        if not tensor_type.HasField("shape"):
            continue

        shape = []

        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape.append(None)
            else:
                shape.append(None)

        shape_map[value_info.name] = shape

    return shape_map


def format_shape(shape):
    if shape is None:
        return "unknown"

    return str([
        "?" if dim is None else dim
        for dim in shape
    ])


def collect_all_tensor_names(model):
    """收集模型中已经使用的名称，避免创建重名 initializer。"""
    names = set()

    for value_info in model.graph.input:
        names.add(value_info.name)

    for value_info in model.graph.output:
        names.add(value_info.name)

    for value_info in model.graph.value_info:
        names.add(value_info.name)

    for initializer in model.graph.initializer:
        names.add(initializer.name)

    for node in model.graph.node:
        names.update(name for name in node.input if name)
        names.update(name for name in node.output if name)

    return names


def make_unique_name(base_name, used_names):
    """生成不重复的张量名称。"""
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    index = 1

    while True:
        candidate = f"{base_name}_{index}"

        if candidate not in used_names:
            used_names.add(candidate)
            return candidate

        index += 1


def clear_old_shape_information(model):
    """
    删除 batch=1 时保存的旧 shape 信息。

    保留：
        - graph.input
        - TensorProto 元素类型
        - initializer

    删除：
        - graph.value_info 中间 shape
        - graph.output 中的旧 shape
    """
    del model.graph.value_info[:]

    for output in model.graph.output:
        tensor_type = output.type.tensor_type

        if tensor_type.HasField("shape"):
            tensor_type.ClearField("shape")


def infer_shapes(model):
    """执行严格的 ONNX shape inference。"""
    return onnx.shape_inference.infer_shapes(
        model,
        check_type=True,
        strict_mode=True,
        data_prop=True,
    )


def change_yolo_batch(
    input_path,
    output_path,
    input_name="images",
    old_batch=1,
    new_batch=4,
):
    """
    将固定 batch=1 的 YOLO ONNX 模型修改为固定 batch=4。

    处理过程：
        1. 对原始模型执行 shape inference；
        2. 获取所有 Reshape 在 batch=1 下的输出 shape；
        3. 找出数据依赖于 images、并且输出第 0 维为 1 的 Reshape；
        4. 将这些 Reshape 的目标 shape 重建为 [4, ...]；
        5. 修改 images 的 batch；
        6. 清除旧 shape；
        7. 重新执行 shape inference；
        8. 检查并保存模型。

    注意：
        本函数用于输入空间尺寸固定、batch 固定的 YOLO 模型。
    """
    print("=" * 70)
    print(f"加载模型: {input_path}")
    print("=" * 70)

    original_model = onnx.load(input_path)

    # 先检查原始模型的基本结构
    onnx.checker.check_model(original_model)
    print("原始模型结构检查通过")

    # ---------------------------------------------------------
    # 1. 对原始 batch=1 模型进行 shape inference
    #
    # 这一步非常重要：
    # 后面不依赖 Reshape 原来的 shape 输入来自哪里，
    # 而是直接读取 Reshape 在 batch=1 下的正确输出 shape。
    # ---------------------------------------------------------
    print()
    print("对原始 batch=1 模型执行 shape inference...")

    try:
        original_inferred = infer_shapes(
            copy.deepcopy(original_model)
        )
    except Exception as error:
        raise RuntimeError(
            "原始 batch=1 模型的 shape inference 已经失败，"
            "无法可靠判断 Reshape 的正确输出 shape。\n"
            f"原始错误：\n{error}"
        ) from error

    print("原始模型 shape inference 成功")

    original_shape_map = get_tensor_shapes(
        original_inferred
    )

    # 后续修改原始模型，而不是修改已经带有大量推导信息的副本
    model = copy.deepcopy(original_model)
    graph = model.graph

    initializer_names = {
        initializer.name
        for initializer in graph.initializer
    }

    real_input_names = {
        value_info.name
        for value_info in graph.input
        if value_info.name not in initializer_names
    }

    if input_name not in real_input_names:
        raise ValueError(
            f"找不到真正的模型输入 '{input_name}'。\n"
            f"当前模型输入为：{sorted(real_input_names)}"
        )

    # ---------------------------------------------------------
    # 2. 建立张量生产者映射
    # ---------------------------------------------------------
    producer_map = {}

    for node in graph.node:
        for output_name in node.output:
            if output_name:
                producer_map[output_name] = node

    # ---------------------------------------------------------
    # 3. 判断一个张量是否依赖模型输入 images
    #
    # 防止误修改完全由 Constant/initializer 产生的辅助张量。
    # ---------------------------------------------------------
    @lru_cache(maxsize=None)
    def depends_on_model_input(tensor_name):
        if tensor_name == input_name:
            return True

        if tensor_name in initializer_names:
            return False

        # 其他模型输入不视为依赖 images
        if tensor_name in real_input_names:
            return False

        producer = producer_map.get(tensor_name)

        if producer is None:
            return False

        if producer.op_type == "Constant":
            return False

        return any(
            depends_on_model_input(node_input)
            for node_input in producer.input
            if node_input
        )

    # ---------------------------------------------------------
    # 4. 找出需要修改的 Reshape
    #
    # 判定条件：
    #   - 数据输入依赖 images；
    #   - 原输出 shape 已知；
    #   - 输出至少二维；
    #   - 输出第 0 维等于旧 batch；
    #   - 其余维度全部为固定值。
    #
    # 对符合条件的 Reshape，直接将目标 shape 替换为：
    #
    #   [new_batch] + 原输出 shape[1:]
    #
    # 因此，无论原 shape 来自：
    #   initializer
    #   Constant
    #   Concat
    #   Shape/Gather/Unsqueeze
    #
    # 都能被统一处理。
    # ---------------------------------------------------------
    used_names = collect_all_tensor_names(model)
    modified_reshape_outputs = []
    modified_count = 0
    skipped_unknown_count = 0

    print()
    print("检查并修复写死 batch 的 Reshape...")

    for node_index, node in enumerate(graph.node):
        if node.op_type != "Reshape":
            continue

        node_name = node.name or f"Reshape_{node_index}"

        if len(node.input) < 2 or not node.output:
            print(f"跳过 {node_name}: 输入或输出不完整")
            continue

        data_input_name = node.input[0]
        output_name = node.output[0]

        # Reshape 的数据必须依赖 images
        if not depends_on_model_input(data_input_name):
            continue

        old_input_shape = original_shape_map.get(
            data_input_name
        )
        old_output_shape = original_shape_map.get(
            output_name
        )

        if old_output_shape is None:
            print(
                f"跳过 {node_name}: "
                f"无法取得原输出 shape"
            )
            skipped_unknown_count += 1
            continue

        # 排除标量和一维 shape 辅助张量
        if len(old_output_shape) < 2:
            continue

        # 只处理原来第 0 维明确等于 old_batch 的情况
        if old_output_shape[0] != old_batch:
            continue

        # 因为最终要生成固定 shape initializer，
        # 所以要求所有维度都是具体数值。
        if any(dim is None for dim in old_output_shape):
            print(
                f"跳过 {node_name}: "
                f"输出包含动态维度 "
                f"{format_shape(old_output_shape)}"
            )
            skipped_unknown_count += 1
            continue

        new_target_shape = list(old_output_shape)
        new_target_shape[0] = new_batch

        new_target_shape = np.asarray(
            new_target_shape,
            dtype=np.int64,
        )

        safe_node_name = (
            node_name
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(".", "_")
        )

        new_shape_name = make_unique_name(
            f"{safe_node_name}_batch{new_batch}_shape",
            used_names,
        )

        new_shape_initializer = numpy_helper.from_array(
            new_target_shape,
            name=new_shape_name,
        )

        graph.initializer.append(new_shape_initializer)

        old_shape_input_name = node.input[1]

        # 让 Reshape 使用新生成的固定 shape
        node.input[1] = new_shape_name

        modified_reshape_outputs.append(
            (node_name, output_name)
        )

        modified_count += 1

        print()
        print(f"修改 Reshape: {node_name}")
        print(
            f"  数据输入 shape: "
            f"{format_shape(old_input_shape)}"
        )
        print(
            f"  原输出 shape:   "
            f"{format_shape(old_output_shape)}"
        )
        print(
            f"  新目标 shape:   "
            f"{new_target_shape.tolist()}"
        )
        print(
            f"  原 shape 输入:  "
            f"{old_shape_input_name}"
        )
        print(
            f"  新 shape 输入:  "
            f"{new_shape_name}"
        )

    if modified_count == 0:
        raise RuntimeError(
            "没有找到需要修改的 Reshape。\n"
            "请确认原始模型输入 batch 是否为 1，"
            "以及原始模型能否推导出 Reshape 输出 shape。"
        )

    # ---------------------------------------------------------
    # 5. 修改模型输入 images 的 batch
    # ---------------------------------------------------------
    print()
    print("修改模型输入 batch...")

    target_input = None

    for model_input in graph.input:
        if model_input.name == input_name:
            target_input = model_input
            break

    tensor_type = target_input.type.tensor_type

    if not tensor_type.HasField("shape"):
        raise ValueError(
            f"输入 '{input_name}' 没有 shape 信息"
        )

    dims = tensor_type.shape.dim

    if len(dims) == 0:
        raise ValueError(
            f"输入 '{input_name}' 是标量，没有 batch 维"
        )

    batch_dim = dims[0]

    if (
        batch_dim.HasField("dim_value")
        and batch_dim.dim_value != old_batch
    ):
        print(
            f"警告：输入当前 batch 为 "
            f"{batch_dim.dim_value}，"
            f"不是预期的 {old_batch}"
        )

    batch_dim.Clear()
    batch_dim.dim_value = new_batch

    input_shape_after_change = []

    for dim in dims:
        if dim.HasField("dim_value"):
            input_shape_after_change.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            input_shape_after_change.append(dim.dim_param)
        else:
            input_shape_after_change.append("?")

    print(
        f"输入 {input_name}: "
        f"batch {old_batch} -> {new_batch}"
    )
    print(
        f"新输入 shape: {input_shape_after_change}"
    )

    # ---------------------------------------------------------
    # 6. 清除 batch=1 时的旧 shape 信息
    #
    # 否则 shape inference 会尝试合并：
    #
    #   新推导：[4, ...]
    #   旧声明：[1, ...]
    #
    # 从而出现 Inferred=4 / Declared=1。
    # ---------------------------------------------------------
    clear_old_shape_information(model)

    # ---------------------------------------------------------
    # 7. 重新进行 shape inference
    # ---------------------------------------------------------
    print()
    print("对 batch=4 模型重新执行 shape inference...")

    try:
        inferred_model = infer_shapes(model)
    except Exception as error:
        output_base, output_ext = os.path.splitext(output_path)

        if not output_ext:
            output_ext = ".onnx"

        debug_path = (
            f"{output_base}_before_shape_inference"
            f"{output_ext}"
        )

        onnx.save(model, debug_path)

        raise RuntimeError(
            "batch=4 模型的 shape inference 失败。\n"
            f"修改后、推导前的调试模型已保存到："
            f"{debug_path}\n"
            f"原始错误：\n{error}"
        ) from error

    print("batch=4 模型 shape inference 成功")

    # ---------------------------------------------------------
    # 8. 验证修改过的 Reshape
    # ---------------------------------------------------------
    final_shape_map = get_tensor_shapes(inferred_model)

    validation_failed = False

    print()
    print("验证修改后的 Reshape 输出 shape...")

    for node_name, output_name in modified_reshape_outputs:
        final_shape = final_shape_map.get(output_name)

        if final_shape is None:
            print(
                f"[警告] {node_name}: "
                f"最终输出 shape 未知"
            )
            continue

        if len(final_shape) == 0:
            print(
                f"[错误] {node_name}: "
                f"最终输出是标量"
            )
            validation_failed = True
            continue

        if final_shape[0] != new_batch:
            print(
                f"[错误] {node_name}: "
                f"最终 shape={format_shape(final_shape)}"
            )
            validation_failed = True
            continue

        print(
            f"[通过] {node_name}: "
            f"{format_shape(final_shape)}"
        )

    if validation_failed:
        raise RuntimeError(
            "部分 Reshape 修改后仍未得到预期 batch，"
            "停止保存最终模型。"
        )

    # ---------------------------------------------------------
    # 9. ONNX checker
    # ---------------------------------------------------------
    print()
    print("执行 ONNX checker...")

    onnx.checker.check_model(inferred_model)

    print("ONNX checker 检查通过")

    # ---------------------------------------------------------
    # 10. 打印模型输出 shape
    # ---------------------------------------------------------
    print()
    print("模型输出：")

    for output in inferred_model.graph.output:
        output_shape = final_shape_map.get(output.name)

        print(
            f"  {output.name}: "
            f"{format_shape(output_shape)}"
        )

    # ---------------------------------------------------------
    # 11. 保存最终模型
    # ---------------------------------------------------------
    onnx.save(inferred_model, output_path)

    print()
    print("=" * 70)
    print("修改完成")
    print(f"修改 Reshape 数量: {modified_count}")
    print(
        f"因动态 shape 跳过数量: "
        f"{skipped_unknown_count}"
    )
    print(f"输出模型: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    change_yolo_batch(
        input_path="yolo11n.onnx",
        output_path="yolo11n_batch4.onnx",
        input_name="images",
        old_batch=1,
        new_batch=4,
    )