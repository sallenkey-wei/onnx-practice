import onnx
import argparse
from onnx import helper, numpy_helper

def add_relu(
    input_model: str,
    output_model: str,
    tensor_name: str
):
    model = onnx.load(input_model)
    graph = model.graph

    relu_output = tensor_name + "_relu"

    # 防止名字冲突
    existing_names = set()

    for node in graph.node:
        existing_names.update(node.input)
        existing_names.update(node.output)

    if relu_output in existing_names:
        raise ValueError(
            f"Tensor name '{relu_output}' already exists"
        )

    # ============================================================
    # 情况1：
    # tensor_name 是 graph input
    #
    # images -> Conv
    #
    # 修改为：
    #
    # images -> Relu -> images_relu -> Conv
    # ============================================================

    graph_input = None

    for value in graph.input:
        if value.name == tensor_name:
            graph_input = value
            break

    if graph_input is not None:

        for node_index, node in enumerate(graph.node):

            if tensor_name in node.input:

                relu_node = helper.make_node(
                    "Relu",
                    inputs=[tensor_name],
                    outputs=[relu_output],
                    name=f"{tensor_name}_Relu"
                )

                # 修改消费者
                for i, input_name in enumerate(node.input):
                    if input_name == tensor_name:
                        node.input[i] = relu_output

                # Relu 必须在消费者之前
                graph.node.insert(node_index, relu_node)

                break

        else:
            raise ValueError(
                f"No node consumes graph input '{tensor_name}'"
            )

    # ============================================================
    # 情况2：
    # tensor_name 是某个 node 的 output
    #
    # Conv -> images -> Node
    #
    # 修改为：
    #
    # Conv -> images_relu -> Relu -> images -> Node
    # ============================================================

    else:

        producer_index = None

        for i, node in enumerate(graph.node):
            if tensor_name in node.output:
                producer_index = i
                break

        if producer_index is None:
            raise ValueError(
                f"Cannot find tensor '{tensor_name}' "
                f"in graph input or node output"
            )

        # 找到 tensor 的所有消费者
        consumers = []

        for node in graph.node:
            if tensor_name in node.input:
                consumers.append(node)

        '''
        if not consumers:
            raise ValueError(
                f"Tensor '{tensor_name}' has no consumers"
            )
            '''

        # 原 producer 输出改名
        producer = graph.node[producer_index]

        for i, output_name in enumerate(producer.output):
            if output_name == tensor_name:
                producer.output[i] = relu_output

        # Relu：
        #
        # images_relu -> Relu -> images
        #
        relu_node = helper.make_node(
            "Relu",
            inputs=[relu_output],
            outputs=[tensor_name],
            name=f"{tensor_name}_Relu"
        )

        # Relu 放到 producer 后面
        graph.node.insert(producer_index + 1, relu_node)

    onnx.checker.check_model(model)

    onnx.save(model, output_model)

    print(f"Saved model to: {output_model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='add relu node'
    )

    parser.add_argument('--input', required=True, help='input model')
    parser.add_argument('--output', required=True, help='output model')
    parser.add_argument('--position', required=True, help='add relu node before the tensor name')
    args = parser.parse_args()

    add_relu(args.input, args.output, args.position)
