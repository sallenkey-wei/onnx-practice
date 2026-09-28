import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
import argparse

# 在yolo11n.onnx 模型的输入前插入DequantizeLinear节点，并将其中一个中间tensor (0/model.2/Concat_output_)加入到graph.output中，方便后续使用onnxruntime推理时获取该tensor的输出
def add_uint8_dequantize_input(
    input: str,
    output: str,
    input_tensor_name: str,
    scale: float,
    zero_point: int,
    ):
    model = onnx.load(input)
    # 在value_info中产生593tensor对应的信息，并找到它
    model = onnx.shape_inference.infer_shapes(model)
    if input_tensor_name is None:
        inputs = list(model.graph.input)
        if len(inputs) != 1:
            raise ValueError('model inputs more than one')
        input_tensor = inputs[0]
    else:
        input_tensor = next(
            (input for input in model.graph.inputs
                if input.name == input_tensor_name
            ),
            None
        )
    tensor_593 = next(
        (
            tensor for tensor in model.graph.value_info
            if tensor.name == '/model.2/Concat_output_0'
        ),
        None
    )
    if tensor_593 is not None:
        print('tensor_593 found in value_info')
        model.graph.output.extend([tensor_593])
    origin_input_name = input_tensor.name
    input_tensor.name = 'my_dequantize_input'
    input_tensor.type.tensor_type.elem_type = TensorProto.UINT8

    scale_initializer = numpy_helper.from_array(
        np.array(scale, dtype=np.float32),
        name='my_scale_dequantize'
    )
    zero_point_initializer = numpy_helper.from_array(
        np.array(zero_point, dtype=np.uint8),
        name='my_zero_point_dequantize'
    )
    model.graph.initializer.extend([scale_initializer, zero_point_initializer])

    dequant_node = helper.make_node(
        op_type = "DequantizeLinear",
        inputs=[
            input_tensor.name,
            'my_scale_dequantize',
            'my_zero_point_dequantize'
        ],
        outputs=[
            origin_input_name
        ],
    )
    model.graph.node.insert(0, dequant_node)
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(
        model,
        strict_mode=True,
    )
    onnx.checker.check_model(model)
    onnx.save(model, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='add dequantize node at the front of model'
    )
    parser.add_argument('--input', required=True, help='input model')
    parser.add_argument('--output', required=True, help='output model')
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0/255.0,
        help='quantization scale'
    )
    parser.add_argument(
        '--zero-point',
        type=int,
        default=0,
        help='quantization zero point'
    )
    parser.add_argument(
        '--dtype',
        choices=['uint8', 'int8', 'float32'],
        default='uint8',
        help='quantization type'
    )
    parser.add_argument(
        '--input_tensor_name',
        default=None,
        help='add dequantize node before this input_tensor_name'
    )
    args = parser.parse_args()
    print("input ", args.input)
    print("output ", args.output)
    print("scale", args.scale)
    print("zero-point", args.zero_point)
    print("dtype", args.dtype)
    print('input_tensor_name', args.input_tensor_name)
    add_uint8_dequantize_input(
        args.input,
        args.output,
        args.input_tensor_name,
        args.scale,
        args.zero_point
    )
