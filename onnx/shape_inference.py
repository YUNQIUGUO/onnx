"""onnx shape inference. Shape inference is not guaranteed to be
complete.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
import onnx.onnx_cpp2py_export.shape_inference as C
from onnx import ModelProto
from six import string_types

"""Apply shape inference to the provided ModelProto.

Inferred shapes are added to the value_info field of the graph.

If the inferred values conflict with values already provided in the
graph, that means that the provided values are invalid (or there is a
bug in shape inference), and the result is unspecified.

Arguments:
    input (ModelProto,bool): ModelProto

Return:
    return (ModelProto) model with inferred shape information
"""


def infer_shapes(model, check_type=False):  # type: (ModelProto,bool) -> ModelProto
    
    if isinstance(model, ModelProto):
        model_str = model.SerializeToString()
        inferred_model_str = C.infer_shapes(model_str, check_type)
        return onnx.load_from_string(inferred_model_str)
    # If using model_path for infer_shapes
    # directly save the inferred model into the original path, return nothing
    elif isinstance(model, string_types):
        C.infer_shapes_path(model, check_type)
        return None
    else:
        raise TypeError('Shape inference only accepts ModelProto and String, '
                         'incorrect type: {}'.format(type(model)))
