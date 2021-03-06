# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This module defines tensor utilities not found in TensorFlow.

The reason these utilities are not defined in TensorFlow is because they may
not be not fully robust, although they work in the vast majority of cases. So
we define them here in order for their behavior to be consistently verified.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

# from ptwse.framework import dtypes
# from ptwse.framework import sparse_tensor
# from ptwse.framework import tensor_util
# from ptwse.ops import tensor_array_ops


def is_dense_tensor(t):
  if not torch.is_tensor(t):
    return False
  #return not torch.is_sparse
  return True


def is_tensor_array(t):
  #assert False
  #return isinstance(t, tensor_array_ops.TensorArray)
  return False


def is_tensor_list(t):
  # TODO(mdan): This is just a heuristic.
  # With TF lacking support for templated types, this is unfortunately the
  # closest we can get right now. A dedicated op ought to be possible to
  # construct.
  return False
  # assert False
  # return (tensor_util.is_tensor(t) and t.dtype == dtypes.variant and
  #         not t.shape.ndims)


def is_range_tensor(t):
  """Returns True if a tensor is the result of a tf.range op. Best effort."""
  return False
  #return tensor_util.is_tensor(t) and hasattr(t, 'op') and t.op.type == 'Range'


# def is_dense_tensor(t):
#   # TODO(mdan): Resolve this inconsistency.
#   return (tensor_util.is_tensor(t) and
#           not isinstance(t, sparse_tensor.SparseTensor))
#
#
# def is_tensor_array(t):
#   return isinstance(t, tensor_array_ops.TensorArray)
#
#
# def is_tensor_list(t):
#   # TODO(mdan): This is just a heuristic.
#   # With TF lacking support for templated types, this is unfortunately the
#   # closest we can get right now. A dedicated op ought to be possible to
#   # construct.
#   return (tensor_util.is_tensor(t) and t.dtype == dtypes.variant and
#           not t.shape.ndims)
#
#
# def is_range_tensor(t):
#   """Returns True if a tensor is the result of a tf.range op. Best effort."""
#   return tensor_util.is_tensor(t) and hasattr(t, 'op') and t.op.type == 'Range'
