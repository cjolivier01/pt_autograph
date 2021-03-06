# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities used in autograph-generated code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from ptwse.framework import tensor_util


def is_tensor(*args):
  """Check if any arguments are tensors.

  Args:
    *args: Python objects that may or may not be tensors.

  Returns:
    True if any *args are TensorFlow types, False if none are.
  """
  return any(tensor_util.is_tensor(a) for a in args)
