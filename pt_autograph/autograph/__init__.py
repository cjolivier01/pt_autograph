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
"""Conversion of plain Python into TensorFlow graph code.

NOTE: In TensorFlow 2.0, AutoGraph is automatically applied when using
`tf.function`. This module contains lower-level APIs for advanced use.

For more information, see the
[AutoGraph guide](https://www.tensorflow.org/guide/autograph).

By equivalent graph code we mean code that generates a TensorFlow graph when
run. The generated graph has the same effects as the original code when executed
(for example with `tf.function` or `tf.compat.v1.Session.run`). In other words,
using AutoGraph can be thought of as running Python in TensorFlow.
"""
# TODO(b/119833526): Link to the new tf.function + autograph tutorial.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(mdan): Bring only the relevant symbols to the top level.
from pt_autograph.autograph import operators
from pt_autograph.autograph import utils
from pt_autograph.autograph.core.converter import ConversionOptions
from pt_autograph.autograph.core.converter import Feature
from pt_autograph.autograph.impl.api import AutoGraphError
from pt_autograph.autograph.impl.api import convert
from pt_autograph.autograph.impl.api import converted_call
from pt_autograph.autograph.impl.api import do_not_convert
#from pt_autograph.autograph.impl.api import StackTraceMapper
from pt_autograph.autograph.impl.api import to_code
from pt_autograph.autograph.impl.api import to_graph
from pt_autograph.autograph.lang.directives import set_element_type
from pt_autograph.autograph.lang.directives import set_loop_options
from pt_autograph.autograph.lang.special_functions import stack
from pt_autograph.autograph.utils import ag_logging
#from pt_autograph.util.all_util import remove_undocumented

# TODO(mdan): Revisit this list once we finalize the generated code mechanism.
_allowed_symbols = [
    # Main API
    'AutoGraphError',
    'ConversionOptions',
    'Feature',
    #'StackTraceMapper',
    'convert',
    'converted_call',
    'do_not_convert',
    'to_code',
    'to_graph',
    # Overloaded operators
    'operators',
    # Python language "extensions"
    'set_element_type',
    'set_loop_options',
    'stack',
    'tensor_list',
    # Utilities: to be removed
    'utils',
]

#remove_undocumented(__name__, _allowed_symbols)
