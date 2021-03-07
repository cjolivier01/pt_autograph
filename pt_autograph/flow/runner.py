# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pt_autograph.autograph as ag
from pt_autograph.autograph.core import converter

from pt_autograph.autograph.impl.api import (
    converted_call
)


def maybe_run_converted(
    fn, 
    *args, 
    **kwargs
):
    """
    If any relevant control flow, convert the function
    Ideally, we walk the AST and *only* for functions which contain relevant 
    (tensor-related) control-flow, do we convert.  And use the no-child
    convert option so that we can still debug the rest more easily.
    """
    # Do our lame quick analysis.
    # ADDENDUM: no maybe for the moment...
    
    # Or really, maybe just build the converted and 
    # if any control flow changes were generated,
    # such as if and for, rather than them all being python-based,
    # then and only then, do the substitution so as to maintain
    # debuggability

    result = converted_call(
        fn,
        args=args,
        kwargs=kwargs,
        caller_fn_scope=None,
        options=converter.STANDARD_OPTIONS,
    )
    return result
