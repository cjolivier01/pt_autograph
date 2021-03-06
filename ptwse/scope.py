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

import contextlib

_FAS_SUPPORTED = False

try:
  import ptwse._PTWSE
  # These are basically direct calls into 
  # pytorch_scope.h
  from ptwse._PTWSE import (
    _ptwse_add_frontend_attribute,
    _ptwse_add_frontend_attributes,
    _ptwse_remove_frontend_attribute,
    _ptwse_remove_frontend_attributes,
  )
  _FAS_SUPPORTED = True
except:
  pass


@contextlib.contextmanager
def ir_scope(scope_text, enabled=True and _FAS_SUPPORTED):
  """
  Yields:
      None.
  """
  if enabled:
    ptwse._PTWSE._ptwse_push_ir_scope(scope_text)
  try:
    yield
  finally:
    if enabled:
      ptwse._PTWSE._ptwse_pop_ir_scope()


def get_matched_op_tag(bwd=False):
  if bwd:
    return f'MATCHED_OP.BWD'
  else:
    return f'MATCHED_OP.FWD'


@contextlib.contextmanager
def frontend_attribute_scope(key=None, value=None, values=None, enabled=_FAS_SUPPORTED):
  """
  Set frontend attributes in the lowering ops for the given scope
  Yields: the (possibly mutated) keys, if any
      scope level.
  """
  emit_key = None
  emit_keys = None
  if enabled:
    if values:
      assert not key and not value
      emit_keys = _ptwse_add_frontend_attributes(
        key,
        value
      )
    else:
      emit_key = _ptwse_add_frontend_attribute(
        key,
        value
      )
  try:
    yield emit_keys or emit_key
  finally:
    if enabled:
      if values:
        assert emit_keys
        _ptwse_remove_frontend_attributes(
          emit_keys
        )
      else:
        assert emit_key
        _ptwse_remove_frontend_attribute(
          emit_key
        )
