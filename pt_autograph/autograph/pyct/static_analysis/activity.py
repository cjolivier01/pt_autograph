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
"""Activity analysis.

Requires qualified name annotations (see qual_names.py).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import weakref

import gast
import six

from pt_autograph.autograph.pyct import anno
from pt_autograph.autograph.pyct import qual_names
from pt_autograph.autograph.pyct import transformer
from pt_autograph.autograph.pyct.static_analysis.annos import NodeAnno


class Scope(object):
  """Encloses local symbol definition and usage information.

  This can track for instance whether a symbol is modified in the current scope.
  Note that scopes do not necessarily align with Python's scopes. For example,
  the body of an if statement may be considered a separate scope.

  Caution - the AST references held by this object are weak.

  Scope objects are mutable during construction only, and must be frozen using
  `Scope.finalize()` before use. Furthermore, a scope is consistent only after
  all its children have been frozen. While analysing code blocks, scopes are
  being gradually built, from the innermost scope outward. Freezing indicates
  that the analysis of a code block is complete. Once frozen, mutation is no
  longer allowed. `is_final` tracks whether the scope is frozen or not. Certain
  properties, like `referenced`, are only accurate when called on frozen scopes.

  Attributes:
    parent: Optional[Scope], the parent scope, if any.
    isolated: bool, whether the scope is a true Python scope (e.g. the scope of
      a function), or just a surrogate tracking an ordinary code block. Using
      the terminology of the Python 3 reference documentation, True roughly
      represents an actual scope, whereas False represents an ordinary code
      block.
    isolated_names: Set[qual_names.QN], identifiers that are isolated to this
      scope (even if the scope is not isolated).
    read: Set[qual_names.QN], identifiers read in this scope.
    modified: Set[qual_names.QN], identifiers modified in this scope.
    deleted: Set[qual_names.QN], identifiers deleted in this scope.
    bound: Set[qual_names.QN], names that are bound to this scope. See
      https://docs.python.org/3/reference/executionmodel.html#binding-of-names
      for a precise definition.
    globals: Set[qual_names.QN], names that are explicitly marked as global in
      this scope. Note that this doesn't include free read-only vars bound to
      global symbols.
    free_vars: Set[qual_names.QN], the free variables in this scope. See
      https://docs.python.org/3/reference/executionmodel.html for a precise
      definition.
    params: WeakValueDictionary[qual_names.QN, ast.Node], function arguments
      visible in this scope, mapped to the function node that defines them.
    enclosing_scope: Scope, the innermost isolated scope that is a transitive
      parent of this scope. May be the scope itself.
    referenced: Set[qual_names.QN], the totality of the symbols used by this
      scope and its parents.
    is_final: bool, whether the scope is frozen or not.

  Note - simple statements may never delete and modify a symbol at the same
  time. However, compound ones like if statements can. In that latter case, it's
  undefined whether the symbol is actually modified or deleted upon statement
  exit. Certain analyses like reaching definitions need to be careful about
  this.
  """

  # Note: this mutable-immutable pattern is used because using a builder would
  # have taken a lot more boilerplate.

  def __init__(self, parent, isolated=True):
    """Create a new scope.

    Args:
      parent: A Scope or None.
      isolated: Whether the scope is isolated, that is, whether variables
        modified in this scope should be considered modified in the parent
        scope.
    """
    self.parent = parent
    self.isolated = isolated

    self.isolated_names = set()

    self.read = set()
    self.modified = set()
    self.deleted = set()

    self.bound = set()
    self.globals = set()

    self.params = weakref.WeakValueDictionary()

    # Certain fields can only be accessed after the scope and all its parent
    # scopes have been fully built. This field guards that.
    self.is_final = False

  @property
  def enclosing_scope(self):
    assert self.is_final
    if self.parent is not None and not self.isolated:
      return self.parent
    return self

  @property
  def referenced(self):
    if self.parent is not None:
      return self.read | self.parent.referenced
    return self.read

  @property
  def free_vars(self):
    enclosing_scope = self.enclosing_scope
    return enclosing_scope.read - enclosing_scope.bound

  def copy_from(self, other):
    """Recursively copies the contents of this scope from another scope."""
    assert not self.is_final
    if self.parent is not None:
      assert other.parent is not None
      self.parent.copy_from(other.parent)
    self.isolated_names = copy.copy(other.isolated_names)
    self.modified = copy.copy(other.modified)
    self.read = copy.copy(other.read)
    self.deleted = copy.copy(other.deleted)
    self.bound = copy.copy(other.bound)
    self.params = copy.copy(other.params)

  @classmethod
  def copy_of(cls, other):
    if other.parent is not None:
      assert other.parent is not None
      parent = cls.copy_of(other.parent)
    else:
      parent = None
    new_copy = cls(parent)
    new_copy.copy_from(other)
    return new_copy

  def merge_from(self, other):
    assert not self.is_final
    if self.parent is not None:
      assert other.parent is not None
      self.parent.merge_from(other.parent)
    self.isolated_names.update(other.isolated_names)
    self.read.update(other.read)
    self.modified.update(other.modified)
    self.bound.update(other.deleted)
    self.params.update(other.params)

  def finalize(self):
    """Freezes this scope."""
    assert not self.is_final
    # TODO(mdan): freeze read, modified, bound.
    if self.parent is not None:
      assert not self.parent.is_final
      if not self.isolated:
        self.parent.read.update(self.read - self.isolated_names)
        self.parent.modified.update(self.modified - self.isolated_names)
        self.parent.bound.update(self.bound - self.isolated_names)
        self.parent.globals.update(self.globals)
      else:
        # TODO(mdan): This is not accurate.
        self.parent.read.update(self.read - self.bound)
    self.is_final = True

  def __repr__(self):
    return 'Scope{r=%s, w=%s}' % (tuple(self.read), tuple(self.modified))

  def mark_param(self, name, owner):
    # Assumption: all AST nodes have the same life span. This lets us use
    # a weak reference to mark the connection between a symbol node and the
    # function node whose argument that symbol is.
    self.params[name] = owner


class _Comprehension(object):

  no_root = True

  def __init__(self):
    # TODO(mdan): Consider using an enum.
    self.is_list_comp = False
    self.targets = set()


class _FunctionOrClass(object):

  def __init__(self):
    self.node = None


class ActivityAnalyzer(transformer.Base):
  """Annotates nodes with local scope information.

  See Scope.

  The use of this class requires that qual_names.resolve() has been called on
  the node. This class will ignore nodes have not been
  annotated with their qualified names.
  """

  def __init__(self, context, parent_scope=None):
    super(ActivityAnalyzer, self).__init__(context)
    self.scope = Scope(parent_scope, isolated=True)

    # Note: all these flags crucially rely on the respective nodes are
    # leaves in the AST, that is, they cannot contain other statements.
    self._in_aug_assign = False

  @property
  def _in_constructor(self):
    context = self.state[_FunctionOrClass]
    if context.level > 2:
      innermost = context.stack[-1].node
      parent = context.stack[-2].node
      return (isinstance(parent, gast.ClassDef) and
              (isinstance(innermost, gast.FunctionDef) and
               innermost.name == '__init__'))
    return False

  def _node_sets_self_attribute(self, node):
    if anno.hasanno(node, anno.Basic.QN):
      qn = anno.getanno(node, anno.Basic.QN)
      # TODO(mdan): The 'self' argument is not guaranteed to be called 'self'.
      if qn.has_attr and qn.parent.qn == ('self',):
        return True
    return False

  def _track_symbol(self, node, composite_writes_alter_parent=False):
    # A QN may be missing when we have an attribute (or subscript) on a function
    # call. Example: a().b
    if not anno.hasanno(node, anno.Basic.QN):
      return
    qn = anno.getanno(node, anno.Basic.QN)

    # When inside a comprehension, ignore reads to any of the comprehensions's
    # targets. This includes attributes or slices of those arguments.
    for l in self.state[_Comprehension]:
      if qn in l.targets:
        return
      if qn.owner_set & set(l.targets):
        return

    if isinstance(node.ctx, gast.Store):
      # In comprehensions, modified symbols are the comprehension targets.
      if self.state[_Comprehension].level > 0:
        self.state[_Comprehension].targets.add(qn)
        # List comprehension targets leak in Python 2.
        # For details, see:
        # https://stackoverflow.com/questions/4198906/list-comprehension-rebinds-names-even-after-scope-of-comprehension-is-this-righ
        if not (six.PY2 and self.state[_Comprehension].is_list_comp):
          return

      self.scope.modified.add(qn)
      self.scope.bound.add(qn)
      if qn.is_composite and composite_writes_alter_parent:
        self.scope.modified.add(qn.parent)
      if self._in_aug_assign:
        self.scope.read.add(qn)

    elif isinstance(node.ctx, gast.Load):
      self.scope.read.add(qn)

    elif isinstance(node.ctx, gast.Param):
      self.scope.bound.add(qn)
      self.scope.mark_param(qn, self.state[_FunctionOrClass].node)

    elif isinstance(node.ctx, gast.Del):
      # The read matches the Python semantics - attempting to delete an
      # undefined symbol is illegal.
      self.scope.read.add(qn)
      # Targets of del are considered bound:
      # https://docs.python.org/3/reference/executionmodel.html#binding-of-names
      self.scope.bound.add(qn)
      self.scope.deleted.add(qn)

    else:
      raise ValueError('Unknown context {} for node "{}".'.format(
          type(node.ctx), qn))

  def _enter_scope(self, isolated):
    self.scope = Scope(self.scope, isolated=isolated)

  def _exit_scope(self):
    exited_scope = self.scope
    exited_scope.finalize()
    self.scope = exited_scope.parent
    return exited_scope

  def _exit_and_record_scope(self, node, tag=anno.Static.SCOPE):
    node_scope = self._exit_scope()
    anno.setanno(node, tag, node_scope)
    return node_scope

  def _process_statement(self, node):
    self._enter_scope(False)
    node = self.generic_visit(node)
    self._exit_and_record_scope(node)
    return node

  def visit_Import(self, node):
    return self._process_statement(node)

  def visit_ImportFrom(self, node):
    return self._process_statement(node)

  def visit_Global(self, node):
    for name in node.names:
      self.scope.globals.add(qual_names.QN(name))
    return node

  def visit_Expr(self, node):
    return self._process_statement(node)

  def visit_Raise(self, node):
    return self._process_statement(node)

  def visit_Return(self, node):
    return self._process_statement(node)

  def visit_Assign(self, node):
    return self._process_statement(node)

  def visit_AnnAssign(self, node):
    return self._process_statement(node)

  def visit_AugAssign(self, node):
    # Special rules for AugAssign. Here, the AST only shows the target as
    # written, when it is in fact also read.
    self._enter_scope(False)

    self._in_aug_assign = True
    node.target = self.visit(node.target)
    self._in_aug_assign = False

    node.op = self.visit(node.op)
    node.value = self.visit(node.value)
    self._exit_and_record_scope(node)
    return node

  def visit_Delete(self, node):
    return self._process_statement(node)

  def visit_Name(self, node):
    node = self.generic_visit(node)
    self._track_symbol(node)
    return node

  def visit_alias(self, node):
    node = self.generic_visit(node)

    if node.asname is None:
      # Only the root name is a real symbol operation.
      qn = qual_names.QN(node.name.split('.')[0])
    else:
      qn = qual_names.QN(node.asname)

    self.scope.modified.add(qn)
    self.scope.bound.add(qn)
    return node

  def visit_Attribute(self, node):
    node = self.generic_visit(node)
    if self._in_constructor and self._node_sets_self_attribute(node):
      self._track_symbol(node, composite_writes_alter_parent=True)
    else:
      self._track_symbol(node)
    return node

  def visit_Subscript(self, node):
    node = self.generic_visit(node)
    # Subscript writes (e.g. a[b] = "value") are considered to modify
    # both the element itself (a[b]) and its parent (a).
    self._track_symbol(node)
    return node

  def visit_Print(self, node):
    self._enter_scope(False)
    node.values = self.visit_block(node.values)
    node_scope = self._exit_and_record_scope(node)
    anno.setanno(node, NodeAnno.ARGS_SCOPE, node_scope)
    return node

  def visit_Assert(self, node):
    return self._process_statement(node)

  def visit_Call(self, node):
    self._enter_scope(False)
    node.args = self.visit_block(node.args)
    node.keywords = self.visit_block(node.keywords)
    # TODO(mdan): Account starargs, kwargs
    self._exit_and_record_scope(node, tag=NodeAnno.ARGS_SCOPE)

    node.func = self.visit(node.func)
    return node

  def _process_block_node(self, node, block, scope_name):
    self._enter_scope(False)
    block = self.visit_block(block)
    self._exit_and_record_scope(node, tag=scope_name)
    return node

  def _process_parallel_blocks(self, parent, children):
    # Because the scopes are not isolated, processing any child block
    # modifies the parent state causing the other child blocks to be
    # processed incorrectly. So we need to checkpoint the parent scope so that
    # each child sees the same context.
    before_parent = Scope.copy_of(self.scope)
    after_children = []
    for child, scope_name in children:
      self.scope.copy_from(before_parent)
      parent = self._process_block_node(parent, child, scope_name)
      after_child = Scope.copy_of(self.scope)
      after_children.append(after_child)
    for after_child in after_children:
      self.scope.merge_from(after_child)
    return parent

  def _process_comprehension(self,
                             node,
                             is_list_comp=False,
                             is_dict_comp=False):
    with self.state[_Comprehension] as comprehension_:
      comprehension_.is_list_comp = is_list_comp
      # Note: it's important to visit the generators first to properly account
      # for the variables local to these generators. Example: `x` is local to
      # the expression `z for x in y for z in x`.
      node.generators = self.visit_block(node.generators)
      if is_dict_comp:
        node.key = self.visit(node.key)
        node.value = self.visit(node.value)
      else:
        node.elt = self.visit(node.elt)
      return node

  def visit_comprehension(self, node):
    # It is important to visit children in this order so that the reads to
    # the target name are appropriately ignored.
    node.iter = self.visit(node.iter)
    node.target = self.visit(node.target)
    return self.generic_visit(node)

  def visit_DictComp(self, node):
    return self._process_comprehension(node, is_dict_comp=True)

  def visit_ListComp(self, node):
    return self._process_comprehension(node, is_list_comp=True)

  def visit_SetComp(self, node):
    return self._process_comprehension(node)

  def visit_GeneratorExp(self, node):
    return self._process_comprehension(node)

  def visit_arguments(self, node):
    return self._process_statement(node)

  def visit_ClassDef(self, node):
    with self.state[_FunctionOrClass] as fn:
      fn.node = node
      # The ClassDef node itself has a Scope object that tracks the creation
      # of its name, along with the usage of any decorator accompanying it.
      self._enter_scope(False)
      node.decorator_list = self.visit_block(node.decorator_list)
      self.scope.modified.add(qual_names.QN(node.name))
      self.scope.bound.add(qual_names.QN(node.name))
      node.bases = self.visit_block(node.bases)
      node.keywords = self.visit_block(node.keywords)
      self._exit_and_record_scope(node)

      # A separate Scope tracks the actual class definition.
      self._enter_scope(True)
      node = self.generic_visit(node)
      self._exit_scope()
      return node

  def visit_FunctionDef(self, node):
    with self.state[_FunctionOrClass] as fn:
      fn.node = node
      # The FunctionDef node itself has a Scope object that tracks the creation
      # of its name, along with the usage of any decorator accompanying it.
      self._enter_scope(False)
      node.decorator_list = self.visit_block(node.decorator_list)
      function_name = qual_names.QN(node.name)
      self.scope.modified.add(function_name)
      self.scope.bound.add(function_name)
      self._exit_and_record_scope(node)

      # A separate Scope tracks the actual function definition.
      self._enter_scope(True)
      node.args = self.visit(node.args)

      # Track the body separately. This is for compatibility reasons, it may not
      # be strictly needed.
      self._enter_scope(False)
      node.body = self.visit_block(node.body)
      self._exit_and_record_scope(node, NodeAnno.BODY_SCOPE)

      self._exit_scope()
      return node

  def visit_Lambda(self, node):
    # Lambda nodes are treated in roughly the same way as FunctionDef nodes.
    with self.state[_FunctionOrClass] as fn:
      fn.node = node
      self._enter_scope(True)
      node = self.generic_visit(node)
      self._exit_and_record_scope(node)
      return node

  def visit_With(self, node):
    self._enter_scope(False)
    node = self.generic_visit(node)
    self._exit_and_record_scope(node, NodeAnno.BODY_SCOPE)
    return node

  def visit_withitem(self, node):
    return self._process_statement(node)

  def visit_If(self, node):
    self._enter_scope(False)
    node.test = self.visit(node.test)
    node_scope = self._exit_and_record_scope(node.test)
    anno.setanno(node, NodeAnno.COND_SCOPE, node_scope)

    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node

  def visit_For(self, node):
    self._enter_scope(False)
    node.target = self.visit(node.target)
    node.iter = self.visit(node.iter)
    self._exit_and_record_scope(node.iter)

    self._enter_scope(False)
    self.visit(node.target)
    if anno.hasanno(node, anno.Basic.EXTRA_LOOP_TEST):
      self._process_statement(anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST))
    self._exit_and_record_scope(node, tag=NodeAnno.ITERATE_SCOPE)

    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node

  def visit_While(self, node):
    self._enter_scope(False)
    node.test = self.visit(node.test)
    node_scope = self._exit_and_record_scope(node.test)
    anno.setanno(node, NodeAnno.COND_SCOPE, node_scope)

    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node

  def visit_ExceptHandler(self, node):
    self._enter_scope(False)
    # try/except oddity: as expected, it leaks any names you defined inside the
    # except block, but not the name of the exception variable.
    if node.name is not None:
      self.scope.isolated_names.add(anno.getanno(node.name, anno.Basic.QN))
    node = self.generic_visit(node)
    self._exit_scope()
    return node


def resolve(node, context, parent_scope=None):
  return ActivityAnalyzer(context, parent_scope).visit(node)
