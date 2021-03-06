"""
System to hook the creation of external python classes or the calling of
external python functions without the necessity of decorating those classes
or function s(i.e. they may reside in an external, read-only library)

It's relatively common to wrap a function in python, via
the wrapt package and other similar pavckages.  Normally, to do this,
a function is wrapped at its place of definition via a decorator (excluding
the situation of runtime-overriding, which this class means to simplify).
The @external_hook decorator strives to easily "replace" one class with
another, decorating the actual class or function at its place of definition.
The desire is such that when code attempts to instantiate the "other" (hooked,
external) class or call the external function, the actual code-path either
creates the new class instead, or replaces all of
the class' member functions with the new class' member functions
(most commonly a derivative of the "hooked" class, thus only shared
non-internal functions are overriden, which a couple of
exceptions -- such as __init__).  The latter is necessary
for situations where the actual location of module is somewhat "fuzzy",
in that it may already be wrapped (i.e. deprecation wrapper) or reassigned to
a different module, and thus resolving its actual instantiation
source is not trivial, or impossible.

"""
import sys
import types
import functools
import inspect
import logging
from contextlib import contextmanager


def full_type_name(obj):
  """
    Return the full type name of a class, including full module location
    """
  if isinstance(obj, type):
    clazz = obj
  else:
    clazz = obj.__class__
  module = clazz.__module__
  if module is None or module == str.__class__.__module__:
    return clazz.__name__
  return module + '.' + clazz.__name__


class _HookedMemberManager(object):
  """
    Manages the actual hooking and unhooking of a class' member functions
    """

  def __init__(
      self,
      hooked_class,
      replacement_class,
      super_function_prefix="__super",
      copy_all_to_hooked_class=True,
  ):
    """
        Initialize _HookedMemberManager class
        Note: Currently does not hook @properties, but could be added easil
        if necessary
        @param hooked_class: The external class that's being hooked
        @param replacement_class: The class which has the functionality to
        replace the hooked class
        @param super_function_prefix: The original (overriden) functions in
        the hooked class will be re-added with the same function name,
        prefixed by this string
        @param copy_all_to_hooked_class: Copy all functions in the hooked
        class to the replacement class.  This is useful for proxy behavior,
        or if the hooked class has already undergone some reflection that
        makes basic derivative behavior not match what is desired
        """
    super().__init__()
    assert inspect.isclass(hooked_class)
    assert inspect.isclass(replacement_class)
    self._hooked_class = hooked_class
    self._source_attributes = inspect.classify_class_attrs(self._hooked_class)
    self._replacement_class = replacement_class
    self._super_function_prefix = super_function_prefix
    self._copy_all_to_hooked_class = copy_all_to_hooked_class
    self._hooked_members = dict()
    self._shared_new_members = dict()
    self._super_members = dict()

    if full_type_name(hooked_class) == full_type_name(replacement_class):
      raise ValueError(f"Hooked and replace classes are the same: "
                       f"{full_type_name(hooked_class)}")

  def __del__(self):
    # Intentionally left blank in order to
    # discourage adding a call to uninstall() here,
    # since the creating module going out of scope should
    # not affect the behavior of existing classes in flight
    pass

  def _is_static(self, member_name):
    for attr in self._source_attributes:
      if attr.name == member_name:
        return attr.kind == 'static_method'
      break
    return False

  def _is_hookable(self, obj):
    if callable(obj):
      # if self._is_static(obj.__name__):
      #     return False
      if obj.__name__ == '__init__':
        return True
      if obj.__name__.startswith("__") and obj.__name__.endswith("__"):
        return False
      if isinstance(obj, types.BuiltinFunctionType):
        return False
      return True
    return False

  def install_hooks(self):
    """
        Install all class-member hooks
        """
    # Get all non-builtin functions in the hooked class

    hooked_members = inspect.getmembers(
        self._hooked_class,
        predicate=self._is_hookable,
    )
    for hm in hooked_members:
      name = hm[0]
      if name != '__class__':
        fn = getattr(
            self._hooked_class,
            name,
        )
        self._hooked_members[name] = fn

    # Get all non-builtin functions in the replacement class
    replacement_members = inspect.getmembers(
        self._replacement_class,
        predicate=self._is_hookable,
    )

    # Create a dict of functions which exist in both classes
    self._shared_new_members.clear()
    for rm in replacement_members:
      name = rm[0]
      if self._copy_all_to_hooked_class or name in self._hooked_members:
        obj = rm[1]
        self._shared_new_members[name] = obj

    # If the replacement class is a derivative, add a function for the
    # hooked class' orignal function with a modified name
    if (issubclass(self._replacement_class, self._hooked_class) and
        self._super_function_prefix):
      for shared_name in self._shared_new_members:
        if shared_name in self._hooked_members:
          padding = "" if shared_name.startswith("__") else "_"
          new_name = (f"{self._super_function_prefix}"
                      f"{padding}{shared_name}")
          if hasattr(self._replacement_class, new_name):
            raise ValueError(
                f"Replacement class already has an attribute named "
                f"{new_name}.  Please choose a different "
                f"super-function prefix")
          if self._copy_all_to_hooked_class:
            # Also copy to the hooked class, since we may be
            # using that instead
            assert not hasattr(self._hooked_class, new_name)
            setattr(
                self._hooked_class,
                new_name,
                self._hooked_members[shared_name],
            )
          setattr(
              self._replacement_class,
              new_name,
              self._hooked_members[shared_name],
          )
          self._super_members[new_name] = (self._hooked_members[shared_name])

    # Replace functions in hooked class with the corresponding function
    # in the replacement class
    for override in self._shared_new_members.items():
      name = override[0]
      if name != '__class__':
        setattr(self._hooked_class, name, override[1])

  def uninstall_hooks(self):
    """
        Uninstall all hooks
        """
    for override in self._hooked_members.items():
      setattr(self._hooked_class, override[0], override[1])
    # Kill a any super members that we created
    for super_name in self._super_members:
      delattr(self._replacement_class, super_name)
      if self._copy_all_to_hooked_class:
        delattr(self._hooked_class, super_name)

  def get_hooked(self, member_name):
    """
        Return the original member function which was replaced via function
        hooking (i.e. in order to chain a call to the original superclass
        function)
        """
    return self._hooked_members[member_name]


class _GroupManager(object):
  """
    Manage one or more groups of hooked functions/classes
    Each group may have multiple hooked target->replacement pairs
    A target->replacement pair may appear in multiple groups
    """

  def __init__(self):
    self._groups = dict()

  @staticmethod
  def _as_list(x):
    if not x:
      return []
    if isinstance(x, list):
      return x
    return [x]

  def add(self, groups, hooked_name, replacement_name, hook_class_object):
    """
        Add the given hooked/replacement pairs form the given group(s)
        """
    assert groups
    name = (hooked_name, replacement_name)
    groups = self._as_list(groups)
    for g in groups:
      if g not in self._groups:
        self._groups[g] = dict()
      else:
        assert name not in self._groups
      self._groups[g][name] = hook_class_object

  def remove(
      self,
      groups,
      hooked_name,
      replacement_name,
  ):
    """
        Remove the given hooked/replacement pairs form the given group(s)
        """
    assert groups
    name = (hooked_name, replacement_name)
    groups = self._as_list(groups)
    for g in groups:
      if g not in self._groups:
        raise ValueError(f"Group not found: {g}")
      if name not in self._groups[g]:
        raise ValueError(f"{name} not found in group: {g}")
      del self._groups[g][name]
      if not self._groups:
        del self._groups[g]

  def visit_group_items(self, groups, fn):
    """
        Perform a callback for all items in the given group(s)
        Only one call is executed per item, even if the item is in
        more than one group
        """
    seen_items = set()
    groups = self._as_list(groups)
    for g in groups:
      for item in self._groups[g].items():
        if item[0] not in seen_items:
          seen_items.add(item[0])
          fn(item[1])


_GROUP_MANAGER = _GroupManager()


class external_hook:
  """
    Decorator used to replace a global class or function with the decorated
    class or function.  The target class or function need not be decorated,
    although the replacement decorated class/function's module must have been
    imported
    """

  def __init__(
      self,
      target,
      member_override=False,
      groups=None,
      enabled=True,
      is_wrapper_class=False,
  ):
    """
        Initialize external_hook decorator/object
        @param target: The function or class to hook/intercept. This can
        either be a string or the class type.  The one to choose may vary
        based upon whether the target has been wrapped or can be immediately
        imported
        @param member_override: Whether to override the class member
        functions (if a class)
        @param groups: Groups for this override behavior to belong to.  This
        is useful in order to enable/disable groups of hooked objects
        together (i.e. with enable_external_hook())
        @param enabled: Whether to enable on startup (and presumably
        permanently).  This is discouraged.
        @param is_wrapper_class: Whether to drill down one inheritance level
        to find the "replacement class".  One use-case for this is when you
        wish to apply a hook only when a particular module is loaded (such as
        the actual Cerebras runtime rather than simply a class in the model
        zoo, which is intended to function without hooking "in the wild")
        """
    if not target:
      raise ValueError("No valid object name to external_hook was supplied")

    target_name = external_hook._make_name(target)

    module_and_attr = external_hook._parse_name(target_name)
    if module_and_attr is None or len(module_and_attr) != 2:
      raise ValueError(f"Please supply a fully qualified class name "
                       f"to external_hook: \"{target_name}\" is not sufficient")
    self._hooked_module_name = module_and_attr[0]
    self._hooked_item_name = module_and_attr[1]

    module = sys.modules.get(self._hooked_module_name)

    if not module:
      raise ValueError(f"Module not found: {self._hooked_module_name}")

    if not hasattr(module, self._hooked_item_name):
      raise ValueError(f"Object {self._hooked_item_name} not found in "
                       "module: {self._hooked_module_name}")
    self._hooked_item = getattr(module, self._hooked_item_name)

    self._is_class = inspect.isclass(self._hooked_item)
    self._replacement_item = None
    self._groups = groups
    self._start_enabled = enabled
    self._enabled_count = 0
    self._member_override = member_override
    self._is_wrapper_class = is_wrapper_class

  def __call__(self, func):
    self._replacement_item = func

    if self._is_wrapper_class:
      if not self._is_class:
        raise ValueError(
            f"Attribute is_wrapper_class evaluated to True, however "
            f"the object supplied to replace is not a class: "
            f"\"{self._replacement_item}\"")
      self._replacement_item = func.__bases__[0]

    module = sys.modules[self._hooked_module_name]

    # Class version
    if self._is_class:  # and self._member_override:
      if self._member_override:
        # Let _HookedMemberManager take care of __init__
        setattr(
            self._hooked_item, "_class_hook",
            _HookedMemberManager(
                self._hooked_item,
                self._replacement_item,
            ))
      else:
        pass
        # Hook __init__
        hooked_init_fn = getattr(self._hooked_item, "__init__")
        # pylint: disable=
        if hasattr(self._replacement_item, "__super__init__"):
          raise ValueError(f"Class was already hooked elsewhere")
        setattr(
            self._replacement_item,
            "__super__init__",
            hooked_init_fn,
        )

      if self._start_enabled:
        self.enable()
      self._add_to_group_manager()
      return self._replacement_item

    # Function version
    @functools.wraps(func)
    def _function_wrapper(*args, **kwargs):
      if self._enabled_count <= 0:
        return self._hooked_item(*args, **kwargs)
      return func(self._hooked_item, *args, **kwargs)

    # sanity check
    target_fn_name = self._hooked_item_name
    if not hasattr(module, target_fn_name):
      raise ValueError(f"Could not find object: {target_fn_name}")
    setattr(
        module,
        target_fn_name,
        _function_wrapper,
    )

    if self._start_enabled:
      self.enable()
    self._add_to_group_manager()
    return _function_wrapper

  def enable(self, allow_forced_disabled=False):
    """
        Enable hooking of this object's class/function
        """
    if self._enabled_count < 0 and not allow_forced_disabled:
      # pylint: disable=logging-format-interpolation
      logging.warning(f"Skipping enabling of disabled hook: "
                      f"{full_type_name(self._hooked_item)}->"
                      f"{full_type_name(self._replacement_item)}")
      return

    if self._enabled_count == 0 and self._is_class:
      try:
        if self._member_override:
          class_hook = getattr(self._hooked_item, "_class_hook")
          class_hook.install_hooks()
        else:
          module = sys.modules[self._hooked_module_name]
          setattr(
              module,
              self._hooked_item_name,
              self._replacement_item,
          )
      except AttributeError:
        pass
    self._enabled_count += 1

  def disable(self, allow_forced_disabled=False):
    """
        Disable hooking of this object's class/function
        """
    if self._enabled_count < 0 and not allow_forced_disabled:
      return
    if self._enabled_count > 0 and allow_forced_disabled:
      raise ValueError(
          f"allow_forced_disabled is not permitted in this context "
          f"since the hooks are already enabled")
    if self._enabled_count == 1 and self._is_class:
      try:
        if self._member_override:
          class_hook = getattr(self._hooked_item, "_class_hook")
          class_hook.uninstall_hooks()
        else:
          module = sys.modules[self._hooked_module_name]
          setattr(
              module,
              self._hooked_item_name,
              self._hooked_item,
          )
      except AttributeError:
        pass
    self._enabled_count -= 1

  def is_enabled(self):
    """
        Return whether this object's class/function's hooking is enabled
        """
    return self._enabled_count > 0

  @staticmethod
  def _make_name(target):
    if isinstance(target, type):
      target_name = full_type_name(target)
    elif callable(target):
      if hasattr(target, "__module__"):
        target_name = target.__module__ + "." + target.__name__
      else:
        # i.e. a method-wrapper object
        target_name = target.__name__
    else:
      target_name = target
    return str(target_name)

  @staticmethod
  def _parse_name(name):
    items = name.split('.')
    if not items:
      return None
    if len(items) == 1:
      return (None, name)
    item_name = items[-1]
    del items[-1]
    return (".".join(items), item_name)

  def _add_to_group_manager(self):
    groups = self._groups
    if not groups:
      groups = ['default']
    _GROUP_MANAGER.add(
        groups=groups,
        hooked_name=self._make_name(self._hooked_item),
        replacement_name=self._make_name(self._replacement_item),
        hook_class_object=self,
    )


@contextmanager
def enable_external_hook(groups='default'):
  """Enters a context where hooks for the given group(s) are enabled

    Yields:
        None.
    """
  _GROUP_MANAGER.visit_group_items(
      groups=groups,
      fn=lambda hk: hk.enable(),
  )
  try:
    yield
  finally:
    _GROUP_MANAGER.visit_group_items(
        groups=groups,
        fn=lambda hk: hk.disable(),
    )


@contextmanager
def disable_external_hook(groups='default'):
  """Enters a context where hooks for the given group(s) are disabled

    Yields:
        None.
    """

  _GROUP_MANAGER.visit_group_items(
      groups=groups,
      fn=lambda hk: hk.disable(allow_forced_disabled=True),
  )
  try:
    yield
  finally:
    _GROUP_MANAGER.visit_group_items(
        groups=groups,
        fn=lambda hk: hk.enable(allow_forced_disabled=True),
    )
