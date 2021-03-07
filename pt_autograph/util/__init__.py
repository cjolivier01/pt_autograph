def is_true(value):
  if value is None:
    return False
  value = value.lower()
  if value in {"t", "true", "y", "yes"}:
    return True
  if value in {"f", "false", "n", "no"}:
    return False
  if isinstance(value, str):
    return bool(int(value))
  return not (not value)
