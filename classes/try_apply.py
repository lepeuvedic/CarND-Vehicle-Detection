def try_apply(f=lambda *x: list(x), default=None, expr=lambda x: x, error=ZeroDivisionError, *args):
    """
    Cleans up a pattern where multiple branches in an expression can trigger the same exception,
    but there is a large combinatorial explosion of which branches fail, although the desired 
    behaviour is just to apply the function f to the remaining branches of expression f(*args).
    If all the branches fail, and a default value has been supplied, it is returned, otherwise
    f is called with no arguments.

    Example: try_apply(lambda *x: min(x), 0, lambda x: l._geom[x]['zmax'], KeyError, key1, key2):
    """
    l = []
    for a in args:
        try:
            l.append(expr(a))
        except error:
            pass
    if l:  return f(*l)
    if not(default is None): return default
    return f()
