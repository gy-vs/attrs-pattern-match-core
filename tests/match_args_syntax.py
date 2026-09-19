def match_simple(obj, cls):
    match obj:
        case cls(first, second):
            return first, second


def match_alias(obj, cls):
    match obj:
        case cls(value):
            return value


def match_mangled_alias(obj, cls):
    match obj:
        case cls(first, second):
            return first, second


def match_excluded(obj, cls):
    match obj:
        case cls(positional, keyword_only=keyword_only, excluded=excluded):
            return positional, keyword_only, excluded


def match_inheritance(obj, cls):
    match obj:
        case cls(a, b, c, d):
            return a, b, c, d


def match_kw_only_child_preserves_parent_position(obj, cls):
    match obj:
        case cls(x, y):
            return x, y


def match_overrides(obj, cls):
    match obj:
        case cls(c, a=a, b=b):
            return c, a, b
