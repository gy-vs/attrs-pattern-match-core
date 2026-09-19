"""
Tests for the generation of ``__match_args__`` (structural pattern
matching, PEP 634) and for field *alias* support.

The tests using the actual ``match`` statement live at module scope so
that the whole file parses on CPython 3.10+ (the minimum version that
supports the syntax).  Behavior on older interpreters is emulated by
flipping ``attr._make.PY310``.
"""
from __future__ import absolute_import

import inspect
import sys

import pytest

import attr

from attr._make import _determine_match_args


needs_match_syntax = pytest.mark.skipif(
    sys.version_info[:2] < (3, 10),
    reason="Structural pattern matching requires Python 3.10 or later.",
)


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


class TestMatchArgsGeneration:
    def test_simple(self):
        """
        Positional init fields become __match_args__ in __init__ order.
        """

        @attr.s
        class C(object):
            x = attr.ib()
            y = attr.ib()

        assert C.__match_args__ == ("x", "y")

    def test_define(self):
        @attr.define
        class C(object):
            x: int
            y: int = 2

        assert C.__match_args__ == ("x", "y")

    def test_frozen(self):
        @attr.frozen
        class C(object):
            x: int

        assert C.__match_args__ == ("x",)

    def test_make_class(self):
        C = attr.make_class(
            "C", {"x": attr.ib(), "y": attr.ib(alias="z")}
        )

        assert C.__match_args__ == ("x", "y")
        assert C(1, z=2).y == 2

    def test_kw_only_excluded(self):
        @attr.s
        class C(object):
            x = attr.ib()
            y = attr.ib(kw_only=True)
            z = attr.ib(default=3)

        assert C.__match_args__ == ("x", "z")

    def test_class_level_kw_only_excluded(self):
        @attr.s(kw_only=True)
        class C(object):
            x = attr.ib()
            y = attr.ib()

        assert C.__match_args__ == ()

    def test_init_false_excluded(self):
        @attr.s
        class C(object):
            x = attr.ib()
            y = attr.ib(init=False, default=2)

        assert C.__match_args__ == ("x",)

    def test_define_init_false_excluded(self):
        @attr.define
        class C(object):
            x: int
            y: int = attr.field(init=False, default=2)

        assert C.__match_args__ == ("x",)

    def test_init_false_no_default_excluded(self):
        """
        init=False without a default isn't part of __init__ at all and must
        not appear in __match_args__ either.
        """

        @attr.define
        class C(object):
            x: int = 0
            y: int = attr.field(init=False)

            def __attrs_post_init__(self):
                self.__dict__["y"] = self.x + 1

        assert C.__match_args__ == ("x",)

    def test_repr_eq_have_no_influence(self):
        @attr.s
        class C(object):
            x = attr.ib(repr=False)
            y = attr.ib(eq=False)

        assert C.__match_args__ == ("x", "y")

    def test_order_matches_init_signature(self):
        @attr.s
        class C(object):
            x = attr.ib()
            y = attr.ib(default=2)
            z = attr.ib(kw_only=True, default=3)

        params = [
            p.name
            for p in inspect.signature(C.__init__).parameters.values()
            if p.kind is p.POSITIONAL_OR_KEYWORD and p.name != "self"
        ]
        assert list(C.__match_args__) == params

    def test_class_init_false_still_gets_match_args(self):
        @attr.define(init=False)
        class C(object):
            x: int
            y: int = 2

            def __init__(self, x):
                self.__attrs_init__(x)

        assert C.__match_args__ == ("x", "y")

    def test_empty(self):
        @attr.s
        class C(object):
            pass

        assert C.__match_args__ == ()


# ---------------------------------------------------------------------------
# alias
# ---------------------------------------------------------------------------


class TestAlias:
    def test_alias_used_in_init(self):
        @attr.s
        class C(object):
            x = attr.ib(alias="z")

        assert C(z=1).x == 1
        with pytest.raises(TypeError):
            C(x=1)

    def test_alias_private_attr(self):
        @attr.s
        class C(object):
            _x = attr.ib(alias="_y")

        assert C(_y=1)._x == 1

    def test_alias_does_not_appear_in_match_args(self):
        """
        __match_args__ always names attributes, not __init__ aliases.
        """

        @attr.s
        class C(object):
            x = attr.ib(alias="z")
            _p = attr.ib(alias="_q", default=1)

        assert C.__match_args__ == ("x", "_p")

    def test_underscore_stripping_is_alias_like(self):
        """
        Without an explicit alias, private attrs get their underscore
        stripped in __init__, but __match_args__ keeps the real name.
        """

        @attr.s
        class C(object):
            _x = attr.ib()

        assert C.__match_args__ == ("_x",)
        assert C(x=1)._x == 1

    def test_attribute_alias(self):
        a = attr.fields_dict(
            attr.make_class("C", {"x": attr.ib(alias="z")})
        )["x"]
        assert a.alias == "z"

    def test_alias_defaults_to_none(self):
        a = attr.fields(attr.make_class("C", ["x"]))[0]
        assert a.alias is None

    @pytest.mark.parametrize("bad_alias", [1, b"x", 1.5])
    def test_alias_must_be_string(self, bad_alias):
        with pytest.raises(ValueError, match="must be a string"):
            attr.make_class("C", {"x": attr.ib(alias=bad_alias)})

    @pytest.mark.parametrize("bad_alias", ["1x", "x-y", "a.b", ""])
    def test_alias_must_be_identifier(self, bad_alias):
        with pytest.raises(ValueError, match="valid Python identifier"):
            attr.make_class("C", {"x": attr.ib(alias=bad_alias)})

    def test_public_attr_cannot_have_private_alias(self):
        with pytest.raises(ValueError, match="can't be private"):
            attr.make_class("C", {"x": attr.ib(alias="_y")})

    def test_private_attr_must_have_private_alias(self):
        with pytest.raises(ValueError, match="must be private too"):
            attr.make_class("C", {"_x": attr.ib(alias="y")})

    def test_duplicate_alias(self):
        with pytest.raises(ValueError, match="same __init__ argument"):
            attr.make_class(
                "C",
                {"x": attr.ib(alias="z"), "_z": attr.ib()},
            )

    def test_alias_collides_with_other_alias(self):
        with pytest.raises(ValueError, match="same __init__ argument"):
            attr.make_class(
                "C",
                {
                    "x": attr.ib(alias="z"),
                    "y": attr.ib(alias="z", default=1),
                },
            )

    def test_duplicate_underscore_names_without_alias(self):
        """
        _x and x already collide in __init__; that remains detected.
        """
        with pytest.raises(ValueError, match="same __init__ argument"):
            attr.make_class(
                "C", {"x": attr.ib(), "_x": attr.ib(default=1)}
            )

    def test_alias_evolves_through_field_transformer(self):
        def transformer(cls, fields):
            return [
                a.evolve(alias="z") if a.name == "x" else a for a in fields
            ]

        C = attr.make_class(
            "C",
            {"x": attr.ib(), "y": attr.ib(default=1)},
            field_transformer=transformer,
        )
        assert C(z=1).x == 1
        assert C.__match_args__ == ("x", "y")


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_single_inheritance(self):
        @attr.s
        class A(object):
            a = attr.ib()

        @attr.s
        class B(A):
            b = attr.ib()

        assert B.__match_args__ == ("a", "b")

    def test_multi_level_inheritance(self):
        @attr.define
        class L1(object):
            a: int

        @attr.define
        class L2(L1):
            b: int = 2
            k: int = attr.field(kw_only=True, default=0)

        @attr.define
        class L3(L2):
            c: int = 3

        assert L1.__match_args__ == ("a",)
        assert L2.__match_args__ == ("a", "b")
        assert L3.__match_args__ == ("a", "b", "c")

    def test_new_kw_only_field_does_not_shift_parent_positions(self):
        @attr.define
        class A(object):
            a: int
            b: int = 2

        @attr.define
        class B(A):
            c: int = attr.field(kw_only=True, default=3)

        assert B.__match_args__ == ("a", "b")

    def test_parent_match_args_unchanged_by_subclass(self):
        @attr.define
        class A(object):
            a: int

        @attr.define
        class B(A):
            b: int = attr.field(kw_only=True, default=3)
            c: int = 4

        assert A.__match_args__ == ("a",)
        assert B.__match_args__ == ("a", "c")

    def test_field_override_moves_to_the_end(self):
        """
        Re-defining a parent field in a subclass makes it a "new" field and
        moves it after all inherited positional fields, both in __init__ and
        __match_args__.
        """

        @attr.define
        class A(object):
            a: int
            b: int

        @attr.define
        class B(A):
            a: int = 0

        assert B.__match_args__ == ("b", "a")

    def test_field_override_kw_only_removes_position(self):
        @attr.define
        class A(object):
            a: int
            b: int = 2

        @attr.define
        class B(A):
            a: int = attr.field(kw_only=True, default=0)

        assert B.__match_args__ == ("b",)

    def test_field_override_other_fields_keep_meaning(self):
        @attr.define
        class A(object):
            a: int
            b: int

        @attr.define
        class B(A):
            a: int = 0

        # `b` inherited position stays; overridden `a` moved to the end.
        assert B.__match_args__ == ("b", "a")
        inst = B(10)
        assert inst.b == 10
        assert inst.a == 0

    def test_inherited_init_false(self):
        @attr.define
        class A(object):
            a: int
            b: int = attr.field(init=False, default=2)

        @attr.define
        class B(A):
            c: int = 3

        assert B.__match_args__ == ("a", "c")

    def test_mro_diamond_collect_by_mro(self):
        @attr.s(collect_by_mro=True)
        class A(object):
            a = attr.ib()

        @attr.s(collect_by_mro=True)
        class B(A):
            b = attr.ib()

        @attr.s(collect_by_mro=True)
        class C(A):
            c = attr.ib()

        @attr.s(collect_by_mro=True)
        class D(B, C):
            d = attr.ib()

        expected = tuple(
            p.name
            for p in inspect.signature(D.__init__).parameters.values()
            if p.kind is p.POSITIONAL_OR_KEYWORD and p.name != "self"
        )
        assert D.__match_args__ == expected == ("a", "c", "b", "d")

        # And real matching agrees with the __init__ layout.
        got = None
        match D(1, 2, 3, 4):
            case D(a, c, b, d):
                got = (a, c, b, d)
        assert got == (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# slots
# ---------------------------------------------------------------------------


class TestSlots:
    def test_slots_match_args_on_final_class(self):
        @attr.define(slots=True)
        class C(object):
            x: int
            y: int = 2

        assert C.__match_args__ == ("x", "y")
        assert "__match_args__" in C.__dict__

    def test_slots_kw_only(self):
        @attr.define(slots=True)
        class C(object):
            x: int
            y: int = attr.field(kw_only=True, default=3)

        assert C.__match_args__ == ("x",)
        assert "__match_args__" in C.__dict__

    def test_slots_inheritance(self):
        @attr.define(slots=True)
        class A(object):
            a: int

        @attr.define(slots=True)
        class B(A):
            b: int = 2
            k: int = attr.field(kw_only=True, default=3)

        assert B.__match_args__ == ("a", "b")
        assert "__match_args__" in B.__dict__
        assert A.__dict__["__match_args__"] == ("a",)

    def test_slots_user_defined_kept_on_final_class(self):
        @attr.define(slots=True)
        class C(object):
            x: int
            __match_args__ = ("z",)

        assert C.__match_args__ == ("z",)
        assert C.__dict__["__match_args__"] == ("z",)

    def test_slots_inherited_user_value_is_replaced(self):
        class Base(object):
            __match_args__ = ("q",)

        @attr.define(slots=True)
        class C(Base):
            x: int

        assert C.__match_args__ == ("x",)
        assert C.__dict__["__match_args__"] == ("x",)


# ---------------------------------------------------------------------------
# User-defined __match_args__ and the switch
# ---------------------------------------------------------------------------


class TestUserDefined:
    def test_user_value_kept_by_default(self):
        @attr.s
        class C(object):
            x = attr.ib()
            __match_args__ = ("z",)

        assert C.__match_args__ == ("z",)

    def test_user_value_kept_with_none(self):
        @attr.s(match_args=None)
        class C(object):
            x = attr.ib()
            __match_args__ = ("z",)

        assert C.__match_args__ == ("z",)

    def test_user_value_rejected_when_enabled(self):
        with pytest.raises(ValueError, match="__match_args__ is explicitly"):

            @attr.s(match_args=True)
            class C(object):
                x = attr.ib()
                __match_args__ = ("z",)

    def test_disabled_never_generates(self):
        @attr.s(match_args=False)
        class C(object):
            x = attr.ib()

        assert not hasattr(C, "__match_args__")

    def test_disabled_keeps_user_value(self):
        @attr.s(match_args=False)
        class C(object):
            x = attr.ib()
            __match_args__ = ("z",)

        assert C.__match_args__ == ("z",)

    def test_inherited_value_is_not_own(self):
        """
        __match_args__ inherited from a non-attrs base must not be treated as
        user-defined on the decorated class.
        """

        class Base(object):
            __match_args__ = ("q",)

        @attr.s
        class C(Base):
            x = attr.ib()

        assert C.__match_args__ == ("x",)
        assert "__match_args__" in C.__dict__

    def test_define_user_value_kept_by_default(self):
        @attr.define
        class C(object):
            x: int
            __match_args__ = ("z",)

        assert C.__match_args__ == ("z",)

    def test_define_user_value_rejected_with_true(self):
        with pytest.raises(ValueError, match="__match_args__ is explicitly"):

            @attr.define(match_args=True)
            class C(object):
                x: int
                __match_args__ = ("z",)

    def test_determine_match_args_helper(self):
        @attr.s
        class Made(object):
            x = attr.ib()
            y = attr.ib(kw_only=True)

        attribs = attr.fields(Made)

        # Plain, not-yet-decorated class so a generated value isn't mistaken
        # for a user-defined one.
        class Plain(object):
            pass

        create, value = _determine_match_args(Plain, attribs, None)
        assert create is True
        assert value == ("x",)

        assert _determine_match_args(Plain, attribs, False) == (False, None)

        # Explicit True is honored regardless of the runtime.
        create, value = _determine_match_args(Plain, attribs, True)
        assert create is True
        assert value == ("x",)

        # A user-defined value is kept for None, rejected for True.
        class U(object):
            __match_args__ = ("z",)

        assert _determine_match_args(U, attribs, None) == (False, None)
        with pytest.raises(ValueError, match="explicitly set"):
            _determine_match_args(U, attribs, True)
        assert _determine_match_args(U, attribs, False) == (False, None)


# ---------------------------------------------------------------------------
# Old runtimes / switch off: class creation must keep working
# ---------------------------------------------------------------------------


class TestOldRuntime:
    def test_none_off_when_no_pattern_matching(self, monkeypatch):
        monkeypatch.setattr("attr._make.PY310", False)

        @attr.s
        class C(object):
            x = attr.ib()

        assert not hasattr(C, "__match_args__")
        # The class itself is fully functional.
        assert C(1).x == 1

    def test_none_user_value_survives_without_pattern_matching(
        self, monkeypatch
    ):
        monkeypatch.setattr("attr._make.PY310", False)

        @attr.s
        class C(object):
            x = attr.ib()
            __match_args__ = ("z",)

        assert C.__match_args__ == ("z",)

    def test_explicit_true_works_without_pattern_matching(
        self, monkeypatch
    ):
        monkeypatch.setattr("attr._make.PY310", False)

        @attr.s(match_args=True)
        class C(object):
            x = attr.ib()
            y = attr.ib(kw_only=True, default=2)

        assert C.__match_args__ == ("x",)
        assert C(1).x == 1

    def test_disabled_without_pattern_matching(self, monkeypatch):
        monkeypatch.setattr("attr._make.PY310", False)

        @attr.s(match_args=False)
        class C(object):
            x = attr.ib()

        assert not hasattr(C, "__match_args__")
        assert C(1).x == 1

    def test_slots_without_pattern_matching(self, monkeypatch):
        monkeypatch.setattr("attr._make.PY310", False)

        @attr.define(slots=True)
        class C(object):
            x: int

        assert not hasattr(C, "__match_args__")
        assert C(1).x == 1
        assert not hasattr(C(1), "__dict__")


# ---------------------------------------------------------------------------
# Real match statements
# ---------------------------------------------------------------------------


@needs_match_syntax
class TestRealMatchStatements:
    def test_positional_destructuring(self):
        @attr.s
        class Point(object):
            x = attr.ib()
            y = attr.ib()

        def describe(p):
            match p:
                case Point(0, 0):
                    return "origin"
                case Point(x, 0):
                    return "x-axis %d" % x
                case Point(x, y):
                    return "point %d,%d" % (x, y)

        assert describe(Point(0, 0)) == "origin"
        assert describe(Point(4, 0)) == "x-axis 4"
        assert describe(Point(1, 2)) == "point 1,2"

    def test_kw_only_cannot_be_matched_positionally(self):
        @attr.define
        class C(object):
            x: int
            y: int = attr.field(kw_only=True, default=0)

        matched = []
        match C(1, y=2):
            case C(x) if True:
                matched.append(x)
            case _:
                pass
        assert matched == [1]

        # Only one positional pattern is accepted.
        with pytest.raises(TypeError):
            match C(1):
                case C(x, y):  # noqa: F841
                    pass

    def test_init_false_not_positional(self):
        @attr.define
        class C(object):
            x: int
            y: int = attr.field(init=False, default=2)

        result = None
        match C(1):
            case C(x):
                result = x
        assert result == 1

    def test_alias_attribute_name_used(self):
        @attr.s
        class C(object):
            x = attr.ib(alias="z")

        result = None
        match C(z=42):
            case C(x):
                result = x
        assert result == 42

    def test_private_alias_attribute_name_used(self):
        @attr.s
        class C(object):
            _x = attr.ib(alias="_y")

        result = None
        match C(_y=7):
            case C(_x):
                result = _x
        assert result == 7

    def test_inheritance_destructuring(self):
        @attr.define
        class A(object):
            a: int

        @attr.define
        class B(A):
            b: int = 2
            k: int = attr.field(kw_only=True, default=3)

        @attr.define
        class C(B):
            c: int = 4

        got = None
        match C(10, 20, c=30, k=40):
            case C(a, b, c):
                got = (a, b, c)
        assert got == (10, 20, 30)

    def test_subclass_new_kw_only_does_not_shift(self):
        @attr.define
        class A(object):
            a: int
            b: int = 2

        @attr.define
        class B(A):
            c: int = attr.field(kw_only=True, default=3)

        got = None
        match B(1):
            case B(a, b):
                got = (a, b)
        assert got == (1, 2)

    def test_field_override_position(self):
        @attr.define
        class A(object):
            a: int
            b: int

        @attr.define
        class B(A):
            a: int = 0

        got = None
        match B(5, 6):
            case B(b, a):
                got = (b, a)
        assert got == (5, 6)

    def test_slots_destructuring(self):
        @attr.define(slots=True)
        class C(object):
            x: int
            y: int = 2

        got = None
        match C(3, 4):
            case C(x, y):
                got = (x, y)
        assert got == (3, 4)

    def test_match_base_class_pattern_on_subclass(self):
        @attr.define
        class A(object):
            a: int

        @attr.define
        class B(A):
            b: int = 2

        got = None
        match B(8):
            case A(a):
                got = a
        assert got == 8

    def test_keyword_pattern_still_works(self):
        @attr.define
        class C(object):
            x: int
            y: int = attr.field(kw_only=True, default=0)

        got = None
        match C(1, y=9):
            case C(x=x, y=y):
                got = (x, y)
        assert got == (1, 9)

    def test_frozen_match(self):
        @attr.frozen
        class C(object):
            x: int

        got = None
        match C(5):
            case C(x):
                got = x
        assert got == 5
