import sys

import pytest

import attr


PY310 = sys.version_info[:2] >= (3, 10)

pytestmark = pytest.mark.skipif(
    not PY310, reason="__match_args__ requires Python 3.10 or later."
)


def test_real_match_statement():
    from tests.match_args_syntax import (
        match_alias,
        match_excluded,
        match_inheritance,
        match_kw_only_child_preserves_parent_position,
        match_mangled_alias,
        match_overrides,
        match_simple,
    )

    @attr.s(slots=True)
    class Classic:
        x = attr.ib()
        y = attr.ib(default=2)

    assert match_simple(Classic(1, 2), Classic) == (1, 2)

    @attr.define
    class Defined:
        x: int
        y: int = 2

    assert match_simple(Defined(1, 2), Defined) == (1, 2)

    @attr.define(slots=False)
    class Private:
        _x: int
        __y: int

    assert match_alias(Private(1, 2), Private) == 1
    assert match_mangled_alias(Private(3, 4), Private) == (3, 4)

    @attr.define
    class Excluded:
        positional: int
        keyword_only: int = attr.field(kw_only=True)
        excluded: int = attr.field(init=False, default=3)

    obj = Excluded(1, keyword_only=2)
    assert match_excluded(obj, Excluded) == (1, 2, 3)

    @attr.define
    class Root:
        a: int
        b: int

    @attr.define
    class Middle(Root):
        c: int

    @attr.define
    class Leaf(Middle):
        d: int
        keyword_only: int = attr.field(kw_only=True, default=5)

    assert match_inheritance(Leaf(1, 2, 3, 4, keyword_only=5), Leaf) == (
        1,
        2,
        3,
        4,
    )

    @attr.define
    class PositionalRoot:
        x: int
        y: int

    @attr.define
    class KeywordOnlyChild(PositionalRoot):
        z: int = attr.field(kw_only=True)

    assert (
        match_kw_only_child_preserves_parent_position(
            KeywordOnlyChild(1, 2, z=3), KeywordOnlyChild
        )
        == (1, 2)
    )

    @attr.s
    class OldPositionalRoot:
        x = attr.ib()
        y = attr.ib()

    @attr.s
    class OldKeywordOnlyChild(OldPositionalRoot):
        z = attr.ib(kw_only=True)

    assert (
        match_kw_only_child_preserves_parent_position(
            OldKeywordOnlyChild(1, 2, z=3), OldKeywordOnlyChild
        )
        == (1, 2)
    )

    @attr.define
    class Overridden(Root):
        a: int = attr.field(kw_only=True)
        b: int = attr.field(init=False, default=3)
        c: int = 4

    obj = Overridden(c=2, a=1)
    assert match_overrides(obj, Overridden) == (2, 1, 3)


@pytest.mark.parametrize("decorator", [attr.s, attr.define])
@pytest.mark.parametrize("slots", [True, False])
def test_match_args_order_and_exclusions(decorator, slots):
    kwargs = {"slots": slots}
    if decorator is attr.s:
        kwargs["collect_by_mro"] = True

    @decorator(**kwargs)
    class Base:
        a = attr.ib()
        b = attr.ib(default=2)

    @decorator(**kwargs)
    class Child(Base):
        c = attr.ib(default=3)
        d = attr.ib(kw_only=True)
        e = attr.ib(init=False, default=5)

    assert Child.__match_args__ == ("a", "b", "c")
    assert "__match_args__" in Child.__dict__


@pytest.mark.parametrize("slots", [True, False])
def test_private_alias_uses_attribute_name(slots):
    @attr.define(slots=slots)
    class Private:
        _x: int
        __y: int

    assert Private.__match_args__ == ("_x", "_Private__y")


def test_multiple_inheritance_final_order():
    @attr.s(collect_by_mro=True)
    class A:
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

    assert D.__match_args__ == ("a", "c", "b", "d")


def test_field_override_changes_position():
    @attr.s(collect_by_mro=True)
    class Base:
        a = attr.ib()
        b = attr.ib()

    @attr.s(collect_by_mro=True)
    class Child(Base):
        a = attr.ib(kw_only=True)
        b = attr.ib(init=False, default=3)
        c = attr.ib(default=4)

    assert Child.__match_args__ == ("c",)


def test_class_level_kw_only_has_no_positional_fields():
    @attr.define(kw_only=True)
    class C:
        x: int
        y: int

    assert C.__match_args__ == ()
    assert C(x=1, y=2).x == 1


def test_init_false_with_default_is_excluded():
    @attr.s
    class C:
        x = attr.ib(init=False, default=3)
        y = attr.ib()

    assert C.__match_args__ == ("y",)
    assert C(1).x == 3


def test_init_false_without_default_is_excluded():
    @attr.define
    class C:
        x: int = attr.field(init=False)
        y: int

    assert C.__match_args__ == ("y",)
    assert C(1).y == 1


def test_class_init_false_still_generates_positional_match_args():
    @attr.define(init=False)
    class C:
        x: int

        def __init__(self, x):
            self.__attrs_init__(x)

    assert C.__match_args__ == ("x",)
    assert C(1).x == 1


def test_disabled_match_args_does_not_set_attribute():
    @attr.s(match_args=False)
    class C:
        x = attr.ib()

    assert "__match_args__" not in C.__dict__
    assert getattr(C, "__match_args__", None) is None


@pytest.mark.parametrize("slots", [True, False])
def test_disabled_match_args_preserves_user_value(slots):
    @attr.define(match_args=False, slots=slots)
    class C:
        x: int
        __match_args__ = ("custom",)

    assert C.__match_args__ == ("custom",)
    assert "__match_args__" in C.__dict__


def test_generated_match_args_on_replaced_slots_class():
    @attr.define(slots=True)
    class C:
        x: int

    assert "__match_args__" in C.__dict__
    assert C.__dict__["__match_args__"] == ("x",)


def test_enabled_match_args_preserves_user_value():
    @attr.define(match_args=True)
    class C:
        x: int
        __match_args__ = ()

    assert C.__match_args__ == ()
    assert "__match_args__" in C.__dict__


def test_make_class_option():
    Enabled = attr.make_class("Enabled", ["a", "b"], slots=True)
    Disabled = attr.make_class("Disabled", ["a", "b"], match_args=False)

    assert Enabled.__dict__["__match_args__"] == ("a", "b")
    assert "__match_args__" not in Disabled.__dict__


def test_inherited_match_args_regenerated_from_final_field_order():
    @attr.define(match_args=False)
    class Base:
        x: int
        __match_args__ = ("custom",)

    @attr.define
    class Child(Base):
        y: int = attr.field(kw_only=True)

    assert Child.__match_args__ == ("x",)

    @attr.define
    class GrandChild(Child):
        z: int

    assert GrandChild.__match_args__ == ("x", "z")


def test_unsupported_runtime_does_not_affect_creation(monkeypatch):
    from attr import _make

    monkeypatch.setattr(_make, "PY310", False)

    @attr.s
    class C:
        x = attr.ib()
        __match_args__ = ("custom",)

    assert C.__match_args__ == ("custom",)
    C(1)
