# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _swig_decoders
else:
    import _swig_decoders

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _swig_decoders.delete_SwigPyIterator

    def value(self):
        return _swig_decoders.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _swig_decoders.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _swig_decoders.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _swig_decoders.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _swig_decoders.SwigPyIterator_equal(self, x)

    def copy(self):
        return _swig_decoders.SwigPyIterator_copy(self)

    def next(self):
        return _swig_decoders.SwigPyIterator_next(self)

    def __next__(self):
        return _swig_decoders.SwigPyIterator___next__(self)

    def previous(self):
        return _swig_decoders.SwigPyIterator_previous(self)

    def advance(self, n):
        return _swig_decoders.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _swig_decoders.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _swig_decoders.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _swig_decoders.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _swig_decoders.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _swig_decoders.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _swig_decoders.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _swig_decoders:
_swig_decoders.SwigPyIterator_swigregister(SwigPyIterator)

class DoubleVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _swig_decoders.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _swig_decoders.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _swig_decoders.DoubleVector___bool__(self)

    def __len__(self):
        return _swig_decoders.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _swig_decoders.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _swig_decoders.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _swig_decoders.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _swig_decoders.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _swig_decoders.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _swig_decoders.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _swig_decoders.DoubleVector_pop(self)

    def append(self, x):
        return _swig_decoders.DoubleVector_append(self, x)

    def empty(self):
        return _swig_decoders.DoubleVector_empty(self)

    def size(self):
        return _swig_decoders.DoubleVector_size(self)

    def swap(self, v):
        return _swig_decoders.DoubleVector_swap(self, v)

    def begin(self):
        return _swig_decoders.DoubleVector_begin(self)

    def end(self):
        return _swig_decoders.DoubleVector_end(self)

    def rbegin(self):
        return _swig_decoders.DoubleVector_rbegin(self)

    def rend(self):
        return _swig_decoders.DoubleVector_rend(self)

    def clear(self):
        return _swig_decoders.DoubleVector_clear(self)

    def get_allocator(self):
        return _swig_decoders.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _swig_decoders.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _swig_decoders.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        _swig_decoders.DoubleVector_swiginit(self, _swig_decoders.new_DoubleVector(*args))

    def push_back(self, x):
        return _swig_decoders.DoubleVector_push_back(self, x)

    def front(self):
        return _swig_decoders.DoubleVector_front(self)

    def back(self):
        return _swig_decoders.DoubleVector_back(self)

    def assign(self, n, x):
        return _swig_decoders.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _swig_decoders.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _swig_decoders.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _swig_decoders.DoubleVector_reserve(self, n)

    def capacity(self):
        return _swig_decoders.DoubleVector_capacity(self)
    __swig_destroy__ = _swig_decoders.delete_DoubleVector

# Register DoubleVector in _swig_decoders:
_swig_decoders.DoubleVector_swigregister(DoubleVector)

class IntVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _swig_decoders.IntVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _swig_decoders.IntVector___nonzero__(self)

    def __bool__(self):
        return _swig_decoders.IntVector___bool__(self)

    def __len__(self):
        return _swig_decoders.IntVector___len__(self)

    def __getslice__(self, i, j):
        return _swig_decoders.IntVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _swig_decoders.IntVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _swig_decoders.IntVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _swig_decoders.IntVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _swig_decoders.IntVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _swig_decoders.IntVector___setitem__(self, *args)

    def pop(self):
        return _swig_decoders.IntVector_pop(self)

    def append(self, x):
        return _swig_decoders.IntVector_append(self, x)

    def empty(self):
        return _swig_decoders.IntVector_empty(self)

    def size(self):
        return _swig_decoders.IntVector_size(self)

    def swap(self, v):
        return _swig_decoders.IntVector_swap(self, v)

    def begin(self):
        return _swig_decoders.IntVector_begin(self)

    def end(self):
        return _swig_decoders.IntVector_end(self)

    def rbegin(self):
        return _swig_decoders.IntVector_rbegin(self)

    def rend(self):
        return _swig_decoders.IntVector_rend(self)

    def clear(self):
        return _swig_decoders.IntVector_clear(self)

    def get_allocator(self):
        return _swig_decoders.IntVector_get_allocator(self)

    def pop_back(self):
        return _swig_decoders.IntVector_pop_back(self)

    def erase(self, *args):
        return _swig_decoders.IntVector_erase(self, *args)

    def __init__(self, *args):
        _swig_decoders.IntVector_swiginit(self, _swig_decoders.new_IntVector(*args))

    def push_back(self, x):
        return _swig_decoders.IntVector_push_back(self, x)

    def front(self):
        return _swig_decoders.IntVector_front(self)

    def back(self):
        return _swig_decoders.IntVector_back(self)

    def assign(self, n, x):
        return _swig_decoders.IntVector_assign(self, n, x)

    def resize(self, *args):
        return _swig_decoders.IntVector_resize(self, *args)

    def insert(self, *args):
        return _swig_decoders.IntVector_insert(self, *args)

    def reserve(self, n):
        return _swig_decoders.IntVector_reserve(self, n)

    def capacity(self):
        return _swig_decoders.IntVector_capacity(self)
    __swig_destroy__ = _swig_decoders.delete_IntVector

# Register IntVector in _swig_decoders:
_swig_decoders.IntVector_swigregister(IntVector)

class StringVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _swig_decoders.StringVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _swig_decoders.StringVector___nonzero__(self)

    def __bool__(self):
        return _swig_decoders.StringVector___bool__(self)

    def __len__(self):
        return _swig_decoders.StringVector___len__(self)

    def __getslice__(self, i, j):
        return _swig_decoders.StringVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _swig_decoders.StringVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _swig_decoders.StringVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _swig_decoders.StringVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _swig_decoders.StringVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _swig_decoders.StringVector___setitem__(self, *args)

    def pop(self):
        return _swig_decoders.StringVector_pop(self)

    def append(self, x):
        return _swig_decoders.StringVector_append(self, x)

    def empty(self):
        return _swig_decoders.StringVector_empty(self)

    def size(self):
        return _swig_decoders.StringVector_size(self)

    def swap(self, v):
        return _swig_decoders.StringVector_swap(self, v)

    def begin(self):
        return _swig_decoders.StringVector_begin(self)

    def end(self):
        return _swig_decoders.StringVector_end(self)

    def rbegin(self):
        return _swig_decoders.StringVector_rbegin(self)

    def rend(self):
        return _swig_decoders.StringVector_rend(self)

    def clear(self):
        return _swig_decoders.StringVector_clear(self)

    def get_allocator(self):
        return _swig_decoders.StringVector_get_allocator(self)

    def pop_back(self):
        return _swig_decoders.StringVector_pop_back(self)

    def erase(self, *args):
        return _swig_decoders.StringVector_erase(self, *args)

    def __init__(self, *args):
        _swig_decoders.StringVector_swiginit(self, _swig_decoders.new_StringVector(*args))

    def push_back(self, x):
        return _swig_decoders.StringVector_push_back(self, x)

    def front(self):
        return _swig_decoders.StringVector_front(self)

    def back(self):
        return _swig_decoders.StringVector_back(self)

    def assign(self, n, x):
        return _swig_decoders.StringVector_assign(self, n, x)

    def resize(self, *args):
        return _swig_decoders.StringVector_resize(self, *args)

    def insert(self, *args):
        return _swig_decoders.StringVector_insert(self, *args)

    def reserve(self, n):
        return _swig_decoders.StringVector_reserve(self, n)

    def capacity(self):
        return _swig_decoders.StringVector_capacity(self)
    __swig_destroy__ = _swig_decoders.delete_StringVector

# Register StringVector in _swig_decoders:
_swig_decoders.StringVector_swigregister(StringVector)

class VectorOfStructVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _swig_decoders.VectorOfStructVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _swig_decoders.VectorOfStructVector___nonzero__(self)

    def __bool__(self):
        return _swig_decoders.VectorOfStructVector___bool__(self)

    def __len__(self):
        return _swig_decoders.VectorOfStructVector___len__(self)

    def __getslice__(self, i, j):
        return _swig_decoders.VectorOfStructVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _swig_decoders.VectorOfStructVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _swig_decoders.VectorOfStructVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _swig_decoders.VectorOfStructVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _swig_decoders.VectorOfStructVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _swig_decoders.VectorOfStructVector___setitem__(self, *args)

    def pop(self):
        return _swig_decoders.VectorOfStructVector_pop(self)

    def append(self, x):
        return _swig_decoders.VectorOfStructVector_append(self, x)

    def empty(self):
        return _swig_decoders.VectorOfStructVector_empty(self)

    def size(self):
        return _swig_decoders.VectorOfStructVector_size(self)

    def swap(self, v):
        return _swig_decoders.VectorOfStructVector_swap(self, v)

    def begin(self):
        return _swig_decoders.VectorOfStructVector_begin(self)

    def end(self):
        return _swig_decoders.VectorOfStructVector_end(self)

    def rbegin(self):
        return _swig_decoders.VectorOfStructVector_rbegin(self)

    def rend(self):
        return _swig_decoders.VectorOfStructVector_rend(self)

    def clear(self):
        return _swig_decoders.VectorOfStructVector_clear(self)

    def get_allocator(self):
        return _swig_decoders.VectorOfStructVector_get_allocator(self)

    def pop_back(self):
        return _swig_decoders.VectorOfStructVector_pop_back(self)

    def erase(self, *args):
        return _swig_decoders.VectorOfStructVector_erase(self, *args)

    def __init__(self, *args):
        _swig_decoders.VectorOfStructVector_swiginit(self, _swig_decoders.new_VectorOfStructVector(*args))

    def push_back(self, x):
        return _swig_decoders.VectorOfStructVector_push_back(self, x)

    def front(self):
        return _swig_decoders.VectorOfStructVector_front(self)

    def back(self):
        return _swig_decoders.VectorOfStructVector_back(self)

    def assign(self, n, x):
        return _swig_decoders.VectorOfStructVector_assign(self, n, x)

    def resize(self, *args):
        return _swig_decoders.VectorOfStructVector_resize(self, *args)

    def insert(self, *args):
        return _swig_decoders.VectorOfStructVector_insert(self, *args)

    def reserve(self, n):
        return _swig_decoders.VectorOfStructVector_reserve(self, n)

    def capacity(self):
        return _swig_decoders.VectorOfStructVector_capacity(self)
    __swig_destroy__ = _swig_decoders.delete_VectorOfStructVector

# Register VectorOfStructVector in _swig_decoders:
_swig_decoders.VectorOfStructVector_swigregister(VectorOfStructVector)

class FloatVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _swig_decoders.FloatVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _swig_decoders.FloatVector___nonzero__(self)

    def __bool__(self):
        return _swig_decoders.FloatVector___bool__(self)

    def __len__(self):
        return _swig_decoders.FloatVector___len__(self)

    def __getslice__(self, i, j):
        return _swig_decoders.FloatVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _swig_decoders.FloatVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _swig_decoders.FloatVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _swig_decoders.FloatVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _swig_decoders.FloatVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _swig_decoders.FloatVector___setitem__(self, *args)

    def pop(self):
        return _swig_decoders.FloatVector_pop(self)

    def append(self, x):
        return _swig_decoders.FloatVector_append(self, x)

    def empty(self):
        return _swig_decoders.FloatVector_empty(self)

    def size(self):
        return _swig_decoders.FloatVector_size(self)

    def swap(self, v):
        return _swig_decoders.FloatVector_swap(self, v)

    def begin(self):
        return _swig_decoders.FloatVector_begin(self)

    def end(self):
        return _swig_decoders.FloatVector_end(self)

    def rbegin(self):
        return _swig_decoders.FloatVector_rbegin(self)

    def rend(self):
        return _swig_decoders.FloatVector_rend(self)

    def clear(self):
        return _swig_decoders.FloatVector_clear(self)

    def get_allocator(self):
        return _swig_decoders.FloatVector_get_allocator(self)

    def pop_back(self):
        return _swig_decoders.FloatVector_pop_back(self)

    def erase(self, *args):
        return _swig_decoders.FloatVector_erase(self, *args)

    def __init__(self, *args):
        _swig_decoders.FloatVector_swiginit(self, _swig_decoders.new_FloatVector(*args))

    def push_back(self, x):
        return _swig_decoders.FloatVector_push_back(self, x)

    def front(self):
        return _swig_decoders.FloatVector_front(self)

    def back(self):
        return _swig_decoders.FloatVector_back(self)

    def assign(self, n, x):
        return _swig_decoders.FloatVector_assign(self, n, x)

    def resize(self, *args):
        return _swig_decoders.FloatVector_resize(self, *args)

    def insert(self, *args):
        return _swig_decoders.FloatVector_insert(self, *args)

    def reserve(self, n):
        return _swig_decoders.FloatVector_reserve(self, n)

    def capacity(self):
        return _swig_decoders.FloatVector_capacity(self)
    __swig_destroy__ = _swig_decoders.delete_FloatVector

# Register FloatVector in _swig_decoders:
_swig_decoders.FloatVector_swigregister(FloatVector)

class Pair(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _swig_decoders.Pair_swiginit(self, _swig_decoders.new_Pair(*args))
    first = property(_swig_decoders.Pair_first_get, _swig_decoders.Pair_first_set)
    second = property(_swig_decoders.Pair_second_get, _swig_decoders.Pair_second_set)
    def __len__(self):
        return 2
    def __repr__(self):
        return str((self.first, self.second))
    def __getitem__(self, index): 
        if not (index % 2):
            return self.first
        else:
            return self.second
    def __setitem__(self, index, val):
        if not (index % 2):
            self.first = val
        else:
            self.second = val
    __swig_destroy__ = _swig_decoders.delete_Pair

# Register Pair in _swig_decoders:
_swig_decoders.Pair_swigregister(Pair)

class PairFloatStringVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _swig_decoders.PairFloatStringVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _swig_decoders.PairFloatStringVector___nonzero__(self)

    def __bool__(self):
        return _swig_decoders.PairFloatStringVector___bool__(self)

    def __len__(self):
        return _swig_decoders.PairFloatStringVector___len__(self)

    def __getslice__(self, i, j):
        return _swig_decoders.PairFloatStringVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _swig_decoders.PairFloatStringVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _swig_decoders.PairFloatStringVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _swig_decoders.PairFloatStringVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _swig_decoders.PairFloatStringVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _swig_decoders.PairFloatStringVector___setitem__(self, *args)

    def pop(self):
        return _swig_decoders.PairFloatStringVector_pop(self)

    def append(self, x):
        return _swig_decoders.PairFloatStringVector_append(self, x)

    def empty(self):
        return _swig_decoders.PairFloatStringVector_empty(self)

    def size(self):
        return _swig_decoders.PairFloatStringVector_size(self)

    def swap(self, v):
        return _swig_decoders.PairFloatStringVector_swap(self, v)

    def begin(self):
        return _swig_decoders.PairFloatStringVector_begin(self)

    def end(self):
        return _swig_decoders.PairFloatStringVector_end(self)

    def rbegin(self):
        return _swig_decoders.PairFloatStringVector_rbegin(self)

    def rend(self):
        return _swig_decoders.PairFloatStringVector_rend(self)

    def clear(self):
        return _swig_decoders.PairFloatStringVector_clear(self)

    def get_allocator(self):
        return _swig_decoders.PairFloatStringVector_get_allocator(self)

    def pop_back(self):
        return _swig_decoders.PairFloatStringVector_pop_back(self)

    def erase(self, *args):
        return _swig_decoders.PairFloatStringVector_erase(self, *args)

    def __init__(self, *args):
        _swig_decoders.PairFloatStringVector_swiginit(self, _swig_decoders.new_PairFloatStringVector(*args))

    def push_back(self, x):
        return _swig_decoders.PairFloatStringVector_push_back(self, x)

    def front(self):
        return _swig_decoders.PairFloatStringVector_front(self)

    def back(self):
        return _swig_decoders.PairFloatStringVector_back(self)

    def assign(self, n, x):
        return _swig_decoders.PairFloatStringVector_assign(self, n, x)

    def resize(self, *args):
        return _swig_decoders.PairFloatStringVector_resize(self, *args)

    def insert(self, *args):
        return _swig_decoders.PairFloatStringVector_insert(self, *args)

    def reserve(self, n):
        return _swig_decoders.PairFloatStringVector_reserve(self, n)

    def capacity(self):
        return _swig_decoders.PairFloatStringVector_capacity(self)
    __swig_destroy__ = _swig_decoders.delete_PairFloatStringVector

# Register PairFloatStringVector in _swig_decoders:
_swig_decoders.PairFloatStringVector_swigregister(PairFloatStringVector)

class PairDoubleStringVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _swig_decoders.PairDoubleStringVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _swig_decoders.PairDoubleStringVector___nonzero__(self)

    def __bool__(self):
        return _swig_decoders.PairDoubleStringVector___bool__(self)

    def __len__(self):
        return _swig_decoders.PairDoubleStringVector___len__(self)

    def __getslice__(self, i, j):
        return _swig_decoders.PairDoubleStringVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _swig_decoders.PairDoubleStringVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _swig_decoders.PairDoubleStringVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _swig_decoders.PairDoubleStringVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _swig_decoders.PairDoubleStringVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _swig_decoders.PairDoubleStringVector___setitem__(self, *args)

    def pop(self):
        return _swig_decoders.PairDoubleStringVector_pop(self)

    def append(self, x):
        return _swig_decoders.PairDoubleStringVector_append(self, x)

    def empty(self):
        return _swig_decoders.PairDoubleStringVector_empty(self)

    def size(self):
        return _swig_decoders.PairDoubleStringVector_size(self)

    def swap(self, v):
        return _swig_decoders.PairDoubleStringVector_swap(self, v)

    def begin(self):
        return _swig_decoders.PairDoubleStringVector_begin(self)

    def end(self):
        return _swig_decoders.PairDoubleStringVector_end(self)

    def rbegin(self):
        return _swig_decoders.PairDoubleStringVector_rbegin(self)

    def rend(self):
        return _swig_decoders.PairDoubleStringVector_rend(self)

    def clear(self):
        return _swig_decoders.PairDoubleStringVector_clear(self)

    def get_allocator(self):
        return _swig_decoders.PairDoubleStringVector_get_allocator(self)

    def pop_back(self):
        return _swig_decoders.PairDoubleStringVector_pop_back(self)

    def erase(self, *args):
        return _swig_decoders.PairDoubleStringVector_erase(self, *args)

    def __init__(self, *args):
        _swig_decoders.PairDoubleStringVector_swiginit(self, _swig_decoders.new_PairDoubleStringVector(*args))

    def push_back(self, x):
        return _swig_decoders.PairDoubleStringVector_push_back(self, x)

    def front(self):
        return _swig_decoders.PairDoubleStringVector_front(self)

    def back(self):
        return _swig_decoders.PairDoubleStringVector_back(self)

    def assign(self, n, x):
        return _swig_decoders.PairDoubleStringVector_assign(self, n, x)

    def resize(self, *args):
        return _swig_decoders.PairDoubleStringVector_resize(self, *args)

    def insert(self, *args):
        return _swig_decoders.PairDoubleStringVector_insert(self, *args)

    def reserve(self, n):
        return _swig_decoders.PairDoubleStringVector_reserve(self, n)

    def capacity(self):
        return _swig_decoders.PairDoubleStringVector_capacity(self)
    __swig_destroy__ = _swig_decoders.delete_PairDoubleStringVector

# Register PairDoubleStringVector in _swig_decoders:
_swig_decoders.PairDoubleStringVector_swigregister(PairDoubleStringVector)

class PairDoubleStringVector2(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _swig_decoders.PairDoubleStringVector2_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _swig_decoders.PairDoubleStringVector2___nonzero__(self)

    def __bool__(self):
        return _swig_decoders.PairDoubleStringVector2___bool__(self)

    def __len__(self):
        return _swig_decoders.PairDoubleStringVector2___len__(self)

    def __getslice__(self, i, j):
        return _swig_decoders.PairDoubleStringVector2___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _swig_decoders.PairDoubleStringVector2___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _swig_decoders.PairDoubleStringVector2___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _swig_decoders.PairDoubleStringVector2___delitem__(self, *args)

    def __getitem__(self, *args):
        return _swig_decoders.PairDoubleStringVector2___getitem__(self, *args)

    def __setitem__(self, *args):
        return _swig_decoders.PairDoubleStringVector2___setitem__(self, *args)

    def pop(self):
        return _swig_decoders.PairDoubleStringVector2_pop(self)

    def append(self, x):
        return _swig_decoders.PairDoubleStringVector2_append(self, x)

    def empty(self):
        return _swig_decoders.PairDoubleStringVector2_empty(self)

    def size(self):
        return _swig_decoders.PairDoubleStringVector2_size(self)

    def swap(self, v):
        return _swig_decoders.PairDoubleStringVector2_swap(self, v)

    def begin(self):
        return _swig_decoders.PairDoubleStringVector2_begin(self)

    def end(self):
        return _swig_decoders.PairDoubleStringVector2_end(self)

    def rbegin(self):
        return _swig_decoders.PairDoubleStringVector2_rbegin(self)

    def rend(self):
        return _swig_decoders.PairDoubleStringVector2_rend(self)

    def clear(self):
        return _swig_decoders.PairDoubleStringVector2_clear(self)

    def get_allocator(self):
        return _swig_decoders.PairDoubleStringVector2_get_allocator(self)

    def pop_back(self):
        return _swig_decoders.PairDoubleStringVector2_pop_back(self)

    def erase(self, *args):
        return _swig_decoders.PairDoubleStringVector2_erase(self, *args)

    def __init__(self, *args):
        _swig_decoders.PairDoubleStringVector2_swiginit(self, _swig_decoders.new_PairDoubleStringVector2(*args))

    def push_back(self, x):
        return _swig_decoders.PairDoubleStringVector2_push_back(self, x)

    def front(self):
        return _swig_decoders.PairDoubleStringVector2_front(self)

    def back(self):
        return _swig_decoders.PairDoubleStringVector2_back(self)

    def assign(self, n, x):
        return _swig_decoders.PairDoubleStringVector2_assign(self, n, x)

    def resize(self, *args):
        return _swig_decoders.PairDoubleStringVector2_resize(self, *args)

    def insert(self, *args):
        return _swig_decoders.PairDoubleStringVector2_insert(self, *args)

    def reserve(self, n):
        return _swig_decoders.PairDoubleStringVector2_reserve(self, n)

    def capacity(self):
        return _swig_decoders.PairDoubleStringVector2_capacity(self)
    __swig_destroy__ = _swig_decoders.delete_PairDoubleStringVector2

# Register PairDoubleStringVector2 in _swig_decoders:
_swig_decoders.PairDoubleStringVector2_swigregister(PairDoubleStringVector2)

class DoubleVector3(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _swig_decoders.DoubleVector3_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _swig_decoders.DoubleVector3___nonzero__(self)

    def __bool__(self):
        return _swig_decoders.DoubleVector3___bool__(self)

    def __len__(self):
        return _swig_decoders.DoubleVector3___len__(self)

    def __getslice__(self, i, j):
        return _swig_decoders.DoubleVector3___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _swig_decoders.DoubleVector3___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _swig_decoders.DoubleVector3___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _swig_decoders.DoubleVector3___delitem__(self, *args)

    def __getitem__(self, *args):
        return _swig_decoders.DoubleVector3___getitem__(self, *args)

    def __setitem__(self, *args):
        return _swig_decoders.DoubleVector3___setitem__(self, *args)

    def pop(self):
        return _swig_decoders.DoubleVector3_pop(self)

    def append(self, x):
        return _swig_decoders.DoubleVector3_append(self, x)

    def empty(self):
        return _swig_decoders.DoubleVector3_empty(self)

    def size(self):
        return _swig_decoders.DoubleVector3_size(self)

    def swap(self, v):
        return _swig_decoders.DoubleVector3_swap(self, v)

    def begin(self):
        return _swig_decoders.DoubleVector3_begin(self)

    def end(self):
        return _swig_decoders.DoubleVector3_end(self)

    def rbegin(self):
        return _swig_decoders.DoubleVector3_rbegin(self)

    def rend(self):
        return _swig_decoders.DoubleVector3_rend(self)

    def clear(self):
        return _swig_decoders.DoubleVector3_clear(self)

    def get_allocator(self):
        return _swig_decoders.DoubleVector3_get_allocator(self)

    def pop_back(self):
        return _swig_decoders.DoubleVector3_pop_back(self)

    def erase(self, *args):
        return _swig_decoders.DoubleVector3_erase(self, *args)

    def __init__(self, *args):
        _swig_decoders.DoubleVector3_swiginit(self, _swig_decoders.new_DoubleVector3(*args))

    def push_back(self, x):
        return _swig_decoders.DoubleVector3_push_back(self, x)

    def front(self):
        return _swig_decoders.DoubleVector3_front(self)

    def back(self):
        return _swig_decoders.DoubleVector3_back(self)

    def assign(self, n, x):
        return _swig_decoders.DoubleVector3_assign(self, n, x)

    def resize(self, *args):
        return _swig_decoders.DoubleVector3_resize(self, *args)

    def insert(self, *args):
        return _swig_decoders.DoubleVector3_insert(self, *args)

    def reserve(self, n):
        return _swig_decoders.DoubleVector3_reserve(self, n)

    def capacity(self):
        return _swig_decoders.DoubleVector3_capacity(self)
    __swig_destroy__ = _swig_decoders.delete_DoubleVector3

# Register DoubleVector3 in _swig_decoders:
_swig_decoders.DoubleVector3_swigregister(DoubleVector3)


def IntDoublePairCompSecondRev(a, b):
    return _swig_decoders.IntDoublePairCompSecondRev(a, b)

def StringDoublePairCompSecondRev(a, b):
    return _swig_decoders.StringDoublePairCompSecondRev(a, b)

def DoubleStringPairCompFirstRev(a, b):
    return _swig_decoders.DoubleStringPairCompFirstRev(a, b)
class RetriveStrEnumerateVocab(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _swig_decoders.RetriveStrEnumerateVocab_swiginit(self, _swig_decoders.new_RetriveStrEnumerateVocab())

    def Add(self, index, str):
        return _swig_decoders.RetriveStrEnumerateVocab_Add(self, index, str)
    vocabulary = property(_swig_decoders.RetriveStrEnumerateVocab_vocabulary_get, _swig_decoders.RetriveStrEnumerateVocab_vocabulary_set)
    __swig_destroy__ = _swig_decoders.delete_RetriveStrEnumerateVocab

# Register RetriveStrEnumerateVocab in _swig_decoders:
_swig_decoders.RetriveStrEnumerateVocab_swigregister(RetriveStrEnumerateVocab)
cvar = _swig_decoders.cvar
OOV_SCORE = cvar.OOV_SCORE
START_TOKEN = cvar.START_TOKEN
UNK_TOKEN = cvar.UNK_TOKEN
END_TOKEN = cvar.END_TOKEN

class Scorer(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, alpha, beta, lm_path, vocabulary):
        _swig_decoders.Scorer_swiginit(self, _swig_decoders.new_Scorer(alpha, beta, lm_path, vocabulary))
    __swig_destroy__ = _swig_decoders.delete_Scorer

    def get_log_cond_prob(self, words):
        return _swig_decoders.Scorer_get_log_cond_prob(self, words)

    def get_sent_log_prob(self, words):
        return _swig_decoders.Scorer_get_sent_log_prob(self, words)

    def get_max_order(self):
        return _swig_decoders.Scorer_get_max_order(self)

    def get_dict_size(self):
        return _swig_decoders.Scorer_get_dict_size(self)

    def is_character_based(self):
        return _swig_decoders.Scorer_is_character_based(self)

    def reset_params(self, alpha, beta):
        return _swig_decoders.Scorer_reset_params(self, alpha, beta)

    def make_ngram(self, prefix):
        return _swig_decoders.Scorer_make_ngram(self, prefix)

    def split_labels(self, labels):
        return _swig_decoders.Scorer_split_labels(self, labels)
    alpha = property(_swig_decoders.Scorer_alpha_get, _swig_decoders.Scorer_alpha_set)
    beta = property(_swig_decoders.Scorer_beta_get, _swig_decoders.Scorer_beta_set)
    dictionary = property(_swig_decoders.Scorer_dictionary_get, _swig_decoders.Scorer_dictionary_set)

# Register Scorer in _swig_decoders:
_swig_decoders.Scorer_swigregister(Scorer)


def ctc_greedy_decoder(probs_seq, vocabulary):
    return _swig_decoders.ctc_greedy_decoder(probs_seq, vocabulary)

def ctc_beam_search_decoder(probs_seq, vocabulary, beam_size, cutoff_prob=1.0, cutoff_top_n=40, ext_scorer=None):
    return _swig_decoders.ctc_beam_search_decoder(probs_seq, vocabulary, beam_size, cutoff_prob, cutoff_top_n, ext_scorer)

def ctc_beam_search_decoder_batch(probs_split, vocabulary, beam_size, num_processes, cutoff_prob=1.0, cutoff_top_n=40, ext_scorer=None):
    return _swig_decoders.ctc_beam_search_decoder_batch(probs_split, vocabulary, beam_size, num_processes, cutoff_prob, cutoff_top_n, ext_scorer)


