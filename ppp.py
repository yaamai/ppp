import pytest
import inspect
import decopatch
import yaml
from contextlib import contextmanager
from pathlib import Path
from pydantic import TypeAdapter, create_model
from typing import List, Any, Callable, Optional
from functools import reduce

PPP_TEST_DATA: dict[str, Any] = {}


@contextmanager
def _patch_yaml_mapping_constructor(func: Callable):
    original_func = yaml.constructor.BaseConstructor.construct_mapping
    setattr(yaml.constructor.BaseConstructor, "org_construct_mapping", original_func)
    yaml.constructor.BaseConstructor.construct_mapping = func
    yield
    yaml.constructor.BaseConstructor.construct_mapping = original_func


def _split_ref_to_path_and_ref(ref: str) -> tuple[Optional[Path], tuple[str, ...]]:
    refpath = Path(ref)
    for idx in range(len(refpath.parts)):
        path = Path(*refpath.parts[:idx+1])
        if path.exists:
            return path, refpath.parts[idx+1:]
    return None, ()


def _resolve_jsonschema_like_refs(self, node, **kwargs):
    m = self.org_construct_mapping(node, **kwargs)

    # resolve ref
    for k in m:
        if k == "$ref":
            path, ref = _split_ref_to_path_and_ref(m[k])
            if not path:
                continue

            with path.open() as f:
                data = yaml.safe_load(f.read())
                return reduce(lambda d,k: d and d.get(k), ref, data)
    return m


def _gen_testdata_model(func):
    annotations = func.__annotations__
    argcount = func.__code__.co_argcount
    argnames = set(func.__code__.co_varnames[:argcount]) & set(annotations.keys())

    model = create_model(
        str(func.__name__ + "TestData"),
        **{k: (v, ...) for k, v in annotations.items()},  # type: ignore
    )
    model.argnames = argnames
    model.__len__ = lambda self: len(self.model_fields)
    model.__iter__ = lambda self: (getattr(self, k) for k in argnames)

    return TypeAdapter(List[model]), argnames


def _get_func_module_path(func) -> str:
    mod = inspect.getmodule(func)
    return mod.__file__  # type: ignore


def _load_testdata(func):
    global PPP_TEST_DATA

    path = Path(_get_func_module_path(func)).with_suffix(".yaml")
    if path in PPP_TEST_DATA:
        return PPP_TEST_DATA[path].get(func.__name__)

    with open(path, "r") as f:
        with _patch_yaml_mapping_constructor(_resolve_jsonschema_like_refs):
            PPP_TEST_DATA[str(path)] = yaml.safe_load(f.read())
            breakpoint()

    return PPP_TEST_DATA[str(path)].get(func.__name__)


def _parse_testdata(testdata, func):
    model, argnames = _gen_testdata_model(func)
    return argnames, model.validate_python(testdata)


def _decorator_factory(api_func):
    @decopatch.decorator  # type: ignore
    def decorator(test_func=decopatch.DECORATED):
        raw_testdata = _load_testdata(test_func)
        testdata_keys, testdata = _parse_testdata(raw_testdata, test_func)
        return api_func(testdata_keys, testdata)(test_func)

    return decorator


@_decorator_factory
def ppp(param_names=[], param_values=[], **kwargs) -> Callable:
    return pytest.mark.parametrize(param_names, param_values, **kwargs)
