from abc import ABC, abstractmethod
from functools import wraps
from inspect import getfullargspec, getmembers, signature
import inspect

from networkx import Graph
from numpy import ndarray
from .info import info_dict
import itertools

construct_verification = {"graph": [(lambda x: type(x) == Graph, "'graph' must be a networkx graph")]}


class Model(ABC):

    def __init__(self, graph):
        self.__graph: Graph = graph
        self._initialized: bool = False
        self._initialized_model: bool = False
        self._fitted: bool = False
        super().__init__()

    def _verify_init_in_init_model(self):
        if not self._initialized:
            raise Exception("The method 'initialize' must be called before initializing the model")

    def _verify_init_in_fit(self):
        if not self._initialized:
            raise Exception("The method 'initialize' must be called before fitting")

    def _verify_init_and_init_model_in_fit(self):
        if not self._initialized:
            raise Exception("The methods 'initialize' and 'initialize_model' must be called before fitting")
        if not self._initialized_model:
            raise Exception("The method 'initialize_model' must be called before fitting")

    def _verify_init_and_fit_in_embed(self):
        if not self._initialized:
            raise Exception("The methods 'initialize' and 'fit' must be called before embedding")
        if not self._fitted:
            raise Exception("The method 'fit' must be called before embedding")

    def _verify_init_and_init_model_and_fit_in_embed(self):
        if not self._initialized:
            raise Exception("The methods 'initialize', 'initialize_model' and 'fit' must be called before embedding")
        if not self._initialized_model:
            raise Exception("The methods 'initialize_model' and 'fit' must be called before embedding")
        if not self._fitted:
            raise Exception("The method 'fit' must be called before embedding")

    def _update_init_in_init_model_fit(self):
        self._initialized = True
        self._initialized_model = False
        self._fitted = False

    def _update_init_in_init_fit(self):
        self._initialized = True
        self._fitted = False

    def _update_init_model_in_model_fit(self):
        self._initialized_model = True
        self._fitted = False

    def _update_fit(self):
        self._fitted = True

    @staticmethod
    def _init_in_init_fit(func):
        @wraps(func)
        def wrap(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self._update_init_in_init_fit()
        return wrap

    @staticmethod
    def _fit_in_init_fit(func):
        @wraps(func)
        def wrap(self, *args, **kwargs):
            self._verify_init_in_fit()
            func(self, *args, **kwargs)
            self._update_fit()
        return wrap

    @staticmethod
    def _embed_in_init_fit(func):
        @wraps(func)
        def wrap(self, *args, **kwargs):
            self._verify_init_and_fit_in_embed()
            result = func(self, *args, **kwargs)
            return result
        return wrap

    @staticmethod
    def _init_in_init_model_fit(func):
        @wraps(func)
        def wrap(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self._update_init_in_init_model_fit()
        return wrap

    @staticmethod
    def _init_model_in_init_model_fit(func):
        @wraps(func)
        def wrap(self, *args, **kwargs):
            self._verify_init_in_init_model()
            func(self, *args, **kwargs)
            self._update_init_model_in_model_fit()
        return wrap

    @staticmethod
    def _fit_in_init_model_fit(func):
        @wraps(func)
        def wrap(self, *args, **kwargs):
            self._verify_init_and_init_model_in_fit()
            func(self, *args, **kwargs)
            self._update_fit()
        return wrap

    @staticmethod
    def _embed_in_init_model_fit(func):
        @wraps(func)
        def wrap(self, *args, **kwargs):
            self._verify_init_and_init_model_and_fit_in_embed()
            res = func(self, *args, **kwargs)
            return res
        return wrap

    @staticmethod
    def _verify_parameters(rules_dict: dict):
        def inner_func(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                func_args = fa if (fa := getfullargspec(func).args) is not None else []
                func_defaults = fd if (fd := getfullargspec(func).defaults) is not None else []
                unnamed_args = dict(zip(func_args, args))
                named_args = dict(zip(reversed(func_args), reversed(func_defaults)))
                named_args.update(unnamed_args)
                named_args.update(kwargs)
                for key, rules in rules_dict.items():
                    val = named_args[key]
                    for rule, err_msg in rules:
                        if not rule(val):
                            raise Exception(err_msg)
                res = func(*args, **kwargs)
                return res
            return wrap
        return inner_func

    @staticmethod
    def dict_union(*dicts):
        return dict(itertools.chain.from_iterable(dct.items() for dct in dicts))

    @abstractmethod
    def initialize(self): pass

    @abstractmethod
    def fit(self): pass

    @abstractmethod
    def embed(self) -> ndarray: pass

    def info(self):
        name = type(self).__name__
        print(info_dict[name])
        function_dict = dict(getmembers(type(self)))
        df_values = []
        # __init__
        sig_init = signature(function_dict["__init__"])
        for value in sig_init.parameters.values():
            if value.name != 'self': df_values.append((value.name, value.default, value.annotation, "__init__"))
        # initialize
        sig_initialize = signature(function_dict["initialize"])
        for value in sig_initialize.parameters.values():
            if value.name != 'self': df_values.append((value.name, value.default, value.annotation, "initialize"))
        # initialize_model if possible
        if "initialize_model" in function_dict:
            sig_initialize_model = signature(function_dict["initialize_model"])
            for value in sig_initialize_model.parameters.values():
                if value.name != 'self': df_values.append((value.name, value.default, value.annotation, "initialize_model"))
        # fit
        sig_fit = signature(function_dict["fit"])
        for value in sig_fit.parameters.values():
            if value.name != 'self': df_values.append((value.name, value.default, value.annotation, "fit"))
        # embed
        sig_embed = signature(function_dict["embed"])
        for value in sig_embed.parameters.values():
            if value.name != 'self': df_values.append((value.name, value.default, value.annotation, "embed"))

        self.__print_info_table(df_values)

        return

    def __print_info_table(self, df_values):
        print("".join(["="] * 108))
        print(f"|{'parameter name':20s}|{'default value':25s}|{'annotated type':42s}|{'stage':16s}|")
        print("".join(["="] * 108))
        for name, default, annotation, stage in df_values:
            if default is inspect._empty:
                str_default = "no default"
            else:
                str_default = str_default if len(str_default := str(default)) <= 25 else str_default[0:7] + "..." + str_default[-15:]
            if annotation is inspect._empty:
                short_annotation = "no annotation"
            else:
                short_annotation = annotation.__module__ + "." + annotation.__name__
            print(f"|{name:20s}|{str_default:25s}|{short_annotation:42s}|{stage:16s}|")
            print("".join(["-"] * 108))

    @staticmethod
    @abstractmethod
    def fast_embed(graph: Graph) -> ndarray: pass

    def get_graph(self) -> Graph: return self.__graph
