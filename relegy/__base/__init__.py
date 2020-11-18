from abc import ABC, abstractmethod
from functools import wraps
from inspect import getfullargspec

from networkx import Graph
from numpy import ndarray


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
            def wrap(self, *args, **kwargs):
                named_args = {
                                **kwargs,
                                **dict(
                                    zip(
                                        filter(
                                            lambda x: x not in kwargs and x != 'self',
                                            getfullargspec(func).args
                                        ),
                                        args
                                    )
                                )
                }
                print(named_args)
                for key, rules in rules_dict.items():
                    val = named_args[key]
                    for rule, err_msg in rules:
                        if not rule(val):
                            raise Exception(err_msg)
                res = func(self, *args, **kwargs)
                return res
            return wrap
        return inner_func

    @abstractmethod
    def initialize(self): pass

    @abstractmethod
    def fit(self): pass

    @abstractmethod
    def embed(self) -> ndarray: pass

    @abstractmethod
    def info(self): pass

    @staticmethod
    @abstractmethod
    def fast_embed(graph: Graph) -> ndarray: pass

    def get_graph(self) -> Graph: return self.__graph
