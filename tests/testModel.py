import sys, inspect
from relegy.graphs.examples import examplesDict
import relegy

def is_method_with_model(c):
    required_methods = ['initialize', 'initialize_model', 'fit', 'embed']
    return all([x in dir(c) for x in required_methods])


def is_method_without_model(c):
    required_methods = ['initialize', 'fit', 'embed']
    forbidden_methods = ['initialize_model']
    return all([x in dir(c) for x in required_methods]) and all([x not in dir(c) for x in forbidden_methods])

def get_embedding_methods_iterable():
    return filter(lambda x: x[0][:2] != "__", inspect.getmembers(sys.modules['relegy.embeddings']))

def get_embedding_methods_iterable_without_GCN_GNN():
    return filter(lambda x: x[0] not in ['GCN','GNN'], get_embedding_methods_iterable())

def check_init_model_no_init_with_model(m):
    try: m.initialize_model()
    except Exception as e: assert str(e) == "The method 'initialize' must be called before initializing the model"

def check_fit_no_init_with_model(m):
    try: m.fit()
    except Exception as e: assert str(e) == "The methods 'initialize' and 'initialize_model' must be called before fitting"

def check_embed_no_init_with_model(m):
    try: m.embed()
    except Exception as e: assert str(e) == "The methods 'initialize', 'initialize_model' and 'fit' must be called before embedding"

def check_fit_no_init_model_with_model(m):
    try: m.fit()
    except Exception as e: assert str(e) == "The method 'initialize_model' must be called before fitting"

def check_embed_no_init_model_with_model(m):
    try: m.embed()
    except Exception as e: assert str(e) == "The methods 'initialize_model' and 'fit' must be called before embedding"

def check_embed_no_fit_with_model(m):
    try: m.embed()
    except Exception as e: assert str(e) == "The method 'fit' must be called before embedding"

def check_fit_no_init_without_model(m):
    try: m.fit()
    except Exception as e: assert str(e) == "The method 'initialize' must be called before fitting"

def check_embed_no_init_without_model(m):
    try: m.embed()
    except Exception as e: assert str(e) == "The methods 'initialize' and 'fit' must be called before embedding"

def check_embed_no_fit_without_model(m):
    try: m.embed()
    except Exception as e: assert str(e) == "The method 'fit' must be called before embedding"

def test_all_methods_are_with_or_without_model():
    invalid_implementations = []
    for name, class_handle in get_embedding_methods_iterable():
        if not (is_method_with_model(class_handle) or is_method_without_model(class_handle)):
            invalid_implementations.append((name, class_handle))
    assert len(invalid_implementations) == 0

def test_all_methods_validate_init_fit_embed_order():
    invalid_implementations = []
    graph = examplesDict['barbell']
    for name, class_handle in get_embedding_methods_iterable_without_GCN_GNN():
        try:
            m = class_handle(graph)
            if is_method_with_model(class_handle):
                # no initialize
                check_init_model_no_init_with_model(m)
                check_fit_no_init_with_model(m)
                check_embed_no_init_with_model(m)

                # no initialize_model
                m.initialize()
                check_fit_no_init_model_with_model(m)
                check_embed_no_init_model_with_model(m)

                # no fit
                m.initialize_model()
                check_embed_no_fit_with_model(m)
            elif is_method_without_model(class_handle):
                # no initialize
                check_fit_no_init_without_model(m)
                check_embed_no_init_without_model(m)

                # no fit
                m.initialize()
                check_embed_no_fit_without_model(m)
        except AssertionError:
            invalid_implementations.append((name, class_handle))
    if len(invalid_implementations) != 0:
        print(invalid_implementations)
    assert len(invalid_implementations) == 0

def test_all_methods_validation_status_resets():
    invalid_implementations = []
    graph = examplesDict['barbell']
    for name, class_handle in get_embedding_methods_iterable_without_GCN_GNN():
        try:
            m = class_handle(graph)
            if is_method_with_model(class_handle):
                m.initialize()
                m.initialize_model()
                m.initialize()
                check_fit_no_init_model_with_model(m)
                check_embed_no_init_model_with_model(m)

                m.initialize()
                m.initialize_model()
                m.fit()
                m.initialize()
                check_fit_no_init_model_with_model(m)
                check_embed_no_init_model_with_model(m)

                m.initialize()
                m.initialize_model()
                m.fit()
                m.initialize_model()
                check_embed_no_fit_with_model(m)

            elif is_method_without_model(class_handle):
                m.initialize()
                m.fit()
                m.initialize()
                check_fit_no_init_without_model(m)
                check_embed_no_init_without_model(m)
        except AssertionError:
            invalid_implementations.append((name, class_handle))
    if len(invalid_implementations) != 0:
        print(invalid_implementations)
    assert len(invalid_implementations) == 0