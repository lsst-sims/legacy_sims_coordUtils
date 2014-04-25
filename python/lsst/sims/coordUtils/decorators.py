from functools import wraps
from collections import OrderedDict
#---------------------------------------------------------------------- 
# Define decorators for get_* methods

# The cached decorator specifies that once the column is computed for
# a given database chunk, it is cached in memory and not computed again.


def cached(f):
    """Decorator for specifying that the computed result should be cached"""
    if not f.__name__.startswith('get_'):
        raise ValueError("@cached can only be applied to get_* methods: "
                         "Method '%s' invalid." % f.__name__)
    colname = f.__name__.lstrip('get_')
    @wraps(f)
    def new_f(self, *args, **kwargs):
        if colname in self._column_cache:
            result = self._column_cache[colname]
        else:
            result = f(self, *args, **kwargs)
            self._column_cache[colname] = result
        return result
    new_f._cache_results = True
    return new_f

def compound(*colnames):
    """Specifies that a column is a "compound column",
 that is, it returns multiple values.  This is useful in the case of,
 e.g. RA/DEC, or magnitudes.

 For example, to return an RA and a DEC together, use, e.g.::

     @compound('ra_corr', 'dec_corr')
     def get_point_correction(self):
         raJ2000 = self.column_by_name('raJ2000')
         decJ2000 - self.column_by_name('decJ2000')
     ra_corr, dec_corr = precess(raJ2000, decJ2000)
     return (ra_corr, dec_corr)

"""
    def wrapper(f):
        @cached
        @wraps(f)
        def new_f(self, *args, **kwargs):
            results = f(self, *args, **kwargs)
            return OrderedDict(zip(colnames, results))
        new_f._compound_column = True
        new_f._colnames = colnames
        return new_f
    return wrapper

def register_class(cls):
    if not hasattr(cls, '_methodRegistry'):
        cls._methodRegistry = {}
    for methodname in dir(cls):
        method=getattr(cls, methodname)
        if hasattr(method, '_registryKey'):
            cls._methodRegistry.update({method._registryKey:method})
    return cls

def register_method(key):
    def wrapper(func):
        func._registryKey=key
        return func
    return wrapper
