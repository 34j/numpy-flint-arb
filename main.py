from cm_time import timer
from flint import arb_series, ctx

from numpy_flint_arb import np

ctx.threads = 32
print(ctx)
a = np.arange(40000000)
b = np.arange(40000000)
with timer() as t:
    a + b
print(f"Time taken for addition: {t.elapsed} seconds")
ctx.cap = 40000000

with timer() as t:
    a = arb_series(a.tolist())
    b = arb_series(b.tolist())
print(f"Time taken for conversion to arb_series: {t.elapsed} seconds")
with timer() as t:
    a + b
# print(a + b)
print(f"Time taken for addition using arb_series: {t.elapsed} seconds")
