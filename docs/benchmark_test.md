You can use the `--run-benchmark` command if you want 
to run a benchmark for multiple tests. An invented example would be this:

```
poetry run pytest --run-benchmark -k test_predict -v
```

Output:
```
❯ poetry run pytest --run-benchmark -k test_predict -v
================================================================================================================== test session starts ===================================================================================================================
platform linux -- Python 3.10.0, pytest-7.4.4, pluggy-1.5.0 -- /home/sorul/.cache/pypoetry/virtualenvs/xnb-mB_DTWSl-py3.10/bin/python
cachedir: .pytest_cache
benchmark: 4.0.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /home/sorul/git/xnb_classifier
configfile: pyproject.toml
plugins: cov-5.0.0, benchmark-4.0.0
collected 22 items / 20 deselected / 2 selected                                                                                                                                                                                                          

tests/xnb_test.py::test_predict PASSED                                                                                                                                                                                                             [ 50%]
tests/xnb_test.py::test_predict_vectorized PASSED                                                                                                                                                                                                  [100%]


----------------------------------------------------------------------------------- benchmark: 2 tests -----------------------------------------------------------------------------------
Name (time in s)                Min                Max               Mean            StdDev             Median               IQR            Outliers     OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_predict_vectorized      8.0936 (1.0)       8.1040 (1.0)       8.0991 (1.0)      0.0040 (1.0)       8.0997 (1.0)      0.0060 (1.0)           2;0  0.1235 (1.0)           5           1
test_predict                45.0030 (5.56)     46.6439 (5.76)     45.7546 (5.65)     0.7003 (173.08)   45.9298 (5.67)     1.1798 (197.21)        2;0  0.0219 (0.18)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
