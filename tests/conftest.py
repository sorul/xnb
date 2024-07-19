"""Conftest utilities for testing."""
import pytest


def pytest_addoption(parser):
  """You can insert a command line option for benchmarking.

  Example:
  poetry run pytest --run-benchmark -k test_calculate_bw_function
  """
  parser.addoption(
      '--run-benchmark',
      action='store_true',
      default=False,
      help='Run benchmark tests'
  )


@pytest.fixture()
def is_benchmark(request):
  """You have to put 'is_benchmark' as parameter in your pytest function."""
  return request.config.getoption('--run-benchmark')
