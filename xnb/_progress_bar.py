from alive_progress import alive_bar
from contextlib import contextmanager


class _DummyBar:
  def __call__(self):
    pass


@contextmanager
def progress_bar(enabled: bool, total: int, title: str = ''):
  """Context manager to show a progress bar."""
  if enabled:
    with alive_bar(total, title=title) as bar:
      yield bar
  else:
    yield _DummyBar()
