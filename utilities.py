#!/usr/bin/env python3
from datetime import timedelta, datetime
from time import monotonic
from functools import wraps
from console import console

midnight = datetime(2000, 1, 1)

def log_timing(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    start_time = monotonic()
    result = func(*args, **kwargs)  # Call the original function
    end_time = monotonic()
    adjusted = midnight + timedelta(seconds=end_time - start_time)
    console.print(f"{func.__name__} took {adjusted.strftime('%M:%S')} seconds to execute\n", style="yellow")
    return result
  return wrapper
