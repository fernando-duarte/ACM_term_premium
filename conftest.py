# Root-level conftest.py: add repo root to sys.path so `import reproduce_acm` works.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
