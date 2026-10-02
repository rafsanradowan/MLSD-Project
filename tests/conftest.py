import sys
from pathlib import Path

# let the tests import the stage scripts from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
