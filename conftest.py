"""pytestがリポジトリルートを解決できるようにするための設定。"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
