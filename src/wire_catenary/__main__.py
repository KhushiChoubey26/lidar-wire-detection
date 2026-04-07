"""Allow running the package as a module: python -m wire_catenary <args>"""

import sys
from .cli import main

sys.exit(main())