"""Built-in boxes.

Every module in this package is one self-contained pane: it imports the pane
contract, the rendering primitives and the data it displays, and never the
curses application.  ``pane_registry`` maps a configured pane name to one of
these modules, and ``config.toml`` can point a name at any other module.
"""

from __future__ import annotations
