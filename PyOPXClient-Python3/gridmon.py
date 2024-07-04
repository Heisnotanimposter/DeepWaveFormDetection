# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from example_libs.gridmon_lib.controller import Controller
from example_libs.gridmon_lib.main_window import MainWindow
import sys

# This is the entry point of the Gridmon application. If run with -demo
# command line toggle, Gridmon assumes that it's connecting to the demo
# version of OmniPlex.
if __name__ == "__main__":
    if "-demo" in sys.argv:
        omniplex_demo_configuration = True
    else:
        omniplex_demo_configuration = False

    # Create instance of Controller class
    app = Controller(omniplex_demo_configuration=omniplex_demo_configuration)
    # Run the application
    app.run()
