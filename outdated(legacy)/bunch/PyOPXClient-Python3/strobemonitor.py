# Strobe_Monitor_Example.py - StrobeMonitorController example that creates the Strobe Monitor allowing 
# the user to view the time stamp and value of each strobe data point

# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.


from example_libs.strobemonitor_lib.controller import Controller

if __name__ == "__main__":
    # Initializes the StrobeMonitorController class
    app = Controller()