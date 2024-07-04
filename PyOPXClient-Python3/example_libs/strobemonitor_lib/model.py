# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from pyopxclient import PyOPXClientAPI


class Model():
    """
    The Model class contains the functions that connect the PyOPXClientAPI to the Strobe Monitor.
    It allows the model to update and run using functions from the PyOPXClientAPI class.

    Args:
        source_number - the source number correlates to the number of the Plexon Digital Input. It is used to ensure
        that the data is strobe data 
    
    Returns:
        Initializes the class object's private global variables.

    self.client - initializes the PyOPXClientAPI into a variable 
    """

    def __init__(self, source_number=10):
        """
        Initializes the PyOPXClientAPI class and sets the source_number to source_number.
        
        Arg:
            source_number - the source number
        
        Returns:
            none
            
        self.client - the variable representing the PyOPXClient class. 
        """
        self.source_number = source_number
        self.client = PyOPXClientAPI()
        self.connected = False
        self.source_ids = []

    def connect(self):
        """
        Runs the connect function from the PyOPXClientAPI allowing the application to collect data from OmniPlex.
        
        Args:
            none
        
        Returns:
            none
        """
        self.client.connect()
        if self.client.connected:
            global_parameters = self.client.get_global_parameters()
            self.source_ids = global_parameters.source_ids
            self.connected = self.client.connected

    def disconnect(self):
        """
        Runs the disconnect function from the PyOPXClientAPI stopping the application from collecting data from
        OmniPlex.
        
        Args:
            none
        
        Returns:
            none
        """
        self.client.disconnect()
        self.connected = self.client.connected

    def get_source_number(self):
        """
        Returns the source number.
        
        Args:
            none
        
        Return:
            self.source - the source number
        """
        return self.source_number
        
    def change_source_number(self, source_number):
        """
        Changes the source number. 
        
        Arg:
            source_number - the source number
        
        Returns:
            none
        """
        self.source_number = source_number

    def get_source_name(self, source_number):
        """
        Gets the name of the source depending on its number.
        
        Arg:
            source_number - the number of the source
        
        Return:
            source_name - the name of the source with source_number

        source_name - see above
        source_type - one of the values SPIKE_TYPE, CONTINUOUS_TYPE, EVENT_TYPE, or OTHER_TYPE (currently 
        not used)
        num_chans - number of channels in the source (currently not used)
        linear_start_chan - starting channel number for the source, within the linear array of channels of 
        the specified type (currently not used)
        """
        source_name, source_type, num_chans, linear_start_chan = self.client.get_source_info(source_number)
        return source_name

    def model_update(self):
        """
        Generates the strobe data by checking to see if the source number of each element in the new_data 
        list matches self.source_number. If this is the case, the time stamp and unit of that element are 
        added into the data list.

        Args:
            none
        
        Return:
            data - a list of tuples that represent the strobe time stamp and value data
        
        tuple_of_data -  creates a named tuple where the first parameter is the time stamp of the
        strobe event and the second parameter is the value of the strobe event 
        current_data - stores the time stamp and value of each strobe event in a temporary variable 
        to be appended into the data list 
        """
        # waits until there is data (wait time max is 1 second)
        self.client.opx_wait(1000)
        new_data = self.client.get_new_data(timestamps_only=True)
        data = []
        PORT_B_BIT_MASK = 32767
        for i in range(new_data.num_timestamps):
            if new_data.source_num_or_type[i] == int(self.source_number):
                # PORT_B_BIT_MASK & new_data.unit[i] masks out the "high bit" that is set on the
                # Port B strobe. This is meant to tell clients and file readers that the strobe came from
                # Port B, but can accidentally be interpreted as a part of the strobed value.
                data.append((new_data.timestamp[i], PORT_B_BIT_MASK & new_data.unit[i]))  
        return data