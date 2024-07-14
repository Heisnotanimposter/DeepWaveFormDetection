# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from pyopxclient import PyOPXClientAPI
from collections import deque


class ChannelBuffer():
    '''
    ChannelBuffer extracts data for one channel/unit, buffers it, and performs ISI (interspike interval) calculations.

    Create an instance of this, then pass it data to process with the Update method.
    '''

    def __init__(self, channel, unit, max_buffer_timestamps, max_timestamp_age, source_id=6):
        '''
        Args:
            channel - the channel number
            unit - the unit number
            source_id - the id number of the source (most likely 6)
            nax_buffer_timestamps - max number of timestamps in the buffer
            max_timestamp_age - the max age of the timestamps
            source_id - the source number (defaults to 6)

        Returns:
            none

        Class Vars:
            self.avg_isi - average interspike interval
            self.avg_isi_frequency - average interspike interval frequency
            self.buffer - a double ended queue (deque) that holds the timestamps
        '''
        self.source_id = source_id
        self.channel = channel
        self.unit = unit
        self.max_buffer_timestamps = max_buffer_timestamps
        self.max_timestamp_age = max_timestamp_age
        self.avg_isi = 0
        self.avg_isi_frequency = 0
        self.buffer = deque(maxlen=self.max_buffer_timestamps)

    def set_source_id(self, source_id):
        self.source_id = source_id

    def Update(self, new_data, current_time):
        '''
        Updates the buffered data, and recalculates the interspike interval and frequency to account for the spikes that have 
        been added and have aged out.

        Args:
            new_data - This is the returned data from running get_new_data() from an instance of PyOPXClientAPI
            current_time - Current time in OmniPlex (used to "age out" old spikes)

        Return:
            self.avg_isi_frequency - average interspike interval frequency
        '''
        # Add new spike timestamps
        for i in range(new_data.num_timestamps):
            if new_data.source_num_or_type[i] == self.source_id:
                if new_data.channel[i] == self.channel and new_data.unit[i] == self.unit:
                    self.buffer.append(new_data.timestamp[i])

        # Age out old spikes
        if len(self.buffer) > 0:
            while (current_time - self.buffer[0]) > self.max_timestamp_age:
                self.buffer.popleft()
                if len(self.buffer) < 1:
                    break

        # Calculate ISI (interspike interval)
        tmp_ts = list(self.buffer)
        if len(tmp_ts) > 1:
            diffs = [t - s for s, t in zip(tmp_ts, tmp_ts[1:])]
            self.avg_isi = sum(diffs) / len(tmp_ts)
        else:
            self.avg_isi = 0

        # Calculate ISI in Hz
        if self.avg_isi > 0:
            self.avg_isi_frequency = 1.0 / self.avg_isi
        else:
            self.avg_isi_frequency = 0
        # returns the ISI frequency over the interval
        return self.avg_isi_frequency


class ChannelsSpikeBuffer():

    def __init__(self, number_of_channels, unit_list, max_buffer_timestamps, max_timestamp_age, source_id=6, channels_list=[]):
        '''
        Args:
            number_of_channels - the number of channels
            source_id - the id of the source (probably 6)
            unit_list - a list of the units the length of which is equal to num_channels
            max_timestamps - the max amount of timestamps
            max_age - the max-age
            channels_list - a list of the channel numbers (optional)

        Returns:
            none

            self.channels_spike_buffers

        '''
        self.number_of_channels = number_of_channels
        self.channels_spike_buffers = []
        self.unit_list = unit_list
        self.max_buffer_timestamps = max_buffer_timestamps
        self.max_timestamp_age = max_timestamp_age
        self.source_id = source_id
        self.channels_list = channels_list
        if len(channels_list) < 1:
            self.channels_list = [0] * number_of_channels

        for i in range(number_of_channels):
            if self.channels_list[i] == 0:
                self.channels_list[i] = i + 1
            try:
                self.channels_spike_buffers.append(ChannelBuffer(self.channels_list[i],
                                                                 self.unit_list[i],
                                                                 self.max_buffer_timestamps,
                                                                 self.max_timestamp_age,
                                                                 self.source_id))
            except IndexError:
                print("INDEXERROR IN CLIENT: i is {}".format(i))

    def change_number_of_channels(self, number_of_channels, channels_list=[]):
        '''
        Changes the number of channels to the number_of_channels given and sets channels_list equal to 1 through the number of 
        channels.

        Arg:
            number_of_channels - number of channels

        Returns:
            none

            channels_spike_buffers - a list of ChannelBuffers number_of_channels long
            channels_list - a list the length of number_of_channels
        '''
        self.number_of_channels = number_of_channels
        self.channels_spike_buffers = []

        if len(channels_list) == self.number_of_channels:
            self.channels_list = channels_list
        else:
            self.channels_list = [0] * self.number_of_channels
            for i in range(self.number_of_channels):
                if self.channels_list[i] == 0:
                    self.channels_list[i] = i + 1
                try:
                    self.channels_spike_buffers.append(ChannelBuffer(self.channels_list[i],
                                                                     self.unit_list[i],
                                                                     self.max_buffer_timestamps,
                                                                     self.max_timestamp_age,
                                                                     self.source_id))
                except IndexError:
                    print("INDEXERROR IN CLIENT: i is {}".format(i))

    def set_source_id(self, source_id):
        for b in self.channels_spike_buffers:
            b.set_source_id(source_id)

    def change_max_timestamp_age(self, max_timestamp_age):
        '''
        Changes the max time stamp age to max_timestamp_age.

        Arg:
            max_timestamp_age - max time stamp age

        Returns:
            none
        '''
        self.max_timestamp_age = max_timestamp_age

    def Update(self, new_data, current_time):
        '''
        Returns the average frequency based on the frequency calculated in the Update function of the Channel Buffer class of each 
        channel in the list.

        Args:
            new_data - new spike data
            current_time - the current time ## double check

        Return:
            list_avg_hz - a list of each channel's average frequency
        '''
        list_avg_hz = []

        for i in range(len(self.channels_spike_buffers)):
            list_avg_hz.append(self.channels_spike_buffers[i].Update(new_data, current_time))
        return list_avg_hz


class GridMonClient():

    def __init__(self, number_of_channels, unit_list, max_buffer_timestamps, max_timestamp_age, channels_list=[], source_id=6, omniplex_demo_configuration=False):
        '''
        Initializes the PyOPXClientAPI in self.client, and it contains functions and classes for gathering data from OmniPlex.
        Args:
            num_channels - the number of channels
            unit_list - a list of the unit numbers
            max_timestamps - the max amount of timestamps
            max_age - the max-age of the spikes
            channels_list - a list of the channel numbers (optional)
            source_id - the id of the source (probably 6)

        Returns:
            none

        self.max_opx_data - the maximum number of data blocks that can be retrieved in one query to OmniPlex Server. Setting
        this too high may result in too much memory being consumed by the client. Only change the default value if you're 
        polling so infrequently that you're always getting the maximum number returned
        '''
        self.client = PyOPXClientAPI()
        self.number_of_channels = number_of_channels
        self.unit_list = unit_list
        self.max_buffer_timestamps = max_buffer_timestamps
        self.max_timestamp_age = max_timestamp_age
        self.channels_list = channels_list
        self.max_opx_data = 100000
        self.client = PyOPXClientAPI(max_opx_data=self.max_opx_data)
        self.source_id = source_id
        self.omniplex_demo_configuration = omniplex_demo_configuration
        self.max_isi_hz = 30
        self.avg_hz_list = []
        self.channel_buffers = ChannelsSpikeBuffer(self.number_of_channels,
                                                   self.unit_list,
                                                   self.max_buffer_timestamps,
                                                   self.max_timestamp_age,
                                                   self.source_id,
                                                   self.channels_list)

    def change_number_of_channels(self, number_of_channels):
        '''
        Changes the number of channels to number_of_channels.

        Arg:
            number_of_channels - the number of channels to be changed to

        Returns:
            none
        '''
        self.number_of_channels = number_of_channels
        self.channel_buffers.change_number_of_channels(self.number_of_channels)

    def change_max_timestamp_age(self, max_timestamp_age):
        '''
        Changes the max timestamp age.

        Arg:
            max_timestamp_age - the max timestamp age to be changed to

        Returns:    
            none
        '''
        self.max_timestamp_age = max_timestamp_age
        self.channel_buffers.change_max_timestamp_age(self.max_timestamp_age)

    def change_max_isi_hz(self, max_isi_hz):
        '''
        Changes the max ISI (interspike interval) frequency.

        Arg:
            max_isi_hz - the max ISI frequency to be changed to

        Returns:
            none
        '''
        self.max_isi_hz = max_isi_hz

    def connect(self):
        '''
        Starts data acquisition from OmniPlex.

        Args:
            none 

        Returns:
            none
        '''
        self.client.connect()
        r = self.client.opx_client.get_opx_system_status()
        if r != 2:
            pass

        global_parameters = self.client.get_global_parameters()

        for source_id in global_parameters.source_ids:
            source_name, _, _, _ = self.client.get_source_info(source_id)
            if source_name == 'SPK':
                self.source_id = source_id

        self.client.opx_client.clear_data(1000)

    def disconnect(self):
        '''
        Stops data acquisition from OmniPlex.

        Args:
            none

        Returns:
            none
        '''
        self.client.disconnect()

    def update(self):
        '''
        Updates the data from OmniPlex and causes the program to wait to allow spike data to accumulate in OmniPlex.

        Args:
            none

        Returns:
            none

        WAITIME - waits for 1000 milliseconds to collect data
        '''
        WAITTIME = 1000
        self.client.opx_wait(WAITTIME)

        new_data = self.client.get_new_data(timestamps_only=True)

        if new_data.num_timestamps == 0:
            pass

        last_wait_event_time = self.client.opx_client.get_last_wait_event_time()

        if self.omniplex_demo_configuration:
            last_wait_event_time /= 40000.0

        self.avg_hz_list = self.channel_buffers.Update(new_data, last_wait_event_time)

    def get_average_isi_frequency(self, decimal_length=5):
        '''
        Gets a list of the average frequency for each channel rounded to the fifth decimal point

        Args:
            none

        Return:
            final_avg_hz_list - list containing the average frequency for each channel. The average frequency is determined
            over a specified time period and includes a specified amount of spikes

        num_length - the number that allows the average frequency to round to the decimal_length place 
        '''
        final_avg_hz_list = []
        num_length = 10 ** decimal_length
        for i in range(len(self.avg_hz_list)):
            final_avg_hz_list.append(round(self.avg_hz_list[i] * num_length) / num_length)
        return final_avg_hz_list

    def get_normalized_average_isi_frequency(self):
        '''
        Finds the normalized average frequency. Normalized means average frequency/max frequency.

        Args:
            max_isi_hz - max frequency

        Return:
            list_nhz - list of the normalized average frequency for each corresponding channel
        '''
        list_nhz = []
        for i in range(len(self.avg_hz_list)):
            temp = self.avg_hz_list[i] / self.max_isi_hz
            list_nhz.append(temp)
        return list_nhz

    def get_channel_name(self, channel_number):
        '''
        Gets the channel name based on the channel number.
        
        Arg:
            channel_number - the number of the channel
        
        Return:
            chan_name - the name of the channel

        result
        chan_name - see above
        rate
        voltage_scaler
        enabled
        '''
        result, chan_name, rate, voltage_scaler, enabled = self.client.opx_client.get_source_chan_info_by_number(self.source_id, channel_number)
        return chan_name
