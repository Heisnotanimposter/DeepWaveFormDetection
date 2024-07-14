# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

import json
import sys
import os
from collections import namedtuple


class GridMonSettings():
    '''
    GridMonSettings is a controller class that handles recalling and storing GridMon settings to disk.
    '''

    def __init__(self):
        # Set up default settings. If a settings file exists, these
        # will be overwritten. If it doesn't exist, a new settings file
        # will be created with these default settings.

        # Python dict for storing settings in a human readable key/value pair
        self.settings = {}
        self.settings['Channel Count'] = 32
        self.settings['Start Channel'] = 1
        self.settings['End Channel'] = 32
        self.settings['Max ISI Value'] = 30
        self.settings['Max Timestamp Age'] = 10
        self.settings['Max Number of Timestamps'] = 50
        self.settings['Channel Labels Enabled'] = True
        self.settings['ISI Labels Enabled'] = True
        self.settings['Units'] = [1] * self.settings['Channel Count']
        self.settings['Rows'] = 8
        self.settings['Columns'] = 4

        if self.read_settings_from_disk() == True:
            # If read_settings_from_disk() returns True, the settings were loaded.
            pass
        else:
            # If not, try to write the default settings to disk.
            if self.write_settings_to_disk() == True:
                pass
            else:
                from tkinter import messagebox
                messagebox.showerror("Settings Error", "Couldn't write settings to disk!\nSettings won't be saved.")

    def read_settings_from_disk(self):
        try:
            with open("settings.json") as json_file:
                self.settings = json.load(json_file)
        except Exception as e:
            # This exception is triggered if there isn't a settings file already created. GridMon will attempt to create one.
            return False
        else:
            return True

    def write_settings_to_disk(self):
        json_string = json.dumps(self.settings, indent=4)
        try:
            with open("settings.json", "w") as json_file: json_file.write(json_string)
        except Exception as e:
            # This exception is triggered if GridMon doesn't have permission to save to the local disk. This isn't a fatal error,
            # it just means the settings will be back to default the next time GridMon is run.
            return False
        else:
            return True

    def get_settings(self):
        '''
        Gets the settings.
        Args:
            none
        Return:
            self.settings - the settings
        '''
        return self.settings

    def set_settings(self, channel_count, max_isi_value, max_timestamp_age, channel_labels_enabled, isi_labels_enabled):
        '''
        Stores settings and writes settings to disk.
        Args:
            channel_count - number of channels
            max_isi_value - max isi (interspike interval) frequency in Hz
            max_spike_age - age when a spike is removed from the buffer
            channel_labels_enabled - true or false
            isi_labels_enabled - true or false
        Returns:
            none
        '''
        pass
        self.settings['Channel Count'] = channel_count
        self.settings['Max ISI Value'] = max_isi_value
        self.settings['Max Timestamp Age'] = max_timestamp_age
        self.settings['Channel Labels Enabled'] = channel_labels_enabled
        self.settings['ISI Labels Enabled'] = isi_labels_enabled

        self.write_settings_to_disk()
