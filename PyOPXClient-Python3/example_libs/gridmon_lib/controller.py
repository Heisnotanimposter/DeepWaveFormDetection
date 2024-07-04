# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from tkinter import Tk, Toplevel
from .main_window import MainWindow
from .client import GridMonClient
from .settings_window import SettingsView
from .settings_controller import GridMonSettings


class Controller():

    def __init__(self, omniplex_demo_configuration=False):
        '''
        The controller connects all the classes that are part of GridMon. MainWindow, GridMonClient, SettingsView,
        and GridMonSettings are instanced in the Controller. The controller allows the settings window to pop up and 
        for the settings chosen in the settings window to be saved and for those settings to be applied to the grid. 
        The grid is updated using data from OmniPlex which is configured to find the average interspike interval frequency,
        and the grid displays the channel names according to OmniPlex. 

        Args:
            none

        Returns:
            none

        root - Tk()
        '''
        self.running = False
        self.omniplex_demo_configuration = omniplex_demo_configuration
        self.blank_text = ""
        self.root = Tk()
        self.apply_settings(True)

    def run(self):
        '''
        Runs the mainloop.

        Args:
            none

        Returns:
            none
        '''
        self.root.mainloop()

    def start_stop_button_clicked(self):
        '''
        Occurs when the start_stop_button is clicked. 
        If the 'start' button is clicked the program is connected to OmniPlex and is able to recieve data. The image 
        on the button changes to the 'stop' image, and the timer_callback is changed to self.master_update. 
        If the 'stop' button is clicked, the program is disconnected from OmniPlex and no longer recieves data. The 
        image on the button changes to the 'start' image, and the timer_callback is changed to a placeholder function.  

        Args:
            none

        Returns:
            none
        '''

        # if self.running is False then the image on the start_stop button changes from the play button to the pause
        # button, the self.window.timer_callback function is set to self.master_update, and the client is connected
        if not self.running:
            self.running = True
            self.window.change_start_stop_button_image(self.running)
            # sets the timer_callback function to self.master_update
            self.window.timer_callback = self.master_update
            self.client.connect()
        # otherwise the image on the start_stop button changes from the pause button to the play button, the function
        # is set to a fake function, and the client is disconnected
        else:
            self.running = False
            self.window.change_start_stop_button_image(self.running)
            # changes the timer_callback function to a 'function' that doesn't do anything
            self.window.timer_callback = lambda: len([1])
            self.client.disconnect()

    def settings_button_clicked(self):
        '''
        Occurs when the setting_button is clicked. The settings window pops up and the options chosen in the settings 
        window are set. The settings are saved to the disk, so when the program is reopened the previous settings are
        still in place. 

        Args:
            none

        Returns:
            none

        top - the Toplevel() allows a pop-up window to open
        settings_view - instantiates the SettingsView class in a variable
        '''
        top = Toplevel()
        settings_view = SettingsView(top)
        # Set the values in the settings dialog box here
        settings_view.set_settings(self.number_of_channels,
                                   self.max_isi_hz,
                                   self.max_timestamp_age,
                                   self.channel_labels_enabled,
                                   self.isi_text_labels_enabled)
        
        
        # Sets the settings_view callbacks to real functions
        settings_view.apply_settings = self.apply_settings
        settings_view.save_settings = self.save_settings
        settings_view.pack()

    def save_settings(self, settings):
        '''
        Saves the settings that the user chooses in a list. (check on this)

        Arg:
            settings - the settings chosen by the user

        Returns:
            none
        '''
        self.settings_controller.set_settings(*settings)

    def apply_settings(self, first_run=False):
        '''
        Applies the settings to the grid monitor. Does certain tasks only during the first run of the function while
        other tasks occur on each run of the function except the first. Additionally, this function contains tasks 
        that are run regardless of which run it is.

        Arg:
            first_run - boolean that determines if this is the first run of the function

        Returns:
            none

        self.gridmon_settings - gets the preset settings found in get_settings() from the settings_controller
        self.number_of_channels - the number_of_channels the grid monitor is showing that are receiving data
        self.max_isi_hz - the max inter-spike-interval frequency (user can change)
        self.max_timestamp_age - the max time stamp age (user can't change)
        self.channel_labels_enabled - boolean that determines whether the channel labels are enabled (user can change)
        self.isi_text_labels_enabled - boolean that determines whether the channel labels are enabled (user can change)
        self.unit_list - the list of units for each box        
        self.max_number_timestamps - max number of timestamps
        self.column - the number of columns in the grid
        self.channel_name_list - the list of channel names for each box
        self.window - instantiation of the MainWindow class
        self.client - instantiation of the GridMonClient class
        '''
        if (first_run):
            # instantiates GridMonSettings() in the variable settings_controller in the first run of the apply_settings
            self.settings_controller = GridMonSettings()

        # sets the current settings to the settings set in the settings menu
        self.gridmon_settings = self.settings_controller.get_settings()

        # sets the variables set in the settings window (which are found in gridmon settings) to variables to be used in
        # the controller class. This is done each time the settings is changed to account for any changes made.
        self.number_of_channels = self.gridmon_settings['Channel Count']
        self.max_isi_hz = self.gridmon_settings['Max ISI Value']
        self.max_timestamp_age = self.gridmon_settings['Max Timestamp Age']
        self.channel_labels_enabled = self.gridmon_settings['Channel Labels Enabled']
        self.isi_text_labels_enabled = self.gridmon_settings['ISI Labels Enabled']
        self.unit_list = self.gridmon_settings['Units']

        self.max_number_timestamps = self.gridmon_settings['Max Number of Timestamps']
        self.column = self.gridmon_settings['Columns']

        self.channel_name_list = []
        if (first_run):
            # initializes the classes that allow GridMon to exist

            if self.number_of_channels == 32:
                self.column = 8
            elif self.number_of_channels == 16:
                self.column = 4
            # initializes the Main Window class in the variable self.window
            self.window = MainWindow(self.root, self.number_of_channels, self.column)
            # sets the function for the start_stop_button to self.start_stop_button_clicked
            self.window.set_start_stop_button_function(self.start_stop_button_clicked)
            # sets the function for the settings_button to self.settings_button_clicked
            self.window.set_settings_button_function(self.settings_button_clicked)
            # initializes GridMonClient in the variable self.client
            self.client = GridMonClient(self.number_of_channels,
                                        self.unit_list,
                                        self.max_number_timestamps,
                                        self.max_timestamp_age,
                                        omniplex_demo_configuration=self.omniplex_demo_configuration)
            # starts the timer
            self.start_timer()
            # binds the root so that whenever the window is moved or resized the program checks to see if the window has
            # been resized if the window has been changed, the grid is resized according to the new width and height
            self.root.bind("<Configure>", self.window.resize_callback)

        else:
            # changes GridMon based on the options selected in the Settings Menu
            if self.number_of_channels == 32:
                self.column = 8
            elif self.number_of_channels == 16:
                self.column = 4
            self.window.change_number_of_boxes(self.number_of_channels, self.column)
            self.client.change_number_of_channels(self.number_of_channels)
            self.client.change_max_timestamp_age(self.max_timestamp_age)
            self.client.change_max_isi_hz(self.max_isi_hz)

        # enables or disables the channel labels and isi (interspike interval) labels based on if the user selected to do so
        # in the Settings Menu
        if self.channel_labels_enabled == 0:
            self.window.disable_channel_labels()
        else:
            self.window.enable_channel_labels()

        if self.isi_text_labels_enabled == 0:
            self.window.disable_isi_labels()
        else:
            self.window.enable_isi_labels()

        # connects the client to allow each channel name to be set
        # if previously disconnected the client is disconnected this is determined by self.running

        self.client.connect()
        for i in range(self.number_of_channels):
            # channel number to be added to the channel_name_list
            tmp_channel_number = i + 1
            self.channel_name_list.append(self.client.get_channel_name(tmp_channel_number))

        self.window.set_channel_labels(self.channel_name_list)

        # disconnects the client if previously disconnected
        if self.running:
            pass
        else:
            self.client.disconnect()

    def master_update(self):
        '''
        Updates the grid monitor by updating the color of the box depending on the normalized average isi (interspike interval) 
        frequency.

        Args:
            none

        Returns:
            none

        normalized_isi_hz - a list of the normalized average isi (interspike interval) frequencies 
            ** normalized means avg_isi_hz / max_isi_hz
        avg_isi_hz - a list of the average isi (interspike interval) frequencies
        '''
        self.client.update()

        normalized_isi_hz_list = self.client.get_normalized_average_isi_frequency()
        self.window.set_color_of_boxes(normalized_isi_hz_list)
        avg_isi_hz = self.client.get_average_isi_frequency()
        self.window.set_isi_labels(avg_isi_hz)

    def start_timer(self):
        '''
        Starts the timer_function from the main_window class.

        Args:
            none

        Returns:
            none
        '''
        self.window.timer_function()
