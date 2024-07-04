# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from tkinter import Tk, Frame, Button, Label, LabelFrame, Radiobutton, StringVar, IntVar, W, Entry, X, E, Checkbutton


class SettingsView(Frame):
    def __init__(self, master):
        '''
        SettingsView holds widgets and methods for displaying and controlling the settings view.

        Args:
            master - Tk root instance

        Returns:
            none
        '''
        super().__init__(master)
        self.master = master

        # Frames for the option sets and buttons
        self.channel_settings_frame = LabelFrame(self.master, text='Channel Count')
        self.isi_settings_frame = Frame(self.master)
        self.display_options_frame = LabelFrame(self.master, text="Display Options")
        self.ok_cancel_frame = Frame(self.master)

        # Channel settings widgets and value storage
        self.channel_settings_radiobutton_value = IntVar()

        self.radio_16 = Radiobutton(self.channel_settings_frame, text='16', variable=self.channel_settings_radiobutton_value, value=16)
        self.radio_32 = Radiobutton(self.channel_settings_frame, text='32', variable=self.channel_settings_radiobutton_value, value=32)

        self.radio_16.grid()
        self.radio_32.grid()

        # ISI settings widgets and value storage
        self.max_isi_entry_text = StringVar()
        self.seconds_to_average_entry_text = StringVar()
        self.max_isi_label = Label(self.isi_settings_frame, text="Max ISI Value (Hz):     ")
        self.max_isi_entry = Entry(self.isi_settings_frame, width=6, textvariable=self.max_isi_entry_text)
        self.seconds_to_average_label = Label(self.isi_settings_frame, text="Seconds to Average (S):     ")
        self.seconds_to_average_entry = Entry(self.isi_settings_frame, width=6, textvariable=self.seconds_to_average_entry_text)
        self.max_isi_label.grid(row=0, column=0, sticky=W)
        self.max_isi_entry.grid(row=0, column=1, sticky=W)
        self.seconds_to_average_label.grid(row=1, column=0)
        self.seconds_to_average_entry.grid(row=1, column=1)

        # Display options widgets and value storage
        self.channel_names_option_var = IntVar()
        self.channel_names_option = Checkbutton(self.display_options_frame, text='Show Channel Names', variable=self.channel_names_option_var)
        self.show_isi_text_option_var = IntVar()
        self.show_isi_text_option = Checkbutton(self.display_options_frame, text='Show ISI Value', variable=self.show_isi_text_option_var)
        self.channel_names_option.grid(sticky=W)
        self.show_isi_text_option.grid(sticky=W)

        # OK/Cancel buttons
        self.ok_button = Button(self.ok_cancel_frame, text="OK", command=self.ok_button)
        self.cancel_button = Button(self.ok_cancel_frame, text="Cancel", command=self.cancel_button)
        self.ok_button.grid(row=0, column=0)
        self.cancel_button.grid(row=0, column=1)

        # Packing of frames and self
        self.channel_settings_frame.pack(fill=X, pady=5, padx=5)
        self.isi_settings_frame.pack(pady=5, padx=5)
        self.display_options_frame.pack(pady=5, padx=5, fill=X)
        self.ok_cancel_frame.pack(pady=5, padx=5)
        self.pack()

        # Set default settings
        self.set_default_settings()

        # These functions will be overridden by the GridMonSettings class
        self.apply_settings = lambda: len([1])
        self.save_settings = lambda x: len([1])

    def set_settings(self, number_of_channels, max_isi, max_timestamp_age, channel_labels_enabled, isi_labels_enabled):
        '''
        Set the fields in this dialog box. Don't confuse this with setting anything
        other than what's in this view.

        Args:
            number_of_channels - number of channels
            max_isi - maximum ISI value (what would be shown as red)
            max_timestamp_age - max age a timestamp can be before it gets removed from the buffer
            channeL_labels_enabled - 1 or 0, for displaying channel labels
            isi_labels_enabled - 1 or 0, for displaying ISI labels

        Returns:
            None
        '''
        # Set channel number
        if number_of_channels == 32:
            self.radio_32.select()
        elif number_of_channels == 16:
            self.radio_16.select()

        # Set max ISI value
        self.max_isi_entry_text.set(str(max_isi))
        # Set max timestamp age (the amount of seconds to average)
        self.seconds_to_average_entry_text.set(str(max_timestamp_age))

        if channel_labels_enabled:
            self.channel_names_option.select()
        else:
            self.channel_names_option.deselect()

        if isi_labels_enabled:
            self.show_isi_text_option.select()
        else:
            self.show_isi_text_option.deselect()

    def set_default_settings(self):
        '''
        Sets default settings if the settings file doesn't exist.

        Args:
            None

        Returns:
            None
        '''
        self.channel_settings_radiobutton_value.set(16)
        self.max_isi_entry_text.set('30')
        self.seconds_to_average_entry_text.set('10')
        self.channel_names_option_var.set(1)
        self.show_isi_text_option_var.set(1)

    def ok_button(self):
        '''
        When the OK button is clicked, the settings are stored to disk.

        Args:
            None

        Returns:
            None
        '''
        num_channels = self.channel_settings_radiobutton_value.get()
        max_isi = int(self.max_isi_entry_text.get())
        max_age = int(self.seconds_to_average_entry_text.get())
        channel_labels_enabled = self.channel_names_option_var.get()
        isi_text_enabled = self.show_isi_text_option_var.get()

        self.save_settings((num_channels, max_isi, max_age, channel_labels_enabled, isi_text_enabled))
        self.apply_settings()
        self.master.destroy()

    def cancel_button(self):
        '''
        Closes the settings view.

        Args:
            None

        Returns:
            None
        '''
        self.master.destroy()
