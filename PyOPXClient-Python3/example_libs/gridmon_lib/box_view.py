# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from colorsys import hsv_to_rgb
from tkinter import CENTER, Frame, Label


class Box(Frame):

    def __init__(self, master, column, row, height=70, width=70, isi_text="", channel_text="", highlightthickness=1):
        '''
        Initializes a frame with two labels and places them.

        Args:
            master - Tk()
            column - the column position of the box
            row - the row position of the box
            height - the height of the box
            width - the width of the box
            isi_text_text - the initial text of the frequency_label
            channel_text - the initial text of the channel_label
            highlightthickness - the thickness of the border surrounding the frame

        Returns:
            none

        self.channel_label - label that shows the channel number
        self.channel_label_enabled - boolean that determines if the channel_label shows the channel label
        self.isi_text_label - label that shows the frequency
        self.isi_label_enabled - boolean that determines if the isi_label shows the ISI (interspike interval) label
        '''
        super().__init__(master)
        self.column = column
        self.row = row
        self.height = height
        self.width = width
        self.highlightthickness = highlightthickness
        self.no_channel_text = ""
        self.no_isi_text = ""
        self.config(height=self.height, width=self.width, highlightthickness=highlightthickness, highlightbackground='black')
        self.grid(row=row + 1, column=column)
        self.channel_label = Label(self, text=channel_text)
        self.channel_label.place(x=0, y=0)
        self.isi_label = Label(self, text=isi_text)
        self.isi_label.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.channel_label_enabled = True
        self.isi_label_enabled = True

    def set_color(self, number):
        '''
        Sets the color of the frame and the background of the labels to a color on the spectrum between red and green.
        The color is determined by num with 0 being green and red being very close to 1.

        Arg:
            num - a value between 0 and 1 ([0, 1)) that represents the normalized isi value

        Returns:
            none

        rgb - a tuple that contains the red, green, and blue numbers necessary to make the color
        color - the color that the background of the box will be
        '''
        if number > 1 or number < 0:
            pass
        else:
            number = 1 - number
            # the num / 3 allows the color to be on the green to red spectrum
            rgb = tuple(i * 255 for i in hsv_to_rgb(number / 3, 1, 1))
            # sets the color in rgb form where a numerical value is given to the red, green, and blue value which when combined causes
            # the desired color
            color = '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            # changes the background of the box, channel_label, and isi_label to the desired color
            self.config(bg=color)
            self.channel_label.config(bg=color)
            self.isi_label.config(bg=color)

    def change_width(self, width):
        '''
        Sets the width of the frame.

        Arg:
            width - the width in pixels

        Returns:
            none
        '''
        self.width = width
        self.config(width=self.width)

    def change_height(self, height):
        '''
        Sets the height of the frame.

        Arg:
            height - the height in pixels

        Return:
            none
        '''
        self.height = height
        self.config(height=self.height)

    def change_size(self, width, height):
        '''
        Changes the size of the frame.

        Args:
            height - the height in pixels
            width - the width in pixels

        Returns:
            none
        '''
        self.change_width(width)
        self.change_height(height)

    def set_channel_label(self, channel_text):
        '''
        Sets the text on the channel_label.

        Arg:
            channel_text - text to be written on the channel_label

        Returns:
            none
        '''
        if self.channel_label_enabled == True:
            self.channel_label.config(text=channel_text)
        else:
            self.channel_label.config(text=self.no_channel_text)

    def enable_channel_label(self):
        '''
        Enables the text on the channel_label.

        Args:
            none

        Returns:
            none
        '''
        self.channel_label_enabled = True

    def disable_channel_label(self):
        '''
        Disables the text on the channel_label.

        Args:
            none

        Returns:
            none
        '''
        self.channel_label_enabled = False

    def change_isi_label(self, isi_text):
        '''
        Changes the text on the isi_text_label.

        Arg:  
            isi_text - the text to be written on the isi_text_label

        Returns:
            none
        '''
        if self.isi_label_enabled:
            self.isi_label.config(text=isi_text)
        else:
            self.isi_label.config(text=self.no_isi_text)

    def enable_isi_label(self):
        '''
        Enables the text on the isi_label.

        Args:
            none

        Return:
            none
        '''
        self.isi_label_enabled = True

    def disable_isi_label(self):
        '''
        Disables the text on the isi_text_label.

        Args:
            none

        Returns:
            none
        '''
        self.isi_label_enabled = False
