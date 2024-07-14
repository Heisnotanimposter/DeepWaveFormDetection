# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from tkinter import Frame
from .box_view import Box


class Grid(Frame):

    def __init__(self, master, number_of_boxes, column=8):
        '''
        Grid is a container that holds, manages, and controls initializations of the Box
        class.

        Args:
            self.master - should be equal to Tk()
            self.row - the number of rows in the grid
            self.column - the number of columns in the grid

        Returns:
            none
        '''
        super().__init__(master)
        self.master = master
        self.number_of_boxes = number_of_boxes
        self.column = column
        self.box_list = []
        self.row = self.number_of_boxes / self.column
        self.default_text = '0'
        for i in range(int(self.row)):
            for j in range(self.column):
                self.box_list.append(Box(self.master, column=j, row=i, isi_text=self.default_text))

    def change_grid_size(self, number_of_boxes, column):
        '''
        Changes the number of boxes depending on the number of columns and rows in the grid.

        Args:
            self.column - the number of columns 
            self.row - the number of rows

        Returns:
            none
        '''

        # causes the program to forget the existing grid
        for b in self.box_list:
            b.grid_forget()
        self.number_of_boxes = number_of_boxes
        self.column = column
        self.box_list = []
        self.row = self.number_of_boxes / self.column

        # creates a new grid depending on the number of columns and rows
        for i in range(int(self.row)):
            for j in range(self.column):
                self.box_list.append(Box(self.master, column=j, row=i, isi_text='0'))

    def change_box_sizes(self, size_list):
        '''
        Sets the size of each box in box_list depending on how the grid has been stretched. 

        Arg:
            size_list - a list of lists that contain the height and width with which to change the boxes
                ex: [[width_1, height_1], [width_2, height_2], [width_3, height_3]]

        Returns:
            none
        '''
        for i in range(len(size_list)):
            self.box_list[i].change_size(size_list[i][0], size_list[i][1])

    def change_box_colors(self, color_number_list):
        '''
        Changes the color of each box in box_list according to color_list.

        Arg:
            color_number_list - a list containing the colors of each box in color_number_list

        Returns:
            none
        '''
        for i in range(len(color_number_list)):
            try:
                self.box_list[i].set_color(color_number_list[i])
            except:
                print("IndexError in grid_view.set_colors(); index is {}".format(i))

    def set_channel_labels(self, channel_names_list):
        '''
        Sets the channel_label of each box in box_list according to channel_names_list.

        Arg:
            channel_names_list - a list containing the text for each channel_label of each box in box_list

        Returns:
            none
        '''
        for i in range(len(channel_names_list)):
            self.box_list[i].set_channel_label(channel_names_list[i])

    def disable_channel_labels(self):
        '''
        Disables the channel_label of each box in box_list.

        Args:
            none

        Returns:
            none
        '''
        for i in range(len(self.box_list)):
            self.box_list[i].disable_channel_label()

    def enable_channel_labels(self):
        '''
        Enables the channel_label of each box in box_list.

        Args:
            none

        Returns:
            none
        '''
        for i in range(len(self.box_list)):
            self.box_list[i].enable_channel_label()

    def change_isi_labels(self, isi_labels_list):
        '''
        Changes the isi_label of each box in box_list according to isi_labels_list.

        Arg:
            isi_labels_list - a list containing the text for each isi_label of each box in box_list

        Returns:
            none
        '''
        for i in range(len(isi_labels_list)):
            self.box_list[i].change_isi_label(isi_labels_list[i])

    def disable_isi_labels(self):
        '''
        Disables the isi_label of each box in box_list.

        Args:
            none

        Returns:
            none
        '''
        for i in range(len(self.box_list)):
            self.box_list[i].disable_isi_label()

    def enable_isi_labels(self):
        '''
        Enables the isi_lable of each box in box_list.

        Args:
            none

        Returns:
            none
        '''
        for i in range(len(self.box_list)):
            self.box_list[i].enable_isi_label()
