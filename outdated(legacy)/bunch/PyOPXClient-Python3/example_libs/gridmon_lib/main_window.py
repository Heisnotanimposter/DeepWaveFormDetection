# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from tkinter import BOTH, Button, DISABLED, Frame, Grid, LEFT, PhotoImage, RIGHT, W
from .grid_view import Grid


class MainWindow(Frame):

    def __init__(self, master, number_of_boxes, column=8):
        '''
        The Main Window consists of a frame that contains the button frame and the grid frame. The button frame contains the 
        start/stop button and the settings button. The grid frame contains a Grid object (from the Grid class in grid_view.py).
        This class handles updating the Grid object as well as the logic required in the timer_callback function.

        Args:
            master - typically is Tk()
            number_of_boxes - the number of boxes
            column - the number of columns in the grid (default is 8)

        Returns:
            none

        self.button_frame - the frame that contains the start/stop button and the settings button
        self.grid_frame - the frame that contains the grid
        self.start_stop_button - the start/stop button 
        self.settings_button - the settings button
        self.grid_view - the grid of boxes
        self.timer_callback - function that is called each time timer_function is called. Will be set to an actual function
        in gridmon_controller.
        self.height - the height of the original window
        self.width - the width of the original window
        self.row - the number rows depending on the self.number_of_boxes and self.column 
            ** self.column * self.row = self.number_of_boxes
        self.start_button_str - a base 64 string of the start image that is converted into the format (PhotoImage) that the
        self.start_stop_button requires to display the image
        self.stop_button_str - a base 64 string of the stop image that is converted into the format (PhotoImage) that the 
        self.start_stop_button requires to display the image
        self.start_button_photo - the self.start_button_str as a PhotoImage
        self.stop_button_photo - the self.stop_button_str as a PhotoImage
        '''
        super().__init__(master)

        self.number_of_boxes = number_of_boxes
        self.column = column
        self.button_frame = Frame(self)
        self.grid_frame = Frame(self)
        self.master.title("Grid Monitor")
        self.start_stop_button = Button(self.button_frame)
        self.settings_button = Button(self.button_frame, text="Settings", height=1)
        self.settings_button.pack(side=RIGHT)
        self.start_stop_button.pack(side=LEFT)
        self.button_frame.grid(row=0, column=0, sticky=W)
        self.grid_view = Grid(self.grid_frame, self.number_of_boxes, column=self.column)
        self.grid_frame.grid(row=1, column=0)
        self.timer_callback = lambda: len([1])
        self.pack(fill=BOTH, expand=1)
        self.height = self.winfo_height()
        self.width = self.winfo_width()
        self.row = self.number_of_boxes / self.column

        self.start_button_str = b"""
            R0lGODlhFAAUAHAAACH5BAEAAPwALAAAAAAUABQAhwAAAAAAMwAAZgAAmQAAzAAA/wArAAArMwArZg
            ArmQArzAAr/wBVAABVMwBVZgBVmQBVzABV/wCAAACAMwCAZgCAmQCAzACA/wCqAACqMwCqZgCqmQCqzA
            Cq/wDVAADVMwDVZgDVmQDVzADV/wD/AAD/MwD/ZgD/mQD/zAD//zMAADMAMzMAZjMAmTMAzDMA/zMrAD
            MrMzMrZjMrmTMrzDMr/zNVADNVMzNVZjNVmTNVzDNV/zOAADOAMzOAZjOAmTOAzDOA/zOqADOqMzOqZj
            OqmTOqzDOq/zPVADPVMzPVZjPVmTPVzDPV/zP/ADP/MzP/ZjP/mTP/zDP//2YAAGYAM2YAZmYAmWYAzG
            YA/2YrAGYrM2YrZmYrmWYrzGYr/2ZVAGZVM2ZVZmZVmWZVzGZV/2aAAGaAM2aAZmaAmWaAzGaA/2aqAG
            aqM2aqZmaqmWaqzGaq/2bVAGbVM2bVZmbVmWbVzGbV/2b/AGb/M2b/Zmb/mWb/zGb//5kAAJkAM5kAZp
            kAmZkAzJkA/5krAJkrM5krZpkrmZkrzJkr/5lVAJlVM5lVZplVmZlVzJlV/5mAAJmAM5mAZpmAmZmAzJ
            mA/5mqAJmqM5mqZpmqmZmqzJmq/5nVAJnVM5nVZpnVmZnVzJnV/5n/AJn/M5n/Zpn/mZn/zJn//8wAAM
            wAM8wAZswAmcwAzMwA/8wrAMwrM8wrZswrmcwrzMwr/8xVAMxVM8xVZsxVmcxVzMxV/8yAAMyAM8yAZs
            yAmcyAzMyA/8yqAMyqM8yqZsyqmcyqzMyq/8zVAMzVM8zVZszVmczVzMzV/8z/AMz/M8z/Zsz/mcz/zM
            z///8AAP8AM/8AZv8Amf8AzP8A//8rAP8rM/8rZv8rmf8rzP8r//9VAP9VM/9VZv9Vmf9VzP9V//+AAP
            +AM/+AZv+Amf+AzP+A//+qAP+qM/+qZv+qmf+qzP+q///VAP/VM//VZv/Vmf/VzP/V////AP//M///Zv
            //mf//zP///wAAAAAAAAAAAAAAAAj/APcJHLgvBoCDAAwQXChw0g0AGZQoSUIkAwAxmRjuY8RggpIMRI
            Zo+EhkAgMxC8VIqChyCIgMQ0J+IGIjzcBJDVxWVCIkJAiROxlMEihmwswhQ+oUyQAyJsgkDdAINECkYk
            WBptxk0DAkIkgA+wwM+RDzQ49oBDFosJokAwwAa4FmgLZQGcWtRA4qaQlToxuuSUAcDElk5gS0A+kQ4e
            r0IIi2RNrqE1jH6kyQSgDEgMlVQwZlyjL0+AkyYleFAH6G9Nx1tVORYPcVbbmztE+nKPdNYiCy7V4NXF
            lybZBR4BikcRkn6axEAqOFHHuQBMlVSQ8GjTRmEgNRSeSfAHAMCdU48C3CGBoDAgA7
        """
        self.start_button_photo = PhotoImage(data=self.start_button_str)
        self.stop_button_str = b"""
            iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjw
            v8YQUAAAAJcEhZcwAADsIAAA7CARUoSoAAAAHPSURBVDhPrZW9SyNBHIbfnSSbBGOhIJ57mMLWIPpnCP
            YK0btKK23uOhu1USy0sroDwU+0soig/4Uf3LVyCCbeedwpCjGbTdZ9JzthzY4fRJ9mwjvze8jM7MwYrg
            caqHpR2XYxNvED+byNaNSQueO4sCwTG98yiJkGhFHLg4SE+wdXODq9w9pWHh8643WZgtLL3yV8zloY6E
            thaLDD76nxSLi5W8Dc/Bna22JIJiN+qqdYrODf/zJmpnswOtzlpwHh1k4BSyvnaE1FYPhTcZyq9w+F/K
            0IZiy9vavg62Qa2ZGaVPbkDv9iduEsJEt3J2WraMw4ljWspYMIt+ri+ORWTlPJyPWNg+3VjGwVuow1rK
            WDLmGXXbkB2jUL7b+HJmMtHXSJT+M/vd00/a7moYMucVG4Dy18M9BBl3gPmYKu97P5iOBn8VboEh+7Eo
            ++tWahgy6x/r3XO5u2HzcQPvv6zIMOuoQZM+RB59kMkmqJYHH5l2wVuoywlg66hCEM9Pe1yoMeuCeQSE
            Swl/sjW4UuYw1r6aDr2cvhJZ68HAiDL1Np5Aul0PR1cAzHskbJyAsXrKm9vrgBr7pgFYzs+hNQqksps6
            y4fAJM7wkILw3wAOGSB6voUDhEAAAAAElFTkSuQmCC
        """
        self.stop_button_photo = PhotoImage(data=self.stop_button_str)

        self.start_stop_button.config(image=self.start_button_photo)

    def resize_callback(self, parameters=""):
        '''
        Checks to see if the window has been resized. If the window has been resized, the resize() function is 
        called.

        Arg:
            parameters - a variable which is used to store any argument that might be unintentionally be given.

        Returns:
            None

        tmp_height - the current height of the window
        tmp_width - the current width of the window

        '''
        tmp_height = self.winfo_height()
        tmp_width = self.winfo_width()

        if self.height != tmp_height or self.width != tmp_width:
            self.resize()
        else:
            pass

    def resize(self):
        '''
        Resizes the window by calculating the width and height that each box needs to be changed to.

        Args:
            None

        Returns:
            None

        tmp_box_width - the width of each box at the current window size
        tmp_box_height - the height of each box at the current window size
        self.size - a list containing tmp_box_width and tmp_box_height
        self.size_list - a list of length self.number_of_boxes containing the width and height each box should be changed to 
        '''
        self.width = self.winfo_width()
        self.height = self.winfo_height()
        tmp_box_width = self.width / self.column
        tmp_box_height = (self.height - self.button_frame.winfo_height()) / self.row
        self.size = [tmp_box_width, tmp_box_height]
        self.size_list = []
        for i in range(self.number_of_boxes):
            self.size_list.append(self.size)
        self.change_grid_size(self.size_list)

    def change_number_of_boxes(self, number_of_boxes, column):
        '''
        Changes the number of boxes depending on the number of channels selected.

        Args:
            number_of_boxes - the number of boxes
            column - the number of columns

        Returns:
            none
        '''
        self.number_of_boxes = number_of_boxes
        self.column = column
        self.row = self.number_of_boxes / self.column
        self.grid_view.change_grid_size(number_of_boxes, self.column)
        self.resize()

    def change_grid_size(self, size_list):
        '''
        Changes the grid size.

        Arg:
            size_list - list containing the width and height dimensions each box will be changed to
                ** [[width_1, height_1], [width_2, height_2], [width_3, height_3], [width_4, height_4]]
        Returns:
            None
        '''
        self.grid_view.change_box_sizes(size_list)

    def change_title_name(self, text):
        '''
        Changes the title name.

        Arg:
            text - the text the title is changed too

        Returns:
            none
        '''
        self.master.title(text)

    def set_color_of_boxes(self, color_number_list):
        '''
        Sets the color of each box in the grid depending on its frequency.

        Arg:
            color_numbers - a list of numbers between [0, 1). Each number correlates to a specific box, and the number
            causes the box to be colored on a spectrum from green to red.

        Returns:
            none
        '''
        self.grid_view.change_box_colors(color_number_list)

    def set_channel_labels(self, number_of_channels):
        '''
        Sets the channel label of each box.

        Arg:
            number_of_channels - a list of channel labels for each box. The position of the box correlates to 
            the position of the channel label in the list

        Returns:
            none
        '''
        self.grid_view.set_channel_labels(number_of_channels)

    def enable_channel_labels(self):
        '''
        Enables the channel label of each box.

        Args:
            none

        Returns:
            none
        '''
        self.grid_view.enable_channel_labels()

    def disable_channel_labels(self):
        '''
        Disables the channel label of each box

        Args:   
            none

        Returns:
            none
        '''
        self.grid_view.disable_channel_labels()

    def set_isi_labels(self, isi_labels):
        '''
        Sets the ISI (interspike interval) label of each box.

        Arg:
            isi_labels - a list of isi labels for each box

        Returns:
            none
        '''
        self.grid_view.change_isi_labels(isi_labels)

    def enable_isi_labels(self):
        '''
        Enables the ISI (interspike interval) label of each box.

        Args:
            none

        Returns:
            none
        '''
        self.grid_view.enable_isi_labels()

    def disable_isi_labels(self):
        '''
        Disables the ISI (interspike interval) label of each box.

        Args:
            none

        Returns:
            none
        '''
        self.grid_view.disable_isi_labels()

    def set_start_stop_button_function(self, func):
        '''
        Sets the start_stop_button to call func whenever it is clicked.

        Arg:
            func - the function that will be called whenever the start_stop_button is clicked

        Returns:
            None
        '''
        self.start_stop_button.config(command=func)

    def set_settings_button_function(self, func):
        '''
        Sets the setting_button to call func whenever it is clicked.

        Arg:
            func - the function that will be called whenever the settings_button is clicked

        Returns:
            None
        '''
        self.settings_button.config(command=func)

    def change_start_stop_button_image(self, running):
        '''
        Changes the start_stop_button depending on if running is True or False.

        Arg:
            running - a boolean that represents if the program is collecting data from OmniPlex

        Returns:
            None
        '''
        if running:
            self.start_stop_button.config(image=self.stop_button_photo)
        else:
            self.start_stop_button.config(image=self.start_button_photo)

    def timer_function(self, wait_time=150):
        '''
        Calls the timer_callback function every wait_time milliseconds. 

        Arg:
            wait_time - the wait time in milliseconds

        Returns:    
            none
        '''
        self.wait_time = wait_time
        self.timer_callback()
        self.after(self.wait_time, self.timer_function)
