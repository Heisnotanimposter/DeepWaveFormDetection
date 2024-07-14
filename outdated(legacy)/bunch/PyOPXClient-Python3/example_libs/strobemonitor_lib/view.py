# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from tkinter import BOTTOM, Button, DISABLED, Frame, END, Entry, Label, LEFT, NORMAL, RIGHT, Scrollbar, Text, Y 


class View():
    """
    The View class contains the functions and variables necessary to create the view window.
    
    Args:
        root - a Tk instance
        WIDTH - width of the text_box
        HEIGTH - height of the text_box
    
    Returns:
        None
   
    self.timer_callback - function to be determined in the Controller class
    self.button_container_fram - the frame the contains the buttons
    self.text_view_frame - the frame that contains the text box and the scroll
    self.connect_disconnect_button - the button with text that either reads connect or disconnect
    quit_button - the button that allows the user to quit the program
    self.text_label - the label displays the text "Source Number" telling the user that the entry_box to the left
    is where they may type their desired source number
    self.entry_box - box to type in the desired source number
    self.enter_button - allows the program to change the source number to the contents in the entry_box
    self.source_label - displays the name of the current source 
    self.scrollbar - the scrollbar that allows the user to scroll in the text_box
    self.text_box - creates the text box where the data is printed
    """

    def __init__(self, root, HEIGHT=20, WIDTH=50):
        WIDTH = WIDTH 
        HEIGHT = HEIGHT

        self.timer_callback = lambda: len(([1]))

        self.button_container_frame = Frame(root)
        self.button_container_frame.pack()
        self.button_container_frame.master.title("Strobe Monitor")
        self.text_view_frame = Frame(root)
        self.text_view_frame.pack(side=BOTTOM)

        self.connect_disconnect_button = Button(self.button_container_frame, text="Connect", command=len([1]))
        quit_button = Button(self.button_container_frame, text="Quit", command=quit)
        self.connect_disconnect_button.pack(side=LEFT)
        quit_button.pack(side=LEFT)
        self.text_label = Label(self.button_container_frame, text="Source Number")
        self.text_label.pack(side=LEFT)
        self.entry_box = Entry(self.button_container_frame, width=10)
        self.entry_box.pack(side=LEFT)
        self.enter_button = Button(self.button_container_frame, text="Enter", command=len([1]))
        self.enter_button.pack(side=LEFT)
        self.source_label = Label(self.button_container_frame, text="here")
        self.source_label.pack(side=LEFT)

        self.scrollbar = Scrollbar(self.text_view_frame)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        
        self.text_box = Text(self.text_view_frame, state=DISABLED, yscrollcommand=self.scrollbar.set, width=WIDTH, height=HEIGHT)
        self.text_box.pack(side=BOTTOM)

        self.scrollbar.config(command=self.text_box.yview)

        self.text_box.see(END)
    
    def get_source_number(self):
        """
        Gets the text found in the entry box and returns that text as an integer
        
        Args:
            none
        
        Return:
            text - the text found in the entry box
        """
        text = self.entry_box.get()
        return int(text)

    def insert_text(self, text):
        """
        Inserts text into the text_box.
        
        Arg:
            text - the text to be inserted into the text_box
        
        Returns:
            none
        """
        self.text_box.config(state=NORMAL)
        self.text_box.insert(END, text)
        self.text_box.config(state=DISABLED)
        self.text_box.see(END)

    def change_connect_disconnect_button_text(self, text):
        """
        Changes the text of the connect/disconnect button.
        
        Arg:
            text - the text to change the name of the button to
        
        Returns:
            none
        """
        self.connect_disconnect_button.config(text=text)

    def change_source_label(self, text):
        """
        Changes the source label to the text given.
        
        Arg:
            text - the text to change the source label to 
        
        Returns:
            none
        """
        self.source_label.config(text=text)
    
    def set_connect_disconnect_button_func(self, function):
        """
        Sets the function which will occur whenever the connect/disconnect button is pressed.
        
        Arg:
            function - the function that will occur
        
        Returns:
            none
        """
        self.connect_disconnect_button.config(command=function)
    
    def set_enter_button_func(self, function):
        """
        Sets the function which will occur whenever the enter button is pressed.
        
        Arg:
            function - the function that will occur
        
        Returns:
            none
        """
        self.enter_button.config(command=function)
        self.entry_box.bind(sequence="<Return>", func=function)

    def UITimerFunction(self, TIME=250):
        """
        Calls the timer_callback function every TIME milliseconds.
        
        Arg:
            TIME -  time in milliseconds between each call of the timer_callback function
        
        Return:
            none
        """
        TIME = TIME
        self.timer_callback()
        self.text_box.after(TIME, self.UITimerFunction)