# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from .model import Model
from .view import View
from tkinter import Tk


class Controller():
    """
    The Controller class contains functions that communicate between the View class
    and the Model class.
    
    Args:
        none
    
    Returns:
        None

    root - creates the window (GUI)
    self.view - instantiates the Model class
    self.model - instantiates the View class
    """

    def __init__(self):
        # Create Tk root window
        self.root = Tk()
        # Create instance of view
        self.view = View(self.root)
        # Create instance of model
        self.client = Model()

        # Sets the connect_disconnect_button to call the button_clicked function
        self.view.set_connect_disconnect_button_func(self.connect_disconnect_button_clicked)
        # Sets the timer_callback function to controller_update
        self.view.timer_callback = self.controller_update
        # Sets the enter_button to call the enter_button_clicked function
        self.view.set_enter_button_func(self.enter_button_clicked)
        # Sets the start source number according to what the source number is
        start_source_number = self.client.get_source_number()
        # Causes the entry_box to contain the starting source number
        self.view.entry_box.insert(0, start_source_number)
        # Causes the source_label to display the name of the source number that is displayed in the entry box
        self.change_source_name(start_source_number, first=True)

        # Start the main UI loop
        self.root.mainloop()

    def change_source_name(self, source_number, first=False):
        """
        Changes the source name displayed.
        
        Arg:
            source_number - the number the source
        
        Returns:
            none
        """
        if first:
            self.source_label = ""
        else:
            if source_number not in self.client.source_ids:  
                self.source_label = "Invalid Source"
            else:
                self.not_running = True

                # connects the client if it is not initially connected
                if self.client.connected:
                    self.not_running = False
                else:
                    self.client.connect()

                self.client.change_source_number(source_number)
                source_name = self.client.get_source_name(source_number)
                if source_name == "":
                    self.source_label = "No Name"
                else:
                    self.source_label = source_name
                # disconnects the client if it is initally disconnected
                if self.not_running:
                    self.client.disconnect()
        self.view.change_source_label(self.source_label)    

    def connect_disconnect_button_clicked(self):
        """
        Is called when the connect/disconnect button is clicked. If the connect button is clicked, it changes the 
        title of the button to disconnect and calls the connect function. If the disconnect button is clicked, 
        it changes the title of the button to calls and initiates the disconnect function.
        
        Args: 
            none
        
        Returns:
            none
        """
        if not self.client.connected:
            self.view.change_connect_disconnect_button_text(text='Disconnect')
            self.connect()
        else:
            self.view.change_connect_disconnect_button_text(text='Connect')
            self.disconnect()     

    def enter_button_clicked(self, useless=0):
        """
        Is called when the enter button is clicked. The source number is changed to the number in the entry box.
        Prints the source number it was changed to. 
        
        Arg:
            useless - the enter/return button is binded to this function and tkinters bind function returns a parameter 
            so useless is the variable that stores that parameter. It is not useful in this function hence the name uselesss. 
        
        Returns:
            none
        
        """
        source_number = self.view.get_source_number()     
        self.change_source_name(source_number)
       
    def connect(self):
        """
        Allows the Strobe Monitor to receive strobe data and display it.
        
        Args:
            none
        
        Returns:
            none
        """
        self.client.connect()
        self.enter_button_clicked()
        self.view.insert_text("Connected\n")
        self.view.UITimerFunction()

    def disconnect(self):
        """
        Stops the Strobe Monitor from receiving strobe data and displaying it.
        
        Args:
            none
        
        Returns:
            none
        """
        self.client.disconnect()
        text = "------------------------------------\n"
        self.view.insert_text(text)

    def controller_update(self):
        """
        Updates the controller.
        
        Args:
            none
        
        Returns:
            none
        """
        data = self.client.model_update()
        for i in range(len(data)):
            text = f"TS: {data[i][0]}\t\t\tValue: {data[i][1]}\n"
            self.view.insert_text(text)



