# timestamp_client.py - PyOPXClient example that demonstrates how to get basic and extended information about the
# OmniPlex system using the client API. The client looks for spike or event timestamps and prints out basic
# information on each returned spike or event. It does not read continuous data or spike waveforms.
#
# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from pyopxclient import PyOPXClientAPI, OPX_ERROR_NOERROR

def run():
    # Initialize the API class
    client = PyOPXClientAPI()

    # Connect to OmniPlex Server, check for success
    client.connect()
    if not client.connected:
        print ("Client isn't connected, exiting.\n")
        print ("Error code: {}\n".format(client.last_result))
        exit()

    print ("Connected to OmniPlex Server\n")

    # Get global parameters, print some information
    global_parameters = client.get_global_parameters()
    print ("Number of spike channels: {}".format(global_parameters.num_spike_chans))
    print ("Number of continuous channels: {}".format(global_parameters.num_cont_chans))
    print ("Number of event channels: {}".format(global_parameters.num_event_chans))

    # Figure out which source is for keyboard events, which is used to end the program
    for source_id in global_parameters.source_ids:
        source_name, _, _, _ = client.get_source_info(source_id)
        if source_name == 'KBD':
            keyboard_event_source = source_id
            print ("Keyboard event source is {}".format(keyboard_event_source))

    print ("\nAfter starting, use CTRL-C or any OmniPlex keyboard event to quit...")
    input("\nPress Enter to start reading data...\n")
    
    running = True

    try:
        while(running):
            # Wait up to 1 second for new data to come in
            client.opx_wait(1000)

            # Get a new batch of client data, timestamps only (no waveform or A/D data)
            new_data = client.get_new_data(timestamps_only = True)
            
            if client.last_result != OPX_ERROR_NOERROR:
                print ("Error getting new data")
                print ("Error code {}".format(client.last_result))
                running = False
            
            else:
                # Go through each returned timestamp and print out information
                for i in range(new_data.num_timestamps):
                    # If a keyboard event is in the returned data, stop the client
                    if new_data.source_num_or_type[i] == keyboard_event_source:
                        print ("OmniPlex keyboard event {} detected; stopping aquisition".format(new_data.channel[i]))
                        running = False
                        
                    else:
                        print ("Source: {}\tChannel: {}\tUnit: {}\tTS: {}".format(new_data.source_num_or_type[i], new_data.channel[i], new_data.unit[i], new_data.timestamp[i]))

    except KeyboardInterrupt:
        print ("\nCTRL-C detected; stopping acquisition.")

    client.disconnect()

if __name__ == '__main__':
    run()