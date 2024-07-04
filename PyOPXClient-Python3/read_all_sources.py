# read_all_sources.py - PyOPXClient example that demonstrates how to get basic and extended information about the
# OmniPlex system using the client API. 
#
# This client demonstrates how to get detailed information out of source information functions, and data acquisition
# functions.
#
# (c) 2019 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

from pyopxclient import PyOPXClientAPI, OPX_ERROR_NOERROR, SPIKE_TYPE, CONTINUOUS_TYPE, EVENT_TYPE, OTHER_TYPE
from pyopxclient import OPXSYSTEM_INVALID, OPXSYSTEM_TESTADC, OPXSYSTEM_AD64, OPXSYSTEM_DIGIAMP, OPXSYSTEM_DHSDIGIAMP
from time import sleep

# Handy strings to have associated to their respective types
system_types = { OPXSYSTEM_INVALID: "Invalid System", OPXSYSTEM_TESTADC: "Test ADC", OPXSYSTEM_AD64: "OPX-A", OPXSYSTEM_DIGIAMP: "OPX-D", OPXSYSTEM_DHSDIGIAMP: "OPX-DHP" }
source_types = { SPIKE_TYPE: "Spike", EVENT_TYPE: "Event", CONTINUOUS_TYPE: "Continuous", OTHER_TYPE: "Other" }

# This will be filled in later. Better to store these once rather than have to call the functions
# to get this information on every returned data block
source_numbers_types = {}
source_numbers_names = {}
source_numbers_rates = {}
source_numbers_voltage_scalers = {}

# To avoid overwhelming the console output, set the maximum number of data
# blocks to print information about
max_block_output = 10

# To avoid overwhelming the console output, set the maximum number of continuous
# samples or waveform samples to output
max_samples_output = 5

# Poll time in seconds
poll_time_s = .250

def run():
    # Initialize the API class
    client = PyOPXClientAPI()

    # Connect to OmniPlex Server, check for success
    client.connect()
    if not client.connected:
        print("Client isn't connected, exiting.\n")
        print("Error code: {}\n".format(client.last_result))
        exit()

    print("Connected to OmniPlex Server\n")

    # Get global parameters
    global_parameters = client.get_global_parameters()

    # Figure out which source is for keyboard events, which is used to end the program
    for source_id in global_parameters.source_ids:
        source_name, _, _, _ = client.get_source_info(source_id)
        if source_name == 'KBD':
            keyboard_event_source = source_id

    # Print information on each source
    for index in range(global_parameters.num_sources):
        # Get general information on the source
        source_name, source_type, num_chans, linear_start_chan = client.get_source_info(global_parameters.source_ids[index])

        # Store information about the source types and names for later use.
        source_numbers_types[global_parameters.source_ids[index]] = source_type
        source_numbers_names[global_parameters.source_ids[index]] = source_name
        
        print("----- Source {} -----".format(global_parameters.source_ids[index]))
        print("Name: {}, Type: {}, Channels: {}, Linear Start Channel: {}".format(source_name,
                                                                            source_types[source_type],
                                                                            num_chans,
                                                                            linear_start_chan))
        if source_type == SPIKE_TYPE:
            # Get information specific to a spike source
            _, rate, voltage_scaler, trodality, pts_per_waveform, pre_thresh_pts = client.get_spike_source_info(source_name)
            
            # Store information about the source rate and voltage scaler for later use.
            source_numbers_rates[global_parameters.source_ids[index]] = rate
            source_numbers_voltage_scalers[global_parameters.source_ids[index]] = voltage_scaler

            print("Digitization Rate: {}, Voltage Scaler: {}, Trodality: {}, Points Per Waveform: {}, Threshold Points: {}".format(rate,
                                                                                                                                        voltage_scaler,
                                                                                                                                        trodality,
                                                                                                                                        pts_per_waveform,
                                                                                                                                        pre_thresh_pts))

        if source_type == CONTINUOUS_TYPE:
            # Get information specific to a continuous source
            _, rate, voltage_scaler = client.get_cont_source_info(source_name)
            
            # Store information about the source rate and voltage scaler for later use.
            source_numbers_rates[global_parameters.source_ids[index]] = rate
            source_numbers_voltage_scalers[global_parameters.source_ids[index]] = voltage_scaler

            print("Digitization Rate: {}, Voltage Scaler: {}".format(rate, voltage_scaler))

        print("\n")

    print("After starting, use CTRL-C or any OmniPlex keyboard event to quit...")
    input("\nPress Enter to start reading data...\n")

    running = True

    try:
        while(running):
            # Wait up to 1 second for new data to come in
            client.opx_wait(1000)

            # Get a new batch of client data, timestamps only (no waveform or A/D data)
            new_data = client.get_new_data()

            # Handle the unlikely case that there are fewer blocks returned than we want to output
            if new_data.num_data_blocks < max_block_output:
                num_blocks_to_output = new_data.num_data_blocks
            else:
                num_blocks_to_output = max_block_output

            # If a keyboard event is in the returned data, stop the loop
            for i in range(new_data.num_data_blocks):
                if new_data.source_num_or_type[i] == keyboard_event_source:
                        print("OmniPlex keyboard event {} detected; stopping acquisition".format(new_data.channel[i]))
                        running = False

            print("{}\tblocks read. Displaying info on first {}\tblocks; first {}\tsamples of continuous/spike data.".format(new_data.num_data_blocks, num_blocks_to_output, max_samples_output))

            for i in range(num_blocks_to_output):
                # Output info
                tmp_source_number = new_data.source_num_or_type[i]
                tmp_channel = new_data.channel[i]
                tmp_source_name = source_numbers_names[tmp_source_number]
                tmp_voltage_scaler = source_numbers_voltage_scalers[tmp_source_number]
                tmp_timestamp = new_data.timestamp[i]
                tmp_unit = new_data.unit[i]
                tmp_rate = source_numbers_rates[tmp_source_number]

                # Convert the samples from AD units to voltage using the voltage scaler
                tmp_samples = new_data.waveform[i][:max_samples_output]
                tmp_samples = [s * tmp_voltage_scaler for s in tmp_samples]
                # Construct a string with the samples for convenience
                tmp_samples_str = '{} ' * len(tmp_samples)
                tmp_samples_str = tmp_samples_str.format(*tmp_samples)
                
                if source_numbers_types[new_data.source_num_or_type[i]] == SPIKE_TYPE:
                    print("SRC: {}\t{}\tRATE: {}\tTS: {}\tCH: {}\tUnit:{}\tWF: {}".format(tmp_source_number, tmp_source_name, tmp_rate, tmp_timestamp, tmp_channel, tmp_unit, tmp_samples_str))

                if source_numbers_types[new_data.source_num_or_type[i]] == CONTINUOUS_TYPE:
                    print("SRC: {}\t{}\tRATE: {}\tTS: {}\tCH: {}\tWF: {}".format(tmp_source_number, tmp_source_name, tmp_rate, tmp_timestamp, tmp_channel, tmp_samples_str))

                if source_numbers_types[new_data.source_num_or_type[i]] == EVENT_TYPE:
                    print("SRC: {}\t{}\tTS: {}\tCH: {}".format(tmp_source_number, tmp_source_name, tmp_timestamp, tmp_channel))

            # Pause execution, allowing time for more data to accumulate in OmniPlex Server
            sleep(poll_time_s)
    
    except KeyboardInterrupt:
        print("\nCTRL-C detected; stopping acquisition.")

    # Disconnect from OmniPlex Server
    client.disconnect()

if __name__ == '__main__':
    run()