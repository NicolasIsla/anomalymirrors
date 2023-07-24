from tqdm import tqdm
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":

    data_routh_psd = os.getenv("DATA_ROUTH_PSD")
    data_routh_state = os.getenv("DATA_ROUTH_STATE")
    
    output_directory = os.getenv("FILTERED_DATA")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

        os.makedirs(output_directory+"/UT1")
        os.makedirs(output_directory+"/UT2")
        os.makedirs(output_directory+"/UT3")
        os.makedirs(output_directory+"/UT4")

    # loop over the telescopes
    for telescope in tqdm(os.listdir(data_routh_psd)):
        # loop over tht psd files
        for file in os.listdir(data_routh_psd+"/"+telescope):
            if "csv" in file:
                # information about the file
                sensor, year, month = file.split("_psds")[0], file[-11:-7], file[-6:-4]
                # load the state file and the psd file
                state = pd.read_csv(data_routh_state+"/"+year+"/"+month+"/"+telescope+"/"+"states_"+telescope+"_"+year+"-"+month+".csv", index_col=0)
                df = pd.read_csv(data_routh_psd+"/"+telescope+"/"+file, index_col=0)
                # join the two dataframes
                joined_df = df.join(state)
                # filter the data on vlti mode
                joined_df = joined_df[(
                                        (joined_df["guiding"] == 1) & 
                                        (joined_df["presetting"] == 0) & 
                                        (joined_df["enc_open"] == 1) &
                                        (joined_df["coude_focus"] == 1)
                                        )]
                                        
                                        
                # save the file
                joined_df.to_csv(f"{output_directory}/{telescope}/{sensor}_{year}-{month}.csv")

