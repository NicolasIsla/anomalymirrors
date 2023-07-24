import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
##############################################################################################################
###################################################UTILS######################################################
def filter_and_save_data(data_path_psd, data_path_state, output_directory, end_loop=False):
    """
    This function filters the data on VLTI mode and saves it in the output directory.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        os.makedirs(output_directory+"/UT1")
        os.makedirs(output_directory+"/UT2")
        os.makedirs(output_directory+"/UT3")
        os.makedirs(output_directory+"/UT4")

    # Loop over the telescopes
    for telescope in tqdm(os.listdir(data_path_psd)):
        # Loop over the psd files
        for file in os.listdir(data_path_psd+"/"+telescope):
            if "csv" in file:
                # Information about the file
                sensor, year, month = file.split("_psds")[0], file[-11:-7], file[-6:-4]
                # Load the state file and the psd file
                state = pd.read_csv(data_path_state+"/"+year+"/"+month+"/"+telescope+"/"+"states_"+telescope+"_"+year+"-"+month+".csv", index_col=0)
                df = pd.read_csv(data_path_psd+"/"+telescope+"/"+file, index_col=0)
                # Join the two dataframes
                joined_df = df.join(state)
                # Filter the data on vlti mode
                joined_df = joined_df[(
                                        (joined_df["guiding"] == 1) & 
                                        (joined_df["presetting"] == 0) & 
                                        (joined_df["enc_open"] == 1) &
                                        (joined_df["coude_focus"] == 1)
                                      )]
                # Save the file
                os.makedirs(f"{output_directory}/{telescope}", exist_ok=True)
                joined_df.to_csv(f"{output_directory}/{telescope}/{sensor}_{year}-{month}.csv")
        if end_loop:
            break

def datasensor(sensor: str, path: str = "./filtered_data/UT1/"):
    """
    This function reads the data from the filtered files and returns a dataframe with the data
    of the sensor passed as argument. The data is concatenated.
    Also the columns that are not useful are dropped, like guiding, presetting, etc and the columns
    that are more than 10 hz are filtered.
    """
    # files with the extension .csv
    csv_files = [file for file in os.listdir(path) if file.endswith(".csv")]

    # filter the files with the sensor name
    filtered_files = [file for file in csv_files if sensor in file]

    # read and concatenate the files
    dataframes = [pd.read_csv(os.path.join(path, file), index_col=0) for file in filtered_files]
    concatenated_data = pd.concat(dataframes)

    # drop the columns that are not useful
    columnas_excluidas = ['guiding', 'presetting', 'enc_open', 'coude_focus', 'nasA_focus',
                          'nasB_focus', 'cas_focus', 'bad_guiding', 'AS_update', 'PS_update']

    concatenated_data = concatenated_data.drop(columnas_excluidas, axis=1)

    columnas = concatenated_data.columns.values.astype(float)

    # filter the columns that are more than 10 hz
    filter10hz = columnas > 10

    return concatenated_data.loc[:, filter10hz]

##############################################################################################################
############ ######################################MODEL######################################################
class Encoder(nn.Module):
    def __init__(self, n_features, n_embedding, drop_out=0.1):
        super(Encoder, self).__init__()

        self.linear_encoder_1 = nn.Linear(n_features, 512)
        self.linear_encoder_2 = nn.Linear(512, 256)
        self.linear_encoder_3 = nn.Linear(256, 128)
        self.linear_encoder_4 = nn.Linear(128, n_embedding)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear_encoder_1(x)))
        x = self.dropout(torch.relu(self.linear_encoder_2(x)))
        x = self.dropout(torch.relu(self.linear_encoder_3(x)))
        x = self.linear_encoder_4(x)
        return x



class Decoder(nn.Module):
    def __init__(self, n_embedding, n_features, dropout=0.1):
        super(Decoder, self).__init__()

        self.linear_decoder_1 = nn.Linear(n_embedding, 128)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_decoder_2 = nn.Linear(128, 256)
        self.dropout_2 = nn.Dropout(dropout)
        self.linear_decoder_3 = nn.Linear(256, 512)
        self.dropout_3 = nn.Dropout(dropout)
        self.linear_decoder_4 = nn.Linear(512, n_features)

    def forward(self, x):
        x = nn.ReLU()(self.linear_decoder_1(x))
        x = self.dropout_1(x)
        x = nn.ReLU()(self.linear_decoder_2(x))
        x = self.dropout_2(x)
        x = nn.ReLU()(self.linear_decoder_3(x))
        x = self.dropout_3(x)
        x = self.linear_decoder_4(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, n_features, n_embedding, dropout=0.1):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(n_features, n_embedding, dropout)
        self.decoder = Decoder(n_embedding, n_features, dropout)

    def forward(self, x, latten = False):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if latten:
            return x, encoded
        else:
            return decoded


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.val_loss_min:
            self.counter = 0
            self.val_loss_min = val_loss
            self.best_model = model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_model(
    model,
    train_dataset,
    val_dataset,
    max_epochs,
    batch_size,
    lr,
    optimizer,
    perdida,
    early_stopping,
    use_gpu=False
    
    ):
    """
    this function trains the model and returns the loss curves.
    """
    #loss curve
    curves = {
        "train_loss": [],
        "val_loss": [],
    }
    
    # Dataloader for the train and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,pin_memory=use_gpu)

    for epoch in range(max_epochs):
        cumulative_train_loss = 0

        # training loop
        model.train()
        for i, x_batch in enumerate(train_loader):
            print(f"\rEpoch {epoch + 1}/{max_epochs} - Batch {i+1}/{len(train_loader)}", end="")
            
            # prediccition
            x_reconstructed = model(x_batch)
            loss = perdida(x_batch,x_reconstructed)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulative_train_loss += loss.item()

        train_loss = cumulative_train_loss / len(train_loader)

        # evaluation loop
        model.eval()
        with torch.no_grad():
            for i, x_val in enumerate(val_loader):
                x_reconstructed_val = model(x_val)
                val_loss = perdida(x_val,x_reconstructed_val).item()
                
        print("")
        print(f" - Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}")
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

        if early_stopping(val_loss, model):
            print("Early stopping triggered!")
            break

    print()
    model.cpu()
    return curves


def show_curves(curves):
    """
    this function plots the loss curves from the trainfunction above.
    """
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    fig.set_facecolor('white')

    epochs = np.arange(len(curves["val_loss"])) + 1
    ax.plot(epochs, curves['val_loss'], label='validation')
    ax.plot(epochs, curves['train_loss'], label='training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss evolution during training')
    ax.legend()
    plt.show()