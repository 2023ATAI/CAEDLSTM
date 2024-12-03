"""
All model structure

<<<<<<<< HEAD
Author: Qingliang Li 
12/23/2022 - V1.0  LSTM, CNN, ConvLSTM edited by Qingliang Li
.........  - V2.0
.........  - V3.0
"""

import torch
import torch.nn as nn
import numpy as np
from convlstm import ConvLSTM
# ------------------------------------------------------------------------------------------------------------------------------
# simple lstm model with fully-connect layer
class LSTMModel(nn.Module):
    """single task model"""

    def __init__(self, cfg,lstmmodel_cfg):
        super(LSTMModel,self).__init__()
        self.lstm = nn.LSTM(lstmmodel_cfg["input_size"], lstmmodel_cfg["hidden_size"],batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(lstmmodel_cfg["hidden_size"],lstmmodel_cfg["out_size"])

    def forward(self, inputs,aux):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        # we only predict the last step
        x = self.dense(x[:,-1,:]) 
        return x
# ------------------------------------------------------------------------------------------------------------------------------
# simple CNN model with fully-connect layer
class CNN(nn.Module):
    """single task model"""

    def __init__(self, cfg):
        super(CNN,self).__init__()
        self.latn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1
        self.lonn = ((2*cfg["spatial_offset"]+1)-cfg["kernel_size"])//cfg["stride_cnn"]+1
        self.cnn = nn.Conv2d(in_channels=cfg["input_size_cnn"],out_channels=cfg["hidden_size"],kernel_size=cfg["kernel_size"],stride=cfg["stride_cnn"])
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(int(cfg["hidden_size"])*int(self.latn)*int(self.lonn),1)

    def forward(self, inputs,aux):
        x = self.cnn(inputs.float())
        x = self.drop(x)
        x = x.reshape(x.shape[0],-1)
        # we only predict the last step
        x = self.dense(x) 
        return x
# ------------------------------------------------------------------------------------------------------------------------------
# simple convlstm model with fully-connect layer
class ConvLSTMModel(nn.Module):
    """single task model"""

    def __init__(self, cfg):
        super(ConvLSTMModel,self).__init__()
        self.ConvLSTM_net = ConvLSTM(input_size=(int(2*cfg["spatial_offset"]+1),int(2*cfg["spatial_offset"]+1)),
                       input_dim=int(cfg["input_size"]),
                       hidden_dim=[int(cfg["hidden_size"]), int(cfg["hidden_size"]/2)],
                       kernel_size=(int(cfg["kernel_size"]), int(cfg["kernel_size"])),
                       num_layers=2,cfg=cfg,batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(int(cfg["hidden_size"]/2)*int(2*cfg["spatial_offset"]+1)*int(2*cfg["spatial_offset"]+1),1)
        #self.batchnorm = nn.BatchNorm1d(int(cfg["hidden_size"]/2))

    def forward(self, inputs,aux,cfg):
        threshold = torch.nn.Threshold(0., 0.0)
        inputs_new = torch.cat([inputs, aux], 2).float()
        #inputs_new = inputs.float()
        hidden =  self.ConvLSTM_net.get_init_states(inputs_new.shape[0])
        last_state, encoder_state =  self.ConvLSTM_net(inputs_new.clone(), hidden)
        last_state = self.drop(last_state)
        #Convout = last_state[:,-1,:,cfg["spatial_offset"],cfg["spatial_offset"]]
        Convout = last_state[:,-1,:,:,:]
        #Convout = self.batchnorm(Convout)
        shape=Convout.shape[0]
        #print('Convout shape is',Convout.shape)
        Convout=Convout.reshape(shape,-1)
        Convout = torch.flatten(Convout,1)
        Convout = threshold(Convout)
        predictions=self.dense(Convout)

        return predictions

# simple ED-LSTM model with fully-connect layer
class LSTMEncoder(nn.Module):
    def __init__(self, cfg):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=cfg["input_size"], hidden_size=cfg["hidden_size"], num_layers= cfg['num_layers'], batch_first=True)
        self.dense = nn.Linear(cfg["hidden_size"], cfg["out_size"])
    def forward(self, x):
        # Initialize hidden state with zeros
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward pass through LSTM
        out, _ = self.lstm(x)
        out = self.dense(out)
        return out

class LSTMDecoder(nn.Module):
    def __init__(self, cfg):
        super(LSTMDecoder, self).__init__()
        # input_size is out_size
        self.lstm = nn.LSTM(input_size=cfg['out_size'] , hidden_size = cfg["hidden_size"], num_layers = cfg['num_layers'], batch_first=True)
        self.fc = nn.Linear(cfg["hidden_size"], cfg['out_size'])
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
    def forward(self, x):
        # Forward pass through LSTM
        out, _= self.lstm(x)
        # Pass through fully connected layer
        out = self.drop(out)
        out = self.fc(out[:,-1,:])
        return out

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder = LSTMEncoder(cfg)
        self.decoder = LSTMDecoder(cfg)
    def forward(self, source):
        encoder_output = self.encoder(source)
        decoder_output = self.decoder(encoder_output)
        return decoder_output
# simple AttLSTM model with fully-connect layer
class AttentionLSTM(nn.Module):
    def __init__(self, cfg):
        super(AttentionLSTM, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=cfg['input_size'], hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'], batch_first=True)

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=cfg['hidden_size'], num_heads=cfg['num_heads'])
        self.dense = nn.Linear(in_features = cfg['hidden_size'],out_features = cfg['out_size'])
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
    def forward(self, x):
        # Forward pass through LSTM
        out, _ = self.lstm(x)
        # Permute the output for attention layer
        out = out.permute(1, 0, 2)

        # Apply multi-head attention
        out, _ = self.attention(out, out, out)

        # Permute the output back to the original shape
        out = out.permute(1, 0, 2)
        out = self.drop(out)
        out = self.dense(out[:,-1,:])
        return out
# AEDLSTM
class ADLSTMAttentionEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super(LSTMAttentionEncoderDecoder, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=cfg['input_size'], hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'],
                            batch_first=True)
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=cfg['hidden_size'], num_heads=cfg['num_heads'])

        self.dense1 = nn.Linear(cfg["hidden_size"], cfg["out_size"])

        # self.encoder = LSTMEncoder(cfg)
        self.decoder = LSTMDecoder(cfg)

    def forward(self, x):
        # Forward pass through LSTM
        out, _ = self.lstm(x)
        # Permute the output for attention layer
        out = out.permute(1, 0, 2)
        # Apply multi-head attention
        out, _ = self.attention(out, out, out)
        out = self.dense1(out)
        out = out.permute(1, 0, 2)
        # Forward pass through encoder
        # encoder_output = self.encoder(out)
        # Forward pass through decoder
        out = self.decoder(out)

        return out


# CAEDLSTM
class LSTMAttentionEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super(LSTMAttentionEncoderDecoder, self).__init__()

        # CNN layers
        input_channels = 16
        self.conv1 = nn.Conv1d(in_channels=input_channels, 
                               out_channels=64, 
                               kernel_size=3, 
                               padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=64, hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'],
                            batch_first=True)

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=cfg['hidden_size'], num_heads=cfg['num_heads'])

        self.dense1 = nn.Linear(cfg["hidden_size"], cfg["out_size"])

        # LSTM Decoder
        self.decoder = LSTMDecoder(cfg)

    def forward(self, x):
        # CNN expects (batch_size, channels, seq_length)
        x = x.permute(0, 2, 1)  # Reshape input for CNN
        x = self.conv1(x.float())
        x = torch.relu(x)
        x = self.pool(x)

        # Reshape for LSTM layer (batch_size, seq_length, channels)
        x = x.permute(0, 2, 1)

        # Forward pass through LSTM
        out, _ = self.lstm(x)

        # Permute the output for attention layer
        out = out.permute(1, 0, 2)

        # Apply multi-head attention
        out, _ = self.attention(out, out, out)

        # Linear layer
        out = self.dense1(out)

        # Permute back to (batch_size, seq_length, out_size)
        out = out.permute(1, 0, 2)

        # Forward pass through decoder
        out = self.decoder(out)

        return out


# simple AttConvLSTM model with fully-connect layer
class ConvLSTMWithAttention(nn.Module):
    def __init__(self, cfg):
        super(ConvLSTMWithAttention, self).__init__()

        self.conv_lstm = ConvLSTM(input_size=(int(2 * cfg["spatial_offset"] + 1), int(2 * cfg["spatial_offset"] + 1)),
                                        input_dim=int(cfg["input_size"]),
                                        hidden_dim=cfg["hidden_size"],
                                        kernel_size=(int(cfg["kernel_size"]), int(cfg["kernel_size"])),
                                        num_layers=cfg['num_layers'], cfg=cfg, batch_first=True)

        # self.multihead_attention = nn.MultiheadAttention(
        #     embed_dim=int(cfg["hidden_size"])*int(2 * cfg["spatial_offset"] + 1)*int(2 * cfg["spatial_offset"] + 1),
        #     num_heads=cfg['num_heads']
        # )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=cfg["hidden_size"],
            num_heads=cfg['num_heads']
        )
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(in_features=int(cfg["hidden_size"])*int(2 * cfg["spatial_offset"] + 1)*int(2 * cfg["spatial_offset"] + 1), out_features=cfg['out_size'])

    def forward(self, x):
        hidden = self.conv_lstm.get_init_states(x.shape[0])
        # ConvLSTM forward pass
        convlstm_out, _ = self.conv_lstm(x, hidden)
        # Rearrange the output for attention layer
        # lstm_out = lstm_out.permute(0, 2, 1, 3, 4).contiguous()
        convlstm_out = convlstm_out.view(convlstm_out.size(0), -1, convlstm_out.size(2))
        # Apply multi-head self-attention
        attn_output, _ = self.multihead_attention(convlstm_out, convlstm_out, convlstm_out)
        # Reshape attention output back to the original size
        # attn_output = attn_output.view(attn_output.size(0), lstm_out.size(1), x.size(2), x.size(3), x.size(4))
        # attn_output = attn_output.permute(0, 2, 1, 3, 4).contiguous()
        attn_output = self.drop(attn_output)
        attn_output = attn_output.view(attn_output.size(0), x.size(1), -1)
        attn_output = self.dense(attn_output[:, -1, :])
        return attn_output

# simple EDConvLSTM model with fully-connect layer
class ConvLSTMEncoder(nn.Module):
    def __init__(self, cfg):
        super(ConvLSTMEncoder, self).__init__()
        self.conv_lstm = ConvLSTM(input_size=(int(2 * cfg["spatial_offset"] + 1), int(2 * cfg["spatial_offset"] + 1)),
                                  input_dim=int(cfg["input_size"]),
                                  hidden_dim=cfg["hidden_size"],
                                  kernel_size=(int(cfg["kernel_size"]), int(cfg["kernel_size"])),
                                  num_layers=cfg['num_layers'], cfg=cfg, batch_first=True)
        # self.conv_lstm = nn.ConvLSTM2d(input_size, hidden_size, kernel_size, num_layers=num_layers, batch_first=True)
        # self.dense = nn.Linear(in_features=cfg['hidden_size'], out_features=cfg['hidden_size'])
        self.dense = nn.Linear(in_features=cfg['hidden_size'], out_features=cfg['input_size'])
    def forward(self, x):
        hidden = self.conv_lstm.get_init_states(x.shape[0])
        # ConvLSTM forward pass
        encoder_output, _ = self.conv_lstm(x, hidden)
        encoder_output = encoder_output.permute(0, 1, 4, 3, 2)
        encoder_output = self.dense(encoder_output)
        encoder_output = encoder_output.permute(0, 1, 4, 3, 2)

        # Forward pass through ConvLSTM
        # encoder_output, _ = self.conv_lstm(x)
        return encoder_output

class ConvLSTMDecoder(nn.Module):
    def __init__(self, cfg):
        super(ConvLSTMDecoder, self).__init__()
        self.conv_lstm = ConvLSTM(input_size=(int(2 * cfg["spatial_offset"] + 1), int(2 * cfg["spatial_offset"] + 1)),
                                  input_dim=int(cfg['input_size']),
                                  hidden_dim=cfg["hidden_size"],
                                  kernel_size=(int(cfg["kernel_size"]), int(cfg["kernel_size"])),
                                  num_layers=cfg['num_layers'], cfg=cfg, batch_first=True)
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.dense = nn.Linear(int(cfg["hidden_size"])*int(2*cfg["spatial_offset"]+1)*int(2*cfg["spatial_offset"]+1),cfg['out_size'])
        # self.conv_lstm = nn.ConvLSTM2d(input_size, hidden_size, kernel_size, num_layers=num_layers, batch_first=True)
        # self.output_layer = nn.Conv2d(hidden_size, 1, kernel_size=1)  # Adjust output channels and kernel size as needed

    def forward(self, encoder_output):
        # Initialize hidden states and cell states for decoder
        hidden = self.conv_lstm.get_init_states(encoder_output.shape[0])
        # ConvLSTM forward pass
        # batch,sql,hidden,w,h
        convlstm_out, _ = self.conv_lstm(encoder_output, hidden)
        # Rearrange the output for attention layer
        # lstm_out = lstm_out.permute(0, 2, 1, 3, 4).contiguous()
        convlstm_out = convlstm_out.view(convlstm_out.size(0), convlstm_out.size(1), -1)
        convlstm_out = self.drop(convlstm_out)

        convlstm_out = self.dense(convlstm_out[:, -1, :])

        return convlstm_out

class ConvLSTMAttentionEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super(ConvLSTMAttentionEncoderDecoder, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=int(cfg["input_size"]) * int(2 * cfg["spatial_offset"] + 1) * int(2 * cfg["spatial_offset"] + 1),
            num_heads=cfg["input_size"]
            # num_heads=cfg['num_heads']
        )
        self.encoder = ConvLSTMEncoder(cfg)
        self.decoder = ConvLSTMDecoder(cfg)

    def forward(self, x):
        intput = x.view(x.size(0), x.size(1), -1)
        intput, _ = self.multihead_attention(intput,intput,intput)
        intput = intput.reshape(x.shape)
        # Forward pass through encoder
        encoder_output = self.encoder(intput)
        # Forward pass through decoder
        output = self.decoder(encoder_output)

        return output
class ConvLSTMEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super(ConvLSTMEncoderDecoder, self).__init__()

        self.encoder = ConvLSTMEncoder(cfg)
        self.decoder = ConvLSTMDecoder(cfg)

    def forward(self, x):
        # Forward pass through encoder
        encoder_output = self.encoder(x)
        # Forward pass through decoder
        output = self.decoder(encoder_output)

        return output








