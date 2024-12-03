import argparse
import pickle
from pathlib import PosixPath, Path
    # Original author : Qingliang Li, Cheng Zhang, 12/23/2022


def get_args() -> dict:
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:1')#'cuda:0'
    # path
    parser.add_argument('--inputs_path', type=str, default='../../../data/hongjian/cnnAEDlstm/')
    parser.add_argument('--nc_data_path', type=str, default='../../../data/')
    parser.add_argument('--product', type=str, default='LandBench')
    parser.add_argument('--workname', type=str, default='LandBench')
    parser.add_argument('--modelname', type=str, default='LSTMAttentionEncoderDecoder')# LSTM;ConvLSTM;CNN'LSTMEncoderDecoder','AttentionLSTM'LSTMAttentionEncoderDecoder
    parser.add_argument('--label',nargs='+', type=str, default=["volumetric_soil_water_layer_1"])#volumetric_soil_water_layer_1;surface_sensible_heat_flux;volumetric_soil_water_layer_20
    parser.add_argument('--stride', type=float, default=20) 
    parser.add_argument('--data_type', type=str, default='float32') 
    # data
    parser.add_argument('--selected_year', nargs='+', type=int, default=[2000,2020])
          # forcing SM:["2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind","precipitation","surface_pressure","specific_humidity"]
          # forcing SSHF:["2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind","precipitation","surface_pressure","specific_humidity","10m_u_component_of_wind",]
    parser.add_argument('--forcing_list', nargs='+', type=str, default=["2m_temperature", "precipitation","10m_u_component_of_wind", 
                                 "10m_v_component_of_wind", "surface_pressure", "specific_humidity"])
          # land surface SM:["surface_solar_radiation_downwards_w_m2","surface_thermal_radiation_downwards_w_m2","soil_temperature_level_2"]
          # land surface SSHF:["surface_solar_radiation_downwards_w_m2","surface_thermal_radiation_downwards_w_m2"]
    parser.add_argument('--land_surface_list', nargs='+', type=str, default=["surface_solar_radiation_downwards_w_m2", "surface_thermal_radiation_downwards_w_m2",
                                 "soil_temperature_level_2", "total_evaporation"])
          # static SM: ["soil_water_capacity"]
          # static SSHF: ["soil_water_capacity"]
    parser.add_argument('--static_list', nargs='+', type=str, default=["soil_water_capacity", "landtype","clay_0-5cm_mean","sand_0-5cm_mean","silt_0-5cm_mean","dem"])#,"dem"

    parser.add_argument('--memmap', type=bool, default=True)
    parser.add_argument('--test_year', nargs='+', type=int, default=[2020])
    parser.add_argument('--input_size', type=float, default=16)
    parser.add_argument('--num_layers', type=float, default=2)
    parser.add_argument('--num_heads', type=float, default=2)
    parser.add_argument('--out_size', type=float, default=1)
    parser.add_argument('--spatial_resolution', type=float, default=1)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--spatial_offset', type=float, default=3) #CNN
    parser.add_argument('--valid_split', type=bool, default=True)

    # physical model
    parser.add_argument('--physical_model', type=bool, default=False)  # Whether it's a physical model
    parser.add_argument('--hbv_parameter', type=str, default='00')  # HBV model parameter selection
    parser.add_argument('--parameters_path', type=str, default='../parameter1du/')
    parser.add_argument('--parameters_names', nargs='+', type=str,
                        default=['BETA.nc', 'FC.nc', 'K0.nc', 'K1.nc', 'K2.nc', 'LP.nc', 'PERC.nc', 'UZL.nc', 'TT.nc',
                                 'CFMAX.nc', 'CFR.nc', 'CWH.nc', 'PCORR.nc', 'SFCF.nc'])
    parser.add_argument('--mid_forcing_list', nargs='+', type=str, default=[
    "soil_wetness", "SLZ"
    ])  # "PRECIP", "RAIN", "SNOW", "SNOWPACK", "melt", "MELTWATER", "refreezing","tosoil", "soil_wetness", "recharge", "SM", "excess", "ETact", "SUZ", "PERC", "Q0", "Q1", "SLZ", "Q2", "Qall"

    # model
    parser.add_argument('--normalize_type', type=str, default='region')#global, #region
    parser.add_argument('--forcast_time', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=float, default=256)
    parser.add_argument('--batch_size', type=float, default=64)
    parser.add_argument('--patience', type=int, default=10) 
    parser.add_argument('--seq_len', type=float, default=365) #365 or 7;
    parser.add_argument('--epochs', type=float, default=1000)#500
    parser.add_argument('--niter', type=float, default=400) #200
    parser.add_argument('--num_repeat', type=float, default=1)#default :1
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--input_size_cnn', type=float, default=64) #CNN (seq_len)*(num of forcing_list+num of land_surface_list)+1
    parser.add_argument('--kernel_size', type=float, default=3) #CNN
    parser.add_argument('--stride_cnn', type=float, default=2) #CNN
    cfg = vars(parser.parse_args())

    return cfg
