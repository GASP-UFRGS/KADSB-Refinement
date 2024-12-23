import argparse
import time
import torch
from pathlib import Path
from functions import train_conv_network


parser = argparse.ArgumentParser()

parser.add_argument('--debug', type=bool, required=True, default=False)

args = parser.parse_args()

debug = args.debug


# "Fast_16_8_4":{"modelConv_type":"Fast","en_elayers_dim":16,"temb_dim":8,"conv_dof":4},
# "Chebyshev_16_8_8":{"modelConv_type":"Chebyshev","en_elayers_dim":16,"temb_dim":8,"conv_dof":8},
# "ReLU_16_8_4":{"modelConv_type":"ReLU","en_elayers_dim":16,"temb_dim":8,"conv_dof":4},
# "Bottleneck_16_8_2":{"modelConv_type":"Bottleneck","en_elayers_dim":16,"temb_dim":8,"conv_dof":2},
# "Bernstein_16_8_1":{"modelConv_type":"Bernstein","en_elayers_dim":16,"temb_dim":8,"conv_dof":1},
# "Jacobi_16_8_1":{"modelConv_type":"Jacobi","en_elayers_dim":16,"temb_dim":8,"conv_dof":1},
# "Wav_16_8_4":{"modelConv_type":"Wav","en_elayers_dim":16,"temb_dim":8,"conv_dof":4}
                                    


conv_models_dict = {"Bottleneck_16_8_2":{"modelConv_type":"Bottleneck","en_elayers_dim":16,"temb_dim":8,"conv_dof":2},
                    "BottleneckAttention_16_8_1":{"modelConv_type":"BottleneckAttention","en_elayers_dim":16,"temb_dim":8,"conv_dof":1}}

sel_conv_models = conv_models_dict.keys()

if __name__ == "__main__":

    for conv_model in sel_conv_models:
        torch.cuda.empty_cache()
        time.sleep(10)
        if debug:
            status = train_conv_network(conv_models_dict[conv_model]['modelConv_type'], conv_models_dict[conv_model]['en_elayers_dim'], conv_models_dict[conv_model]['temb_dim'], conv_models_dict[conv_model]['conv_dof'], n_iter=10)
        else:            
            try:
                status = status = train_conv_network(conv_models_dict[conv_model]['modelConv_type'], conv_models_dict[conv_model]['en_elayers_dim'], conv_models_dict[conv_model]['temb_dim'], conv_models_dict[conv_model]['conv_dof'], n_iter=10)
                print(f"{status} - Energy: {conv_models_dict[conv_model]['modelConv_type']}_{conv_models_dict[conv_model]['en_elayers_dim']}_{conv_models_dict[conv_model]['temb_dim']}_{conv_models_dict[conv_model]['conv_dof']}, iter{conv_models_dict[conv_model]['conv_model_iter']}")
            except:
                print(f"#----- Falha: Energy: {conv_models_dict[conv_model]['modelConv_type']}_{conv_models_dict[conv_model]['en_elayers_dim']}_{conv_models_dict[conv_model]['temb_dim']}_{conv_models_dict[conv_model]['conv_dof']}, iter{conv_models_dict[conv_model]['conv_model_iter']} -----#")

    print("Done!")