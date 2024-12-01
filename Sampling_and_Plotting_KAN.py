import argparse
import torch
from plotting_functions import sampling_and_plotting


parser = argparse.ArgumentParser()

parser.add_argument('--debug', type=bool, required=True, default=False)

args = parser.parse_args()

debug = args.debug

# # !!!!!!!!!! "Lagrange_128_8_128":{"modelEnergy_type":"Lagrange","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},
#                   "Bernstein_128_8_128":{"modelEnergy_type":"Bernstein","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},
#                   "Bernstein_128_64_128":{"modelEnergy_type":"Bernstein","en_elayers_dim":128,"pos_dim":64,"en_model_iter":-1},
#                   "Bottleneck_128_64_128":{"modelEnergy_type":"Bottleneck","en_elayers_dim":128,"pos_dim":64,"en_model_iter":-1},
#                   "Chebyshev_128_64_128":{"modelEnergy_type":"Chebyshev","en_elayers_dim":128,"pos_dim":64,"en_model_iter":-1},
#                   "Fast_128_64_128":{"modelEnergy_type":"Fast","en_elayers_dim":128,"pos_dim":64,"en_model_iter":-1},
#                   "Gram_128_64_128":{"modelEnergy_type":"Gram","en_elayers_dim":128,"pos_dim":64,"en_model_iter":-1},
#                   "Jacobi_128_8_128":{"modelEnergy_type":"Jacobi","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},
#                   "Jacobi_128_64_128":{"modelEnergy_type":"Jacobi","en_elayers_dim":128,"pos_dim":64,"en_model_iter":-1},
#                   "Jacobi_256_128_256":{"modelEnergy_type":"Jacobi","en_elayers_dim":256,"pos_dim":128,"en_model_iter":-1},
#                   "Lagrange_128_8_128":{"modelEnergy_type":"Lagrange","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},
#                   "Lagrange_128_64_128":{"modelEnergy_type":"Lagrange","en_elayers_dim":128,"pos_dim":64,"en_model_iter":-1},
#                   "ReLU_128_8_128":{"modelEnergy_type":"ReLU","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},
#                   "ReLU_128_64_128":{"modelEnergy_type":"ReLU","en_elayers_dim":128,"pos_dim":64,"en_model_iter":-1},
                  


en_models_dict = {"Bernstein_16_8_16":{"modelEnergy_type":"Bernstein","en_elayers_dim":16,"pos_dim":8,"en_model_iter":-1},
                  "Chebyshev_16_8_16":{"modelEnergy_type":"Chebyshev","en_elayers_dim":16,"pos_dim":8,"en_model_iter":-1},
                  "Gram_16_8_16":{"modelEnergy_type":"Gram","en_elayers_dim":16,"pos_dim":8,"en_model_iter":-1},
                  "Jacobi_16_8_16":{"modelEnergy_type":"Jacobi","en_elayers_dim":16,"pos_dim":8,"en_model_iter":-1},
                  "Lagrange_16_8_16":{"modelEnergy_type":"Lagrange","en_elayers_dim":16,"pos_dim":8,"en_model_iter":-1}}




conv_models_dict = {"Bottleneck_256_128_2":{"modelConv_type":"Bottleneck", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":2, "conv_model_iter":-1},
                    "Bottleneck_128_8_2":{"modelConv_type":"Bottleneck", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":2, "conv_model_iter":-1},
                    "BottleneckKAGNAttentionConv_16_8_1":{"modelConv_type":"BottleneckKAGNAttentionConv", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":1, "conv_model_iter":-1},
                    "BottleneckKAGNLinear_256_128_32":{"modelConv_type":"BottleneckKAGNLinear", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":32, "conv_model_iter":-1},
                    "BottleneckKAGNLinear_128_8_4":{"modelConv_type":"BottleneckKAGNLinear", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    "Chebyshev_256_128_8":{"modelConv_type":"Chebyshev", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":8, "conv_model_iter":-1},
                    "Chebyshev_128_8_4":{"modelConv_type":"Chebyshev", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    "Fast_256_128_4":{"modelConv_type":"Fast", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":4, "conv_model_iter":-1},
                    "Fast_128_8_4":{"modelConv_type":"Fast", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    "FastLinear_256_128_32":{"modelConv_type":"FastLinear", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":32, "conv_model_iter":-1},
                    "FastLinear_128_8_4":{"modelConv_type":"FastLinear", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    "FastWide_128_8_4":{"modelConv_type":"FastWide", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    "Gram_16_8_1":{"modelConv_type":"Gram", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":1, "conv_model_iter":-1},
                    "Lagrange_16_8_1":{"modelConv_type":"Lagrange", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":1, "conv_model_iter":-1},
                    "ReLU_256_128_4":{"modelConv_type":"ReLU", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":4, "conv_model_iter":-1},
                    "ReLU_128_8_4":{"modelConv_type":"ReLU", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    "ReLULinear_128_8_4":{"modelConv_type":"ReLULinear", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    "SQuIRELS_256_128_32":{"modelConv_type":"SQuIRELS", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":32, "conv_model_iter":-1},
                    "SQuIRELS_16_8_2":{"modelConv_type":"SQuIRELS", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":2, "conv_model_iter":-1},
                    "SQuIRELSLinear_256_128_32":{"modelConv_type":"SQuIRELSLinear", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":32, "conv_model_iter":-1},
                    "SQuIRELSLinear_128_8_4":{"modelConv_type":"SQuIRELSLinear", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    "Wav":{"modelConv_type":"Wav", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":2, "conv_model_iter":-1}}

sel_en_models = en_models_dict.keys()#["SQuIRELS_256_128_256","Fast_16_8_16"]
sel_conv_models = ["SQuIRELS_256_128_32"]#conv_models_dict.keys()#["SQuIRELS_256_128_32"]

record_metrics = True
generate_plots = True
full_model_metrics = False
energy_intervals = True
cuda = True

if __name__ == "__main__":

    for en_model in sel_en_models:
        for conv_model in sel_conv_models:
            torch.cuda.empty_cache()
            if debug:
                status = sampling_and_plotting(modelEnergy_type=en_models_dict[en_model]['modelEnergy_type'], en_elayers_dim=en_models_dict[en_model]['en_elayers_dim'], 
                                                pos_dim=en_models_dict[en_model]['pos_dim'], en_model_iter=en_models_dict[en_model]['en_model_iter'],
                                                modelConv_type=conv_models_dict[conv_model]['modelConv_type'],conv_elayers_dim=conv_models_dict[conv_model]['conv_elayers_dim'],
                                                temb_dim=conv_models_dict[conv_model]['temb_dim'],conv_dof=conv_models_dict[conv_model]['conv_dof'],conv_model_iter=conv_models_dict[conv_model]['conv_model_iter'],
                                                record_metrics=record_metrics, generate_plots=generate_plots, full_model_metrics=full_model_metrics, energy_intervals=energy_intervals, cuda=cuda)
            else:            
                try:
                    status = sampling_and_plotting(modelEnergy_type=en_models_dict[en_model]['modelEnergy_type'], en_elayers_dim=en_models_dict[en_model]['en_elayers_dim'], 
                                                    pos_dim=en_models_dict[en_model]['pos_dim'], en_model_iter=en_models_dict[en_model]['en_model_iter'],
                                                    modelConv_type=conv_models_dict[conv_model]['modelConv_type'],conv_elayers_dim=conv_models_dict[conv_model]['conv_elayers_dim'],
                                                    temb_dim=conv_models_dict[conv_model]['temb_dim'],conv_dof=conv_models_dict[conv_model]['conv_dof'],conv_model_iter=conv_models_dict[conv_model]['conv_model_iter'],
                                                    record_metrics=record_metrics, generate_plots=generate_plots, full_model_metrics=full_model_metrics, energy_intervals=energy_intervals, cuda=cuda)
                    print(f"{status} - Energy: {en_models_dict[en_model]['modelEnergy_type']}_{en_models_dict[en_model]['en_elayers_dim']}_{en_models_dict[en_model]['pos_dim']}_{en_models_dict[en_model]['en_elayers_dim']}, iter{en_models_dict[en_model]['en_model_iter']}")
                    print(f"{status} - Conv: {conv_models_dict[conv_model]['modelConv_type']}_{conv_models_dict[conv_model]['conv_elayers_dim']}_{conv_models_dict[conv_model]['temb_dim']}_{conv_models_dict[conv_model]['conv_dof']}, iter{conv_models_dict[conv_model]['conv_model_iter']}")
                except:
                    print(f"#----- Falha: Energy: {en_models_dict[en_model]['modelEnergy_type']}_{en_models_dict[en_model]['en_elayers_dim']}_{en_models_dict[en_model]['pos_dim']}_{en_models_dict[en_model]['en_elayers_dim']}, iter{en_models_dict[en_model]['en_model_iter']} -----#")
                    print(f"#-----------: Conv: {conv_models_dict[conv_model]['modelConv_type']}_{conv_models_dict[conv_model]['conv_elayers_dim']}_{conv_models_dict[conv_model]['temb_dim']}_{conv_models_dict[conv_model]['conv_dof']}, iter{conv_models_dict[conv_model]['conv_model_iter']} -----#")

    print("Done Sampling and Plotting!")
