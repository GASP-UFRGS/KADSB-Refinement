import argparse
import time
import torch
from plotting_functions import sampling_and_plotting


parser = argparse.ArgumentParser()

parser.add_argument('--debug', type=bool, required=True, default=False)

args = parser.parse_args()

debug = args.debug

# # !!!!!!!!!! "Lagrange_128_8_128":{"modelEnergy_type":"Lagrange","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},




# "Fast_16_8_16":{"modelEnergy_type":"Fast","en_elayers_dim":16,"pos_dim":8,"en_model_iter":[9]}
# "Chebyshev_16_8_16":{"modelEnergy_type":"Chebyshev","en_elayers_dim":16,"pos_dim":8,"en_model_iter":[9]},
# 


en_models_dict = {"Bernstein_128_8_128":{"modelEnergy_type":"Bernstein","en_elayers_dim":128,"pos_dim":8,"en_model_iter":[9]},
                  "Bernstein_128_64_128":{"modelEnergy_type":"Bernstein","en_elayers_dim":128,"pos_dim":64,"en_model_iter":[9]},
                  "Bottleneck_128_64_128":{"modelEnergy_type":"Bottleneck","en_elayers_dim":128,"pos_dim":64,"en_model_iter":[9]},
                  "Chebyshev_128_64_128":{"modelEnergy_type":"Chebyshev","en_elayers_dim":128,"pos_dim":64,"en_model_iter":[9]},
                  "Fast_128_64_128":{"modelEnergy_type":"Fast","en_elayers_dim":128,"pos_dim":64,"en_model_iter":[9]},
                  "Gram_128_64_128":{"modelEnergy_type":"Gram","en_elayers_dim":128,"pos_dim":64,"en_model_iter":[9]},
                  "Jacobi_128_8_128":{"modelEnergy_type":"Jacobi","en_elayers_dim":128,"pos_dim":8,"en_model_iter":[9]},
                  "Jacobi_128_64_128":{"modelEnergy_type":"Jacobi","en_elayers_dim":128,"pos_dim":64,"en_model_iter":[9]},
                  "Jacobi_256_128_256":{"modelEnergy_type":"Jacobi","en_elayers_dim":256,"pos_dim":128,"en_model_iter":[9]},
                  "Lagrange_128_8_128":{"modelEnergy_type":"Lagrange","en_elayers_dim":128,"pos_dim":8,"en_model_iter":[9]},
                  "Lagrange_128_64_128":{"modelEnergy_type":"Lagrange","en_elayers_dim":128,"pos_dim":64,"en_model_iter":[9]},
                  "ReLU_128_8_128":{"modelEnergy_type":"ReLU","en_elayers_dim":128,"pos_dim":8,"en_model_iter":[9]},
                  "ReLU_128_64_128":{"modelEnergy_type":"ReLU","en_elayers_dim":128,"pos_dim":64,"en_model_iter":[9]},
                  "Fast_256_32_256":{"modelEnergy_type":"Fast","en_elayers_dim":256,"pos_dim":32,"en_model_iter":[9]},
                  "Fast_256_8_256":{"modelEnergy_type":"Fast","en_elayers_dim":256,"pos_dim":8,"en_model_iter":[9]},
                  "Fast_128_128_128":{"modelEnergy_type":"Fast","en_elayers_dim":128,"pos_dim":128,"en_model_iter":[9]},
                  "Fast_128_32_128":{"modelEnergy_type":"Fast","en_elayers_dim":128,"pos_dim":32,"en_model_iter":[9]},
                  "Fast_16_128_16":{"modelEnergy_type":"Fast","en_elayers_dim":16,"pos_dim":128,"en_model_iter":[9]},
                  "Fast_16_32_16":{"modelEnergy_type":"Fast","en_elayers_dim":16,"pos_dim":32,"en_model_iter":[9]},
                  "Fast_16_8_16":{"modelEnergy_type":"Fast","en_elayers_dim":16,"pos_dim":8,"en_model_iter":[9]},
                  "Chebyshev_16_8_16":{"modelEnergy_type":"Chebyshev","en_elayers_dim":16,"pos_dim":8,"en_model_iter":[9]},
                  "SQuIRELS_256_128_256":{"modelEnergy_type":"SQuIRELS","en_elayers_dim":256,"pos_dim":128,"en_model_iter":[9]}
                  }
                    

conv_models_dict = {"Bottleneck_16_8_2":{"modelConv_type":"Bottleneck", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":2, "conv_model_iter":[9]},
                    "Bottleneck_128_8_2":{"modelConv_type":"Bottleneck", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":2, "conv_model_iter":[9]},
                    "Bottleneck_256_128_2":{"modelConv_type":"Bottleneck", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":2, "conv_model_iter":[9]},
                    "BottleneckAttention_16_8_1":{"modelConv_type":"BottleneckAttention", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":1, "conv_model_iter":[9]},
                    "Chebyshev_16_8_8":{"modelConv_type":"Chebyshev", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":8, "conv_model_iter":[9]},
                    "Chebyshev_128_8_4":{"modelConv_type":"Chebyshev", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":[9]},
                    "Chebyshev_256_128_8":{"modelConv_type":"Chebyshev", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":8, "conv_model_iter":[9]},
                    "Fast_16_8_4":{"modelConv_type":"Fast","conv_elayers_dim":16,"temb_dim":8,"conv_dof":4,"conv_model_iter":[9]},
                    "Fast_128_8_4":{"modelConv_type":"Fast", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":[9]},
                    "Fast_256_128_4":{"modelConv_type":"Fast", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":4, "conv_model_iter":[9]},
                    "ReLU_16_8_4":{"modelConv_type":"ReLU","conv_elayers_dim":16,"temb_dim":8,"conv_dof":4,"conv_model_iter":[9]},
                    "ReLU_128_8_4":{"modelConv_type":"ReLU", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":[9]},
                    "ReLU_256_128_4":{"modelConv_type":"ReLU", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":4, "conv_model_iter":[9]},
                    "Bernstein_16_8_1":{"modelConv_type":"Bernstein","conv_elayers_dim":16,"temb_dim":8,"conv_dof":1,"conv_model_iter":[9]},
                    "Gram_16_8_1":{"modelConv_type":"Gram", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":1, "conv_model_iter":[9]},
                    "Jacobi_16_8_1":{"modelConv_type":"Jacobi","conv_elayers_dim":16,"temb_dim":8,"conv_dof":1,"conv_model_iter":[9]},
                    "Lagrange_16_8_1":{"modelConv_type":"Lagrange", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":1, "conv_model_iter":[9]},
                    "SQuIRELS_256_128_32":{"modelConv_type":"SQuIRELS", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":32, "conv_model_iter":[9]}
                    }

sel_en_models = en_models_dict.keys()#["Chebyshev_16_8_16"]#["SQuIRELS_256_128_256","Fast_16_8_16"] en_models_dict.keys()
sel_conv_models = conv_models_dict.keys()#["SQuIRELS_256_128_32"]#

# "Chebyshev_16_8_16",
# "Lagrange_16_8_1",

en_models_list = ["Fast_16_8_16","SQuIRELS_256_128_256","SQuIRELS_256_128_256"]
conv_models_list = ["Bottleneck_256_128_2","Chebyshev_128_8_4","SQuIRELS_256_128_32"]

record_metrics = True
generate_plots = False
full_model_metrics = True
energy_intervals = False
cuda = True

if __name__ == "__main__":

    # for en_model in sel_en_models:
    #     for conv_model in sel_conv_models:
    #         torch.cuda.empty_cache()
    #         if debug:
    #             status = sampling_and_plotting(modelEnergy_type=en_models_dict[en_model]['modelEnergy_type'], en_elayers_dim=en_models_dict[en_model]['en_elayers_dim'], 
    #                                             pos_dim=en_models_dict[en_model]['pos_dim'], en_model_iter=en_models_dict[en_model]['en_model_iter'],
    #                                             modelConv_type=conv_models_dict[conv_model]['modelConv_type'],conv_elayers_dim=conv_models_dict[conv_model]['conv_elayers_dim'],
    #                                             temb_dim=conv_models_dict[conv_model]['temb_dim'],conv_dof=conv_models_dict[conv_model]['conv_dof'],conv_model_iter=conv_models_dict[conv_model]['conv_model_iter'],
    #                                             record_metrics=record_metrics, generate_plots=generate_plots, full_model_metrics=full_model_metrics, energy_intervals=energy_intervals, cuda=cuda,
    #                                             abs_path='/mnt/f/UFRGS/TCC/Dados', ex=True, ey=True)
    #         else:            
    #             try:
    #                 status = sampling_and_plotting(modelEnergy_type=en_models_dict[en_model]['modelEnergy_type'], en_elayers_dim=en_models_dict[en_model]['en_elayers_dim'], 
    #                                                 pos_dim=en_models_dict[en_model]['pos_dim'], en_model_iter=en_models_dict[en_model]['en_model_iter'],
    #                                                 modelConv_type=conv_models_dict[conv_model]['modelConv_type'],conv_elayers_dim=conv_models_dict[conv_model]['conv_elayers_dim'],
    #                                                 temb_dim=conv_models_dict[conv_model]['temb_dim'],conv_dof=conv_models_dict[conv_model]['conv_dof'],conv_model_iter=conv_models_dict[conv_model]['conv_model_iter'],
    #                                                 record_metrics=record_metrics, generate_plots=generate_plots, full_model_metrics=full_model_metrics, energy_intervals=energy_intervals, cuda=cuda,
    #                                                 abs_path='/mnt/f/UFRGS/TCC/Dados', ex=True, ey=True)
    #                 print(f"{status} - Energy: {en_models_dict[en_model]['modelEnergy_type']}_{en_models_dict[en_model]['en_elayers_dim']}_{en_models_dict[en_model]['pos_dim']}_{en_models_dict[en_model]['en_elayers_dim']}, iter{en_models_dict[en_model]['en_model_iter']}")
    #                 print(f"{status} - Conv: {conv_models_dict[conv_model]['modelConv_type']}_{conv_models_dict[conv_model]['conv_elayers_dim']}_{conv_models_dict[conv_model]['temb_dim']}_{conv_models_dict[conv_model]['conv_dof']}, iter{conv_models_dict[conv_model]['conv_model_iter']}")
    #             except:
    #                 print(f"#----- Falha: Energy: {en_models_dict[en_model]['modelEnergy_type']}_{en_models_dict[en_model]['en_elayers_dim']}_{en_models_dict[en_model]['pos_dim']}_{en_models_dict[en_model]['en_elayers_dim']}, iter{en_models_dict[en_model]['en_model_iter']} -----#")
    #                 print(f"#-----------: Conv: {conv_models_dict[conv_model]['modelConv_type']}_{conv_models_dict[conv_model]['conv_elayers_dim']}_{conv_models_dict[conv_model]['temb_dim']}_{conv_models_dict[conv_model]['conv_dof']}, iter{conv_models_dict[conv_model]['conv_model_iter']} -----#")

    # print("Done Sampling and Plotting!")



    for i in range(len(en_models_list)):
        en_model = en_models_dict[en_models_list[i]]
        conv_model = conv_models_dict[conv_models_list[i]]
        # torch.cuda.empty_cache()
        if debug:
            status = sampling_and_plotting(modelEnergy_type=en_model['modelEnergy_type'], en_elayers_dim=en_model['en_elayers_dim'], 
                                            pos_dim=en_model['pos_dim'], en_model_iter=en_model['en_model_iter'],
                                            modelConv_type=conv_model['modelConv_type'],conv_elayers_dim=conv_model['conv_elayers_dim'],
                                            temb_dim=conv_model['temb_dim'],conv_dof=conv_model['conv_dof'],conv_model_iter=conv_model['conv_model_iter'],
                                            record_metrics=record_metrics, generate_plots=generate_plots, full_model_metrics=full_model_metrics, energy_intervals=energy_intervals, cuda=cuda,
                                            abs_path='/mnt/f/UFRGS/TCC/Dados', ex=True, ey=True)
        else:            
            try:
                status = sampling_and_plotting(modelEnergy_type=en_model['modelEnergy_type'], en_elayers_dim=en_model['en_elayers_dim'], 
                                                pos_dim=en_model['pos_dim'], en_model_iter=en_model['en_model_iter'],
                                                modelConv_type=conv_model['modelConv_type'],conv_elayers_dim=conv_model['conv_elayers_dim'],
                                                temb_dim=conv_model['temb_dim'],conv_dof=conv_model['conv_dof'],conv_model_iter=conv_model['conv_model_iter'],
                                                record_metrics=record_metrics, generate_plots=generate_plots, full_model_metrics=full_model_metrics, energy_intervals=energy_intervals, cuda=cuda,
                                                abs_path='/mnt/f/UFRGS/TCC/Dados', ex=True, ey=True)
                print(f"{status} - Energy: {en_model['modelEnergy_type']}_{en_model['en_elayers_dim']}_{en_model['pos_dim']}_{en_model['en_elayers_dim']}, iter{en_model['en_model_iter']}")
                print(f"{status} - Conv: {conv_model['modelConv_type']}_{conv_model['conv_elayers_dim']}_{conv_model['temb_dim']}_{conv_model['conv_dof']}, iter{conv_model['conv_model_iter']}")
            except:
                print(f"#----- Falha: Energy: {en_model['modelEnergy_type']}_{en_model['en_elayers_dim']}_{en_model['pos_dim']}_{en_model['en_elayers_dim']}, iter{en_model['en_model_iter']} -----#")
                print(f"#-----------: Conv: {conv_model['modelConv_type']}_{conv_model['conv_elayers_dim']}_{conv_model['temb_dim']}_{conv_model['conv_dof']}, iter{conv_model['conv_model_iter']} -----#")

    print("Done Sampling and Plotting!")
