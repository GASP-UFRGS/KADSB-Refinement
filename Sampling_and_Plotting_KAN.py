from plotting_functions import sampling_and_plotting


en_models_dict = [{"modelEnergy_type":"Bottleneck","en_elayers_dim":256,"pos_dim":128,"en_model_iter":-1},
                  {"modelEnergy_type":"Bottleneck","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},
                  {"modelEnergy_type":"Bottleneck","en_elayers_dim":16,"pos_dim":8,"en_model_iter":-1},
                  {"modelEnergy_type":"Chebyshev","en_elayers_dim":256,"pos_dim":128,"en_model_iter":-1},
                  {"modelEnergy_type":"Chebyshev","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},
                  {"modelEnergy_type":"Fast","en_elayers_dim":256,"pos_dim":128,"en_model_iter":-1},
                  {"modelEnergy_type":"Fast","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},
                  {"modelEnergy_type":"Fast","en_elayers_dim":16,"pos_dim":8,"en_model_iter":-1},
                  {"modelEnergy_type":"Gram","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},
                  {"modelEnergy_type":"Lagrange","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},
                  {"modelEnergy_type":"ReLU","en_elayers_dim":256,"pos_dim":128,"en_model_iter":-1},
                  {"modelEnergy_type":"ReLU","en_elayers_dim":16,"pos_dim":8,"en_model_iter":-1},
                  {"modelEnergy_type":"SQuIRELS","en_elayers_dim":256,"pos_dim":128,"en_model_iter":-1},
                  {"modelEnergy_type":"SQuIRELS","en_elayers_dim":16,"pos_dim":8,"en_model_iter":-1},
                  {"modelEnergy_type":"Wav","en_elayers_dim":128,"pos_dim":8,"en_model_iter":-1},
                  {"modelEnergy_type":"Wav","en_elayers_dim":16,"pos_dim":8,"en_model_iter":-1}]


conv_models_dict = [{"modelConv_type":"Bottleneck", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":2, "conv_model_iter":-1},
                    {"modelConv_type":"Bottleneck", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":2, "conv_model_iter":-1},
                    {"modelConv_type":"BottleneckKAGNAttentionConv", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":1, "conv_model_iter":-1},
                    {"modelConv_type":"BottleneckKAGNLinear", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":32, "conv_model_iter":-1},
                    {"modelConv_type":"BottleneckKAGNLinear", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    {"modelConv_type":"Chebyshev", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":8, "conv_model_iter":-1},
                    {"modelConv_type":"Chebyshev", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    {"modelConv_type":"Fast", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":4, "conv_model_iter":-1},
                    {"modelConv_type":"Fast", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    {"modelConv_type":"FastLinear", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":32, "conv_model_iter":-1},
                    {"modelConv_type":"FastLinear", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    {"modelConv_type":"FastWide", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":16, "conv_model_iter":-1},
                    {"modelConv_type":"FastWide", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    {"modelConv_type":"Gram", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":1, "conv_model_iter":-1},
                    {"modelConv_type":"Lagrange", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":1, "conv_model_iter":-1},
                    {"modelConv_type":"ReLU", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":4, "conv_model_iter":-1},
                    {"modelConv_type":"ReLU", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    {"modelConv_type":"ReLULinear", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    {"modelConv_type":"SQuIRELS", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":32, "conv_model_iter":-1},
                    {"modelConv_type":"SQuIRELS", "conv_elayers_dim":16, "temb_dim":8, "conv_dof":2, "conv_model_iter":-1},
                    {"modelConv_type":"SQuIRELSLinear", "conv_elayers_dim":256, "temb_dim":128, "conv_dof":32, "conv_model_iter":-1},
                    {"modelConv_type":"SQuIRELSLinear", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":4, "conv_model_iter":-1},
                    {"modelConv_type":"Wav", "conv_elayers_dim":128, "temb_dim":8, "conv_dof":2, "conv_model_iter":-1}]


if __name__ == "__main__":

    for model in en_models_dict:
        try:
            status = sampling_and_plotting(model['modelEnergy_type'], model['en_elayers_dim'], model['pos_dim'], model['en_model_iter'],
                                        modelConv_type="SQuIRELS",temb_dim=16,conv_elayers_dim=8,conv_dof=2,conv_model_iter=19,
                                        record_metrics=True,generate_plots=True,full_model_metrics=False)
            print(f"{status} - Modelo {model['modelEnergy_type']}_{model['en_elayers_dim']}_{model['pos_dim']}_{model['en_elayers_dim']}, iter{model['en_model_iter']}")
        except:
            print(f"#----- Falha: Modelo {model['modelEnergy_type']}_{model['en_elayers_dim']}_{model['pos_dim']}_{model['en_elayers_dim']}, iter{model['en_model_iter']} -----#")

    print("Done Sampling and Plotting!")
