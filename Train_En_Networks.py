import argparse
import torch
from pathlib import Path
from functions import train_en_network


parser = argparse.ArgumentParser()

parser.add_argument('--debug', type=bool, required=True, default=False)

args = parser.parse_args()

debug = args.debug


# "Bernstein_128_8_128":{"modelEnergy_type":"Bernstein","en_elayers_dim":128,"pos_dim":8},
# "Bernstein_128_64_128":{"modelEnergy_type":"Bernstein","en_elayers_dim":128,"pos_dim":64},
# "Bottleneck_128_64_128":{"modelEnergy_type":"Bottleneck","en_elayers_dim":128,"pos_dim":64},
# "Chebyshev_128_64_128":{"modelEnergy_type":"Chebyshev","en_elayers_dim":128,"pos_dim":64},
# "Fast_128_64_128":{"modelEnergy_type":"Fast","en_elayers_dim":128,"pos_dim":64},
# "Gram_128_64_128":{"modelEnergy_type":"Gram","en_elayers_dim":128,"pos_dim":64},
# "Jacobi_128_8_128":{"modelEnergy_type":"Jacobi","en_elayers_dim":128,"pos_dim":8},
# "Jacobi_128_64_128":{"modelEnergy_type":"Jacobi","en_elayers_dim":128,"pos_dim":64},
# "Jacobi_256_128_256":{"modelEnergy_type":"Jacobi","en_elayers_dim":256,"pos_dim":128},
# "Lagrange_128_8_128":{"modelEnergy_type":"Lagrange","en_elayers_dim":128,"pos_dim":8},
# "Lagrange_128_64_128":{"modelEnergy_type":"Lagrange","en_elayers_dim":128,"pos_dim":64},
# "ReLU_128_8_128":{"modelEnergy_type":"ReLU","en_elayers_dim":128,"pos_dim":8},
# "ReLU_128_64_128":{"modelEnergy_type":"ReLU","en_elayers_dim":128,"pos_dim":64},
# "Bernstein_16_8_16":{"modelEnergy_type":"Bernstein","en_elayers_dim":16,"pos_dim":8},
# "Chebyshev_16_8_16":{"modelEnergy_type":"Chebyshev","en_elayers_dim":16,"pos_dim":8},
# "Gram_16_8_16":{"modelEnergy_type":"Gram","en_elayers_dim":16,"pos_dim":8},
# "Jacobi_16_8_16":{"modelEnergy_type":"Jacobi","en_elayers_dim":16,"pos_dim":8},
# "Lagrange_16_8_16":{"modelEnergy_type":"Lagrange","en_elayers_dim":16,"pos_dim":8}
                                 
                  

en_models_dict = {"Fast_256_32_256":{"modelEnergy_type":"Fast","en_elayers_dim":256,"pos_dim":32},
                  "Fast_256_8_256":{"modelEnergy_type":"Fast","en_elayers_dim":256,"pos_dim":8},
                  "Fast_128_128_128":{"modelEnergy_type":"Fast","en_elayers_dim":128,"pos_dim":128},
                  "Fast_128_32_128":{"modelEnergy_type":"Fast","en_elayers_dim":128,"pos_dim":32},
                  "Fast_16_128_16":{"modelEnergy_type":"Fast","en_elayers_dim":16,"pos_dim":128},
                  "Fast_16_32_16":{"modelEnergy_type":"Fast","en_elayers_dim":16,"pos_dim":32}}

sel_en_models = en_models_dict.keys()

if __name__ == "__main__":

    for en_model in sel_en_models:
        torch.cuda.empty_cache()
        if debug:
            status = train_en_network(en_models_dict[en_model]['modelEnergy_type'], en_models_dict[en_model]['en_elayers_dim'], en_models_dict[en_model]['pos_dim'], n_iter=10)
        else:            
            try:
                status = status = train_en_network(en_models_dict[en_model]['modelEnergy_type'], en_models_dict[en_model]['en_elayers_dim'], en_models_dict[en_model]['pos_dim'], n_iter=10)
                print(f"{status} - Energy: {en_models_dict[en_model]['modelEnergy_type']}_{en_models_dict[en_model]['en_elayers_dim']}_{en_models_dict[en_model]['pos_dim']}_{en_models_dict[en_model]['en_elayers_dim']}, iter{en_models_dict[en_model]['en_model_iter']}")
            except:
                print(f"#----- Falha: Energy: {en_models_dict[en_model]['modelEnergy_type']}_{en_models_dict[en_model]['en_elayers_dim']}_{en_models_dict[en_model]['pos_dim']}_{en_models_dict[en_model]['en_elayers_dim']}, iter{en_models_dict[en_model]['en_model_iter']} -----#")

    print("Done!")