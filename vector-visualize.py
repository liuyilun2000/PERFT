import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import torch


import argparse
import torch
import transformers
from transformers import (
    GenerationConfig,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from mixtral_modification.configuration_mixtral import MixtralAdapterConfig
from mixtral_modification.modeling_mixtral import (
    MixtralAdapterForCausalLM,
    MixtralAdapterModel,
)

AutoConfig.register("mixtral-adapter", MixtralAdapterConfig)
AutoModel.register(MixtralAdapterConfig, MixtralAdapterModel)
AutoModelForCausalLM.register(MixtralAdapterConfig, MixtralAdapterForCausalLM)


from olmoe_modification.configuration_olmoe import OlmoeAdapterConfig
from olmoe_modification.modeling_olmoe import (
    OlmoeAdapterForCausalLM,
    OlmoeAdapterModel,
)

AutoConfig.register("olmoe-adapter", OlmoeAdapterConfig)
AutoModel.register(OlmoeAdapterConfig, OlmoeAdapterModel)
AutoModelForCausalLM.register(OlmoeAdapterConfig, OlmoeAdapterForCausalLM)



from utils import (
    get_adapter_args,
    init_trainable_parameters,
    convert_trainable_parameters,
    print_trainable_parameters,
)

from safetensors import safe_open
from safetensors.torch import load_file


from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import umap_ as UMAP
import torch
from tqdm import tqdm



def generate_text(tokenizer, model, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def extract_layer_vectors(layer, data_type='all'):
    """
    Extract vectors from a layer based on the specified data_type.
    
    :param layer: The layer to extract vectors from
    :param data_type: String specifying which data to extract. 
                      Options: 'all', 'ffn', 'shared_expert', 'shared_routing_expert', 'embedded_expert'
    :return: Dictionary containing the extracted vectors
    """
    result = {}
    if data_type in ['all', 'ffn']:
        # Extract router vectors if gate exists
        if hasattr(layer.mlp, 'gate'):
            result['router_vectors'] = layer.mlp.gate.weight
        # Extract FFN expert vectors
        if hasattr(layer.mlp, 'experts'):
            result['ffn_expert_vectors'] = [
                expert.up_proj.weight 
                for expert in layer.mlp.experts
            ]
    if data_type in ['all', 'shared_expert']:
        # Extract shared adapter vectors if it exists
        if hasattr(layer.mlp, 'shared_adapter'):
            result['shared_adapter_vectors'] = [
                adapter.unit.lora_A.weight
                for adapter in layer.mlp.shared_adapter
            ]
    if data_type in ['all', 'shared_routing_expert']:
        # Extract shared routing vectors if it exists
        if hasattr(layer.mlp, 'shared_routing_adapter_gate'):
            result['shared_routing_gate_vectors'] = layer.mlp.shared_routing_adapter_gate.weight
            result['shared_routing_adapter_vectors'] = [
                adapter.unit.lora_A.weight 
                for adapter in layer.mlp.shared_routing_adapter
            ]
    if data_type in ['all', 'embedded_expert']:
        # Extract embedded expert vectors if it exists
        if hasattr(layer.mlp, 'gate'):
            result['router_vectors'] = layer.mlp.gate.weight
        if hasattr(layer.mlp, 'experts'):
            result['embedded_expert_vectors'] = [
                expert.embedded_routing_adapter.unit.lora_A.weight 
                for expert in layer.mlp.experts]
    return result


def prepare_data(vectors):
    """
    Prepare data from extracted vectors for visualization.
    
    :param vectors: Dictionary containing the extracted vectors
    :return: Pandas DataFrame with prepared data
    """
    all_data = []
    if 'router_vectors' in vectors:
        for i, vec in enumerate(vectors['router_vectors']):
            all_data.append({
                'vector': vec.detach().to(torch.float32).cpu().numpy(),
                'expert': i,
                'type': 'FFN Routing Gate'
            })
    if 'ffn_expert_vectors' in vectors:
        for i, vector in enumerate(vectors['ffn_expert_vectors']):
            for vec in vector:
                all_data.append({
                    'vector': vec.detach().to(torch.float32).cpu().numpy(),
                    'expert': i,
                    'type': 'FFN Expert'
                })
    if 'shared_adapter_vectors' in vectors:
        for i, vector in enumerate(vectors['shared_adapter_vectors']):
            for vec in vector:
                all_data.append({
                    'vector': vec.detach().to(torch.float32).cpu().numpy(),
                    'expert': i,
                    'type': 'Shared PEFT'
                })
    if 'shared_routing_gate_vectors' in vectors:
        for i, vec in enumerate(vectors['shared_routing_gate_vectors']):
            all_data.append({
                'vector': vec.detach().to(torch.float32).cpu().numpy(),
                'expert': i,
                'type': 'Shared Routing Gate'
            })
    if 'shared_routing_adapter_vectors' in vectors:
        for i, vector in enumerate(vectors['shared_routing_adapter_vectors']):
            for vec in vector:
                all_data.append({
                    'vector': vec.detach().to(torch.float32).cpu().numpy(),
                    'expert': i,
                    'type': 'Shared Routing PEFT'
                })
    if 'embedded_expert_vectors' in vectors:
        for i, vector in enumerate(vectors['embedded_expert_vectors']):
            for vec in vector:
                all_data.append({
                    'vector': vec.detach().to(torch.float32).cpu().numpy(),
                    'expert': i,
                    'type': 'Embedded PEFT'
                })
    return pd.DataFrame(all_data)


def load_model(base_model, peft_model):
    config_path = f"{peft_model}/config.json"
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    config.output_router_logits = False
    #
    model = OlmoeAdapterForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    #
    checkpoint_name = f"{peft_model}/model.safetensors"
    if torch.cuda.is_available():
        adapters_weights = load_file(checkpoint_name, device="cuda")
    else:
        adapters_weights = load_file(checkpoint_name)
    #
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in adapters_weights.items() if k in model_dict}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"##### Successfully loaded parameters from {checkpoint_name} #####")
    model.eval()
    return model


def load_model_and_extract_data(base_model, domain, config, layer_index):
    model_path = f"{domain}/checkpoints/OLMoE-1B-7B-0924.{config}"
    model = load_model(base_model, model_path)
    layer = model.model.layers[layer_index]
    if config.endswith('e'):
        data_type = 'embedded_expert'
    elif '-' in config:
        data_type = 'shared_routing_expert'
    else:
        data_type = 'shared_expert'
    vectors = extract_layer_vectors(layer, data_type)
    return prepare_data(vectors)

def get_vectors(df):
    return np.stack(df['vector'].values)

# Apply PCA and UMAP
def apply_projections(data, pca, umap_model):
    return umap_model.transform(pca.transform(data))




# Define marker styles for different vector types
marker_styles = {
    'FFN Routing Gate': 'X',
    'FFN Expert': '.',
    'Shared PEFT': '.',
    'Shared Routing Gate': 'X',
    'Shared Routing PEFT': '.',
    'Embedded PEFT': '.'
}
# Define size scale for different vector types
size_scale = {
    'FFN Routing Gate': 64,
    'FFN Expert': 10,
    'Shared PEFT': 32,
    'Shared Routing Gate': 64,
    'Shared Routing PEFT': 32,
    'Embedded PEFT': 32
}

alpha = {
    'FFN Routing Gate': 1,
    'FFN Expert': 0.2,
    'Shared PEFT': 0.4,
    'Shared Routing Gate': 1,
    'Shared Routing PEFT': 0.4,
    'Embedded PEFT': 0.4
}

edge_color = {
    'FFN Routing Gate': (1,1,1),
    'FFN Expert': None,
    'Shared PEFT': None,
    'Shared Routing Gate': (1,1,1),
    'Shared Routing PEFT': None,
    'Embedded PEFT': None,
}


def remove_outliers(data, threshold=1.5):
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
    return data[mask], mask



def plot_config(ax, proj, df, title, is_first_subplot=False):
    if is_first_subplot:
        # Remove outliers only for the first subplot
        proj_clean, mask = remove_outliers(proj)
        df_clean = df[mask]
    else:
        proj_clean, df_clean = proj, df
    # Get unique expert values and create a color map
    expert_values = df_clean['expert'].unique()
    color_palette = [[i for i in color] for color in sns.color_palette("hls", len(expert_values))]
    color_map = dict(zip(expert_values, color_palette))
    #
    types = list(df['type'].unique())
    gate_types = [t for t in types if 'Gate' in t]
    non_gate_types = [t for t in types if 'Gate' not in t]
    types = non_gate_types + gate_types
    #
    # Use proj_clean and df_clean instead of proj and df in the scatter plot
    for vec_type in types:
        mask = df_clean['type'] == vec_type
        ax.scatter(proj_clean[mask, 0], proj_clean[mask, 1], 
                label=vec_type, 
                alpha=alpha[vec_type],
                c=[color_map[expert] for expert in df_clean.loc[mask, 'expert']],
                edgecolors=edge_color[vec_type],
                linewidth=0.5,
                marker=marker_styles[vec_type],
                s=size_scale[vec_type]
        )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(which='both', linestyle=':', linewidth='0.5', color='gray', alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])



plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams['font.family'] = 'QTOptimum'

# Define the configurations
base_model_name = "allenai/OLMoE-1B-7B-0924"




'''

domain = "commonsense"
config_names = ["lora_32.1", "lora_32.2", "lora_32.4", "lora_32.2-2", "lora_32.4-2", "lora_32.4-4", "lora_32.8-2", "lora_32.64-8", "lora_32.e"]
titles = ['Base Model', 'S (1)', 'D (2)', 'D (4)', 'R (Top2/2)', 'R (Top2/4)', 'R (Top4/4)', 'R (Top2/8)', 'R (Top8/64)', 'E (Top8/64)']
yfigs, xfigs = 2, 5

'''
domain = "math"
config_names = ["lora_32.1", "lora_32.2", "lora_32.4", "lora_32.2-2", "lora_32.4-2", "lora_32.4-4", "lora_32.8-2", "lora_32.8-8", "lora_32.e"]
titles = ['Base Model', 'S (1)', 'D (2)', 'D (4)', 'R (Top2/2)', 'R (Top2/4)', 'R (Top4/4)', 'R (Top2/8)', 'R (Top8/8)', 'E (Top8/64)']
yfigs, xfigs = 2, 5

figsize = (xfigs*2, yfigs*2)
for layer_index in [12,13,14,15]:#tqdm(range(16)):
    # Load the base model for the first subfigure
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
    base_layer = base_model.model.layers[layer_index]
    base_vectors = extract_layer_vectors(base_layer, 'ffn')
    base_df = prepare_data(base_vectors)
    # Generate data for all configurations
    all_dfs = [base_df]  # Start with the base model data
    for config in config_names:
        df = load_model_and_extract_data(base_model_name, domain, config, layer_index)
        all_dfs.append(df)
    #
    # Prepare data, learn PCA and UMAP on the base model (all_dfs[0])
    #base_vectors = get_vectors(all_dfs[0])
    base_vectors = get_vectors(all_dfs[0][all_dfs[0]['type']=='FFN Expert'])
    scaler = StandardScaler()
    base_vectors_scaled = scaler.fit_transform(base_vectors)
    # PCA with cumulative explained ratio of 0.5
    pca = PCA(n_components=0.5, svd_solver='full')
    pca_result = pca.fit_transform(base_vectors_scaled)
    #
    print(f"Number of PCA components for 50% explained variance: {pca.n_components_}")
    #
    n_neighbors, min_dist, n_components = (20, 0.5, 2)
    umap_model = UMAP.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    umap_model.fit(pca_result)
    # Apply projections to all DataFrames
    projected_data = []
    for df in all_dfs:
        vectors = get_vectors(df)
        vectors_scaled = scaler.transform(vectors)
        projected = apply_projections(vectors_scaled, pca, umap_model)
        projected_data.append(projected)
    #
    fig, axs = plt.subplots(yfigs, xfigs, figsize=figsize, sharex=True, sharey=True)
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing
    # Plot configurations for each subplot
    for i, (proj, df, title) in enumerate(zip(projected_data, all_dfs, titles)):
        ax = axs[i]
        plot_config(ax, proj, df, title)
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{domain}_layer_{layer_index}_model_vector_projections.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{domain}_layer_{layer_index}_model_vector_projections.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()






