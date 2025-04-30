# Install required libraries
pip install transformers matplotlib seaborn

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Define a sample query
query = "PLACEHOLDER FOR YOUR QUERY"
inputs = tokenizer(query, return_tensors='pt')

# Process the query through the model
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # List of attention tensors (one per layer)

# Get token labels for visualization
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Function to visualize all attention heads across all layers
def visualize_all_attention_heads(tokens, attentions):
    fig, axes = plt.subplots(12, 12, figsize=(60, 60), facecolor='none')
    fig.suptitle('All Attention Heads from All Layers (12 Layers × 12 Heads)',
                 fontsize=20, bbox=dict(facecolor='none'))

    for layer_idx, attention_layer in enumerate(attentions):
        for head_idx in range(12):  # BERT base has 12 heads per layer
            ax = axes[layer_idx, head_idx]
            attn = attention_layer[0, head_idx].numpy()
            sns.heatmap(
                attn,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='viridis',
                ax=ax,
                cbar=False,
                annot=False
            )
            ax.set_title(f'L{layer_idx+1} H{head_idx+1}', fontsize=8)
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('none')  # Make individual subplot background transparent

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save as transparent PNG
    plt.savefig('attention_heads.png', format='png', dpi=300, transparent=True, bbox_inches='tight')

    # Display the image in Colab
    plt.show()
    plt.close()

# Call the visualization function
visualize_all_attention_heads(tokens, attentions)

# Verify file was created
ls -l attention_heads.png