# Step 1: Install required libraries
pip install transformers matplotlib seaborn torch bitsandbytes

# Step 2: Import libraries
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 3: Load the Unsloth Qwen3-0.6B-unsloth-bnb-4bit model
print("Loading Qwen3-0.6B-unsloth-bnb-4bit model...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-0.6B-unsloth-bnb-4bit", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    output_attentions=True,
    trust_remote_code=True,
    device_map="auto",
    load_in_4bit=True  # Enable 4-bit quantization
)

# Step 4: Define input query
query = "Run a self meta-cognitive audit where you are self-aware about being an LLM."
inputs = tokenizer(query, return_tensors='pt').to(model.device)

# Step 5: Process query and extract attention weights
print("Processing input and extracting attention weights...")
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # List of tensors (one per layer)

# Step 6: Get token labels for visualization
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Step 7: Function to visualize all heads across all layers
def visualize_all_heads_detailed(tokens, attentions, num_heads=16, num_layers=28):
    print("Generating detailed visualization...")
    
    # Create a 28x16 grid (rows=layers, cols=heads)
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(48, 84), facecolor='none')
    fig.suptitle(
        'All 16 Attention Heads Across 28 Layers (Qwen3-0.6B)\n'
        'KV Heads: Every 2 Q Heads Share 1 KV Head (e.g., H0+H1 → KV0, H2+H3 → KV1, ...)',
        fontsize=20, y=0.998, bbox=dict(facecolor='none')
    )

    for layer_idx, attention_layer in enumerate(attentions):
        for head_idx in range(num_heads):
            ax = axes[layer_idx, head_idx]
            attn = attention_layer[0, head_idx].cpu().numpy()
            
            # Determine shared KV head index (8 KV heads total)
            kv_idx = head_idx // 2  # H0+H1 → KV0, H2+H3 → KV1, etc.
            
            # Plot heatmap
            sns.heatmap(
                attn,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='viridis',
                ax=ax,
                cbar=False,
                annot=False
            )
            # Title with layer, head, and shared KV index
            ax.set_title(f'L{layer_idx+1} H{head_idx+1} (KV{kv_idx})', fontsize=10)
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('none')  # Transparent subplot

    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig('qwen_attention_detailed.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✅ Visualization saved as 'qwen_attention_detailed.png'")

# Step 8: Run visualization
visualize_all_heads_detailed(tokens, attentions)

# Step 9: Verify file was created
ls -l qwen_attention_detailed.png
