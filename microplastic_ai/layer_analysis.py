import numpy as np
from tensorflow.keras.models import Model
from tqdm import tqdm

def analyze_layer_usage(model, generator, layer_name, max_batches=None):
    """
    Analyzes "usage" of a layer (Conv2D or Dense) by computing mean activations.

    Args:
        model: Keras model
        generator: ImageDataGenerator iterator (train or val)
        layer_name: string, name of the layer to analyze
        max_batches: optional int, number of batches to process (None = all)

    Returns:
        usage_dict: dictionary with keys:
            'mean_activation': mean activation per neuron/filter
            'nonzero_fraction': fraction of inputs activating each neuron/filter
            'top_indices': indices of most active neurons/filters (descending)
    """
    # Create sub-model that outputs the layer activations
    layer_model = Model(inputs=model.input,
                        outputs=model.get_layer(layer_name).output)

    all_activations = []

    # Loop over generator batches
    batches_processed = 0
    for x_batch, _ in tqdm(generator, desc=f"Processing layer {layer_name}"):
        activations = layer_model.predict(x_batch, verbose=0)

        # If convolutional layer: average over spatial dimensions (H, W)
        if len(activations.shape) == 4:  # (batch, H, W, channels)
            activations = np.mean(activations, axis=(1,2))  # (batch, channels)

        all_activations.append(activations)
        batches_processed += 1
        if max_batches is not None and batches_processed >= max_batches:
            break

    all_activations = np.concatenate(all_activations, axis=0)  # (num_samples, units)

    # Metrics
    mean_activation = np.mean(all_activations, axis=0)
    nonzero_fraction = np.mean(all_activations > 1e-5, axis=0)
    top_indices = np.argsort(-mean_activation)  # descending

    usage_dict = {
        'mean_activation': mean_activation,
        'nonzero_fraction': nonzero_fraction,
        'top_indices': top_indices
    }
    return usage_dict

def summarize_layer_usage(layer_usage, threshold=0.01):
    mean_activations = layer_usage['mean_activation']

    summary = {
        "near_zero": int((mean_activations < threshold).sum()),
        "min": float(mean_activations.min()),
        "median": float(np.median(mean_activations)),
        "max": float(mean_activations.max())
    }

    return summary