from .data import get_classes, get_generators
from .model import build_model
from .train import train_model
from .fine_tune import fine_tune_model
from .analysis import compute_particle_area, compute_confidence, compute_correlation

def run_full_experiment(config):

    classes = get_classes(config)
    train_gen, val_gen = get_generators(config)

    model, base_model = build_model(config, len(classes))

    print("Phase 1: Initial training")
    history1 = train_model(model, train_gen, val_gen, config)

    print("Phase 2: Fine-tuning")
    model = fine_tune_model(
        model,
        base_model,
        unfreeze_from="mixed6",
        learning_rate=1e-4
    )

    history2 = train_model(model, train_gen, val_gen, config)

    print("Saving model...")
    model.save("final_model.keras")

    return model, history1, history2