import pandas as pd
import os

def train_model(model, train_gen, val_gen, config):
    history = model.fit(
        train_gen,
        epochs=config.epochs,
        validation_data=val_gen,
        verbose=2
    )

    os.makedirs(config.save_path, exist_ok=True)

    pd.DataFrame(history.history).to_csv(
        os.path.join(config.save_path, "history.csv"),
        index=False
    )

    return history