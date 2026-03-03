from PolyVision.microplastic_ai.pipeline import run_full_experiment
from PolyVision.microplastic_ai.config import ExperimentConfig

config = ExperimentConfig(
    training_type="microplastic",
    learning_rate=0.0001,
    base_dirs={
        "whisky": r"C:\Users\joshk\OneDrive - University of Strathclyde\AI Microplastics Data\datasets\whisky_webs\v1",
        "microplastic": r"C:\Users\joshk\OneDrive - University of Strathclyde\AI Microplastics Data\datasets\v1.14"
    },
    microplastic_classes=[
        'pet', 'rubber', 'pp', 'pvc', 'neoprene', 'nylon', 'pe',
        'pla', 'pmma', 'pu', 'ps'
    ],
    whisky_classes=[
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
        'Q','R','S','T','U','V','W','Y','Z','AA','AB','AC','AD'
    ],
    weights_path=r"/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
    save_path=r"C:\Users\joshk\OneDrive - University of Strathclyde\AI Microplastics Data\results\v1.19",
    epochs=15
)

model, h1, h2 = run_full_experiment(config)