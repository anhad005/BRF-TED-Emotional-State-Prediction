"""
main.py

BRF-TED Emotional State Prediction

Pipeline:
1. Generate synthetic dataset
2. Train/Test split
3. BRF feature extraction
4. Sequence generation
5. TED training
6. Evaluation
"""

import pandas as pd

from src.config import (
    TRAIN_IDS,
    TEST_IDS,
    HIDDEN_DIM
)

from src.synthetic_data import (
    generate_synthetic
)

from src.brf import (
    BRFExtractor,
    FEATURE_COLS
)

from src.ted_model import (
    TEDModel
)

from src.train import (
    prepare_dataloaders,
    train_model
)


def main():

    print("=" * 60)
    print("BRF-TED Emotional State Prediction")
    print("=" * 60)

    # ---------------------------------------------------------
    # Generate Dataset
    # ---------------------------------------------------------

    print("\nGenerating synthetic dataset...")

    df = generate_synthetic()

    print(f"Dataset Shape: {df.shape}")

    # ---------------------------------------------------------
    # Train/Test Split
    # ---------------------------------------------------------

    train_df = (
        df[df["student_id"].isin(TRAIN_IDS)]
        .reset_index(drop=True)
    )

    test_df = (
        df[df["student_id"].isin(TEST_IDS)]
        .reset_index(drop=True)
    )

    print(
        f"\nTrain Samples: {len(train_df):,}"
    )

    print(
        f"Test Samples: {len(test_df):,}"
    )

    # ---------------------------------------------------------
    # BRF Feature Extraction
    # ---------------------------------------------------------

    print("\nApplying BRF normalization...")

    brf = BRFExtractor()

    X_train = brf.fit_transform(
        train_df
    )

    X_test = brf.transform(
        test_df
    )

    y_train = (
        train_df["label"]
        .values
    )

    y_test = (
        test_df["label"]
        .values
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    print(
        "\nPreparing sequence data..."
    )

    (
        loader_train,
        loader_test,
        Xs_tr,
        ys_tr,
        Xs_te,
        ys_te
    ) = prepare_dataloaders(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print(
        f"Train Sequences: {Xs_tr.shape}"
    )

    print(
        f"Test Sequences : {Xs_te.shape}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print(
        "\nInitializing TED model..."
    )

    model = TEDModel(
        input_dim=len(FEATURE_COLS),
        hidden_dim=HIDDEN_DIM
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print(
        "\nTraining model..."
    )

    results, history, _ = train_model(
        model=model,
        loader_train=loader_train,
        loader_test=loader_test,
        label="LSTM-TED"
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for k, v in results.items():
        print(f"{k:<12}: {v}")

    # ---------------------------------------------------------
    # Save Outputs
    # ---------------------------------------------------------

    history.to_csv(
        "training_history.csv",
        index=False
    )

    pd.DataFrame(
        [results]
    ).to_csv(
        "results.csv",
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        "  training_history.csv"
    )

    print(
        "  results.csv"
    )

    print(
        "\nPipeline Complete."
    )


if __name__ == "__main__":
    main()