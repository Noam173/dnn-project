import Data_Manipulation as dp
from Create_model import create_model as model
from plot import plot_training as plt
import encode_data as encode
import Reset_data as reset


def main() -> None:
    reset

    path = dp.create_directory()
    file = "~/train_features.csv"
    file = encode.encoder(file)
    dp.Split_Train(file)

    x = {128, 32}

    for batchs in x:
        hist = model(batchs, 50)
        plt(hist, batchs)


if __name__ == "__main__":
    main()
