import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union


def func(x1: Union[np.ndarray, pd.Series], 
        x2: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    return 3 * x1 / (1 - np.exp(-x2))


def generate_csv(size: int, file_path: str) -> None:
    x1 = np.linspace(0.1, 3, size)
    x2 = np.linspace(10, 20, size)
    y = func(x1, x2)
    pd.DataFrame({"y": y, "x1": x1, "x2": x2}).to_csv(f"{file_path}", index=False)


def draw_plot(xlabel: str, title: str, x: pd.Series, y: Union[np.ndarray, pd.Series]) -> None:
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color="green", edgecolors='black', linewidths=0.5)
    plt.plot(x, y, color='red')
    plt.xlabel(xlabel)
    plt.ylabel("y")
    plt.title(title)
    plt.grid()
    plt.show()


def func_slices(data: pd.DataFrame) -> None:
    x2 = data["x2"].iloc[len(data) // 2]
    x1 = data["x1"]
    y = func(x1, x2)
    draw_plot("x1", f"y(x1) and x2 = const ({x2})", x1, y)

    x1 = data["x1"].iloc[len(data) // 2]
    x2 = data["x2"]
    y = func(x1, x2)
    draw_plot("x2", f"y(x2) and x1 = const ({x1})", x2, y)


def get_stats(data: pd.DataFrame) -> tuple[float, float]:
    mean_x1 = data["x1"].mean()
    mean_x2 = data["x2"].mean()
    print(f"Means:\nx1: {mean_x1}; x2: {mean_x2}; y: {data['y'].mean()}\n")
    print(f"Mins:\nx1: {data['x1'].min()}; x2: {data['x2'].min()}; y: {data['y'].min()}\n")
    print(f"Maxs:\nx1: {data['x1'].max()}; x2: {data['x2'].max()}; y: {data['y'].max()}\n")

    return mean_x1, mean_x2


def generate_csv_by_condition(file_path: str, data: pd.DataFrame) -> None:
    mean_x1, mean_x2 = get_stats(data)
    new_data = data[(data["x1"] < mean_x1) | (data["x2"] < mean_x2)]
    new_data.to_csv(file_path, index=False)


def draw_3D_plot(data: pd.DataFrame) -> None:
    x1 = np.array(data['x1'])
    x2 = np.array(data['x2'])
    x1, x2 = np.meshgrid(x1, x2)
    y = func(x1, x2)
    
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x1, x2, y, cmap='viridis')
    plt.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def main() -> None:
    file_path_1 = "lab2/data.csv"
    file_path_2 = "lab2/new_data.csv"

    generate_csv(500, file_path_1)
    data = pd.read_csv(file_path_1)
    func_slices(data)
    generate_csv_by_condition(file_path_2, data)
    draw_3D_plot(data)


if __name__ == "__main__":
    main()