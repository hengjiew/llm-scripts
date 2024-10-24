import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse

def plot(batch_size, df_batch, name):
    fig = plt.figure()  # Create a new figure for each call
    ax1 = fig.add_subplot(111)

    categories = df_batch['Seq Length']
    index = np.arange(len(categories))

    bar_width=0.4
    ax1.bar(index, df_batch['flashattn'], width=bar_width, label='flashattn', align='center')
    ax1.bar(index+bar_width, df_batch['flashinfer'], width=bar_width, label='flashinfer', align='center')

    ax1.set_xlabel('Seq Length')
    ax1.set_ylabel('Latency (µs)')
    ax1.set_title('Batch Size 1')
    ax1.legend(loc='upper left')

    ax1.set_xticks(index)  # Setting the tick positions first
    ax1.set_xticklabels([f"{list(categories)[v]}" for v in index])
    plt.xticks(rotation=45)

    # Create another axis for Speedup
    ax2 = ax1.twinx()
    ax2.plot(index + bar_width / 2, df_batch['Speedup'], color='red', marker='o', label='flashinfer Speedup over flashattn', linestyle='--')
    ax2.set_ylabel('Speedup')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.savefig(Path(name)/(f"{name}_bs{batch_size}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the benchmark results.")
    parser.add_argument("--name", type=str, default="flashattn2.6.2_vs_flashinfer0.1.6", help="Name of the benchmark")
    args = parser.parse_args()
    print(args)
    df = pd.read_csv(Path(args.name)/(args.name+'.csv'))
    # Index(['Batch Size', 'Seq Length', 'flashattn', 'flashinfer', 'Speedup'], dtype='object')
    for batch_size in df["Batch Size"].unique():
        df_batch = df[df['Batch Size'] == batch_size]
        plot(batch_size, df_batch, args.name)
