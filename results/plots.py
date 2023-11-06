import pandas as pd
from matplotlib import pyplot as plt

test_results = 'csv/test.csv'

plot_results_filename = 'plots/results.png'


def plot_csv_data(csv_filename, plot_filename):
    # Leggi il file CSV
    data = pd.read_csv(csv_filename, sep=';')

    # Crea un grafico a linee per i dati
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot Tempi
    axes[0].plot(data['test'], data['tSeq'], marker='o', label='Sequenziale')
    for i in range(2, len(data.columns) - 2, 2):
        axes[0].plot(data['test'], data[f'tPar{i}'], marker='o', label=f'Parallelo {i} Thread')
    axes[0].plot(data['test'], data['tCuda'], marker='o', label=f'CUDA')
    axes[0].set_xlabel('Dimensione del test')
    axes[0].set_ylabel('Tempo (secondi)')
    axes[0].set_title('Tempi di esecuzione')
    axes[0].legend()
    axes[0].grid()

    # Plot Speedup
    for i in range(2, len(data.columns) - 2, 2):
        axes[1].plot(data['test'], data[f'speedUp{i}'], marker='o', label=f'Parallelo {i} Thread')
    axes[1].plot(data['test'], data['speedUpCuda'], marker='o', label=f'CUDA')
    axes[1].set_xlabel('Dimensione del test')
    axes[1].set_ylabel('Speedup')
    axes[1].set_title('Speedup')
    axes[1].legend()
    axes[1].grid()

    # Mostra il grafico
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.show()


def main():
    plot_csv_data(csv_filename=test_results, plot_filename=plot_results_filename)


if __name__ == '__main__':
    main()
