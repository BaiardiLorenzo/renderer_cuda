import pandas as pd
from matplotlib import pyplot as plt

test_results = 'csv/result-512-512.csv'

plot_omp_times = 'plots/omp_times.png'
plot_omp_speedup = 'plots/omp_speedup.png'

plot_cuda_times = 'plots/cuda_times.png'
plot_cuda_speedup = 'plots/cuda_speedup.png'

plot_results_times = 'plots/results_times.png'
plot_results_speedup = 'plots/results_speedup.png'


def plot_csv(data):
    # Leggi il file CSV
    data = pd.read_csv(data, sep=';')

    # Plot OMP
    plt.figure()
    plt.plot(data['test'], data['tSeq'], marker='o', label='Sequenziale')
    for i in range(2, len(data.columns) - 2, 2):
        plt.plot(data['test'], data[f'tPar{i}'], marker='o', label=f'Parallelo {i} Thread')
    plt.xlabel('Dimensione del test')
    plt.ylabel('Tempo (secondi)')
    plt.title('Tempi di esecuzione')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_omp_times)
    # plt.show()

    # Plot Speedup
    plt.figure()
    for i in range(2, len(data.columns) - 2, 2):
        plt.plot(data['test'], data[f'speedUp{i}'], marker='o', label=f'Parallelo {i} Thread')
    plt.xlabel('Dimensione del test')
    plt.ylabel('Speedup')
    plt.title('Speedup')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_omp_speedup)
    # plt.show()

    # Plot CUDA
    plt.figure()
    plt.plot(data['test'], data['tSeq'], marker='o', label='Sequenziale')
    plt.plot(data['test'], data['tCuda'], marker='o', label=f'CUDA')
    plt.xlabel('Dimensione del test')
    plt.ylabel('Tempo (secondi)')
    plt.title('Tempi di esecuzione')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_cuda_times)
    # plt.show()

    # Plot Speedup
    plt.figure()
    plt.plot(data['test'], data['speedUpCuda'], marker='o', label=f'CUDA')
    plt.xlabel('Dimensione del test')
    plt.ylabel('Speedup')
    plt.title('Speedup')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_cuda_speedup)
    # plt.show()

    # Plot Tempi
    plt.figure()
    plt.plot(data['test'], data['tSeq'], marker='o', label='Sequenziale')
    for i in range(2, len(data.columns) - 2, 2):
        plt.plot(data['test'], data[f'tPar{i}'], marker='o', label=f'Parallelo {i} Thread')
    plt.plot(data['test'], data['tCuda'], marker='o', label=f'CUDA')
    plt.xlabel('Dimensione del test')
    plt.ylabel('Tempo (secondi)')
    plt.title('Tempi di esecuzione')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_results_times)

    # Plot Speedup
    plt.figure()
    for i in range(2, len(data.columns) - 2, 2):
        plt.plot(data['test'], data[f'speedUp{i}'], marker='o', label=f'Parallelo {i} Thread')
    plt.plot(data['test'], data['speedUpCuda'], marker='o', label=f'CUDA')
    plt.xlabel('Dimensione del test')
    plt.ylabel('Speedup')
    plt.title('Speedup')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_results_speedup)
    # plt.show()


def main():
    plot_csv(data=test_results)


if __name__ == '__main__':
    main()
