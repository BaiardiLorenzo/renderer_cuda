import pandas as pd
from matplotlib import pyplot as plt


results_folder = 'csv/result-'

plot_folder = 'plots/'
plot_omp_times = 'omp_times.png'
plot_omp_speedup = 'omp_speedup.png'

plot_cuda_times = 'cuda_times.png'
plot_cuda_speedup = 'cuda_speedup.png'

plot_cuda_color_times = 'cuda_color_times.png'
plot_cuda_color_speedup = 'cuda_color_speedup.png'

plot_results_times = 'results_times.png'
plot_results_speedup = 'results_speedup.png'

test = "TEST"
t_seq = "T_SEQ"
t_par = "T_PAR"
speedup = "SPEEDUP"
t_cuda = "T_CUDA"
speedup_cuda = "SPEEDUP_CUDA"
t_cuda_color = "T_CUDA_COLOR"
speedup_cuda_color = "SPEEDUP_CUDA_COLOR"


def plot_csv(data, folder):
    # Read file CSV
    data = pd.read_csv(data, sep=';')

    # Plot OMP
    plt.figure()
    plt.plot(data[test], data[t_seq], marker='o', label='Sequential')
    for i in range(2, len(data.columns) - 4, 2):
        plt.plot(data[test], data[t_par + str(i)], marker='o', label=f'{i} Threads')
    plt.xlabel('Planes')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Times')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + plot_omp_times)
    # plt.show()

    # Plot Speedup
    plt.figure()
    for i in range(2, len(data.columns) - 4, 2):
        plt.plot(data[test], data[speedup + str(i)], marker='o', label=f'{i} Threads')
    plt.xlabel('Planes')
    plt.ylabel('Speedup')
    plt.title('Speedups')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + plot_omp_speedup)
    # plt.show()

    # Plot CUDA
    plt.figure()
    plt.plot(data[test], data[t_seq], marker='o', label='Sequential')
    plt.plot(data[test], data[t_cuda], marker='o', label=f'CUDA')
    plt.xlabel('Planes')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Times')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + plot_cuda_times)
    # plt.show()

    # Plot Speedup
    plt.figure()
    plt.plot(data[test], data[speedup_cuda], marker='o', label=f'CUDA')
    plt.xlabel('Planes')
    plt.ylabel('Speedup')
    plt.title('Speedups')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + plot_cuda_speedup)
    # plt.show()

    # Plot CUDA Color
    plt.figure()
    plt.plot(data[test], data[t_seq], marker='o', label='Sequential')
    plt.plot(data[test], data[t_cuda_color], marker='o', label=f'CUDA Color')
    plt.xlabel('Planes')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Times')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + plot_cuda_color_times)
    # plt.show()

    # Plot Speedup
    plt.figure()
    plt.plot(data[test], data[speedup_cuda_color], marker='o', label=f'CUDA Color')
    plt.xlabel('Planes')
    plt.ylabel('Speedup')
    plt.title('Speedups')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + plot_cuda_color_speedup)
    # plt.show()

    # Plot Tempi
    plt.figure()
    plt.plot(data[test], data[t_seq], marker='o', label='Sequential')
    for i in range(2, len(data.columns) - 4, 2):
        plt.plot(data[test], data[t_par + str(i)], marker='o', label=f'{i} Threads')
    plt.plot(data[test], data[t_cuda], marker='o', label=f'CUDA')
    plt.plot(data[test], data[t_cuda_color], marker='o', label=f'CUDA Color')
    plt.xlabel('Planes')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Times')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + plot_results_times)

    # Plot Speedup
    plt.figure()
    for i in range(2, len(data.columns) - 4, 2):
        plt.plot(data[test], data[speedup + str(i)], marker='o', label=f'{i} Threads')
    plt.plot(data[test], data[speedup_cuda], marker='o', label=f'CUDA')
    plt.plot(data[test], data[speedup_cuda_color], marker='o', label=f'CUDA Color')
    plt.xlabel('Planes')
    plt.ylabel('Speedup')
    plt.title('Speedups')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + plot_results_speedup)
    # plt.show()


def main():
    img_sizes = [256, 512, 1024]
    for size in img_sizes:
        data = results_folder + str(size) + '-' + str(size) + '.csv'
        plot_csv(data, plot_folder + str(size) + "/")


if __name__ == '__main__':
    main()
