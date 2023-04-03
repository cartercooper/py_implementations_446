import numpy as np
import pandas as pd
import tracemalloc
import time


def simple_moving_average(data, window_size):

    tracemalloc.start()

    ma = data.rolling(window_size, center=True).mean()
    y_true = data[window_size:]
    y_pred = ma[window_size-1:-1]
    rmse = np.sqrt(((y_true - y_pred)**2).mean())

    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    return rmse, peak_memory


def classical_time_series_decomposition(data, period, window_size):

    tracemalloc.start()

    seasonal_means = np.array([data[i::period].mean() for i in range(period)])
    seasonal_data = np.tile(seasonal_means, len(data) // period + 1)[:len(data)]

    trend_data = data - seasonal_data
    trend_rolling_avg = trend_data.rolling(window_size).mean()

    residual_data = trend_data - trend_rolling_avg

    rmse = np.sqrt(((trend_rolling_avg[window_size-1:-1] + seasonal_data[window_size:] + residual_data[window_size:]) - data[window_size:]) ** 2).mean()

    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    return rmse, peak_memory


def polynomial_regression(data, degree):

    tracemalloc.start()

    data = pd.DataFrame(data)
    feature_size = data.shape[1]
    sample_size = data.shape[0]
    
    num_terms = int(np.power(degree + 1, feature_size - 1))
    
    x = np.zeros((sample_size, num_terms))
    y = data.iloc[:, feature_size - 1].values

    for i in range(sample_size):
        index = 0
        for comb in range(num_terms):
            temp_comb = comb
            value = 1
            
            for feature in range(feature_size - 1):
                power = temp_comb % (degree + 1)
                value *= np.power(data.iloc[i, feature], power)
                temp_comb //= (degree + 1)
                
            x[i, index] = value
            index += 1

    x_transpose = x.T
    x_transpose_x = np.matmul(x_transpose, x)
    x_transpose_y = np.matmul(x_transpose, y)

    coefficients = np.linalg.solve(x_transpose_x, x_transpose_y)

    rmse = np.sqrt(np.mean(np.square(np.matmul(x, coefficients) - y)))

    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    return rmse, peak_memory



def main():

    turbine = pd.read_csv('turbine.csv', names=['generator', 'rotor', 'dir', 't-2', 't-1','real_power']).tail(150)

    start = time.time()
    print("SMA RMSE/MEM - TURBINE - 24", simple_moving_average(turbine['real_power'], 24))
    end = time.time()
    print(end - start)

    start = time.time()
    print("SMA RMSE/MEM - TURBINE - 48", simple_moving_average(turbine['real_power'], 48))
    end = time.time()
    print(end - start)

    start = time.time()
    print("SMA RMSE/MEM - TURBINE - 72", simple_moving_average(turbine['real_power'], 72))
    end = time.time()
    print(end - start)

    start = time.time()
    print("TSD RMSE/MEM - TURBINE - 24", classical_time_series_decomposition(turbine['real_power'], 48, 24))
    end = time.time()
    print(end - start)

    start = time.time()
    print("TSD RMSE/MEM - TURBINE - 48", classical_time_series_decomposition(turbine['real_power'], 96, 24))
    end = time.time()
    print(end - start)

    start = time.time()
    print("TSD RMSE/MEM - TURBINE - 72", classical_time_series_decomposition(turbine['real_power'], 144, 24))
    end = time.time()
    print(end - start)

    start = time.time()
    print("POLY RMSE/MEM - TURBINE - 1", polynomial_regression(turbine, 1))
    end = time.time()
    print(end - start)

    start = time.time()
    print("POLY RMSE/MEM - TURBINE - 2", polynomial_regression(turbine, 2))
    end = time.time()
    print(end - start)

    start = time.time()
    print("POLY RMSE/MEM - TURBINE - 3", polynomial_regression(turbine, 3))
    end = time.time()
    print(end - start)





    dehli = pd.read_csv('dehli.csv', names=['humidity', 'temp', 'last-1', 'last', 'dew']).tail(730)

    start = time.time()
    print("SMA RMSE/MEM - DELHI - 14", simple_moving_average(dehli['dew'], 14))
    end = time.time()
    print(end - start)

    start = time.time()
    print("SMA RMSE/MEM - DELHI - 21", simple_moving_average(dehli['dew'], 21))
    end = time.time()
    print(end - start)

    start = time.time()
    print("SMA RMSE/MEM - DELHI - 14", simple_moving_average(dehli['dew'], 28))
    end = time.time()
    print(end - start)

    start = time.time()
    print("TSD RMSE/MEM - DELHI - 30", classical_time_series_decomposition(dehli['dew'], 30, 365))
    end = time.time()
    print(end - start)

    start = time.time()
    print("TSD RMSE/MEM - DELHI - 60", classical_time_series_decomposition(dehli['dew'], 60, 365))
    end = time.time()
    print(end - start)

    start = time.time()
    print("TSD RMSE/MEM - DELHI - 90", classical_time_series_decomposition(dehli['dew'], 90, 365))
    end = time.time()
    print(end - start)

    start = time.time()
    print("POLY RMSE/MEM - DELHI - 1", polynomial_regression(dehli, 1))
    end = time.time()
    print(end - start)

    start = time.time()
    print("POLY RMSE/MEM - DELHI - 2", polynomial_regression(dehli, 2))
    end = time.time()
    print(end - start)

    start = time.time()
    print("POLY RMSE/MEM - DELHI - 3", polynomial_regression(dehli, 3))
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
