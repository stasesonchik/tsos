import numpy as np
import matplotlib.pyplot as plt
import time


# =========================================================
# 1. Функция "медленного" ДПФ (DFT slow)
# =========================================================
def DFT_slow(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


# =========================================================
# 2. Генерация сигнала косинусов
# =========================================================
def generate_cosine_signal(frequencies, amplitudes, fs, duration):
    t = np.arange(0, duration, 1 / fs)
    x = np.zeros_like(t)
    for f, A in zip(frequencies, amplitudes):
        x += A * np.cos(2 * np.pi * f * t)
    return t, x


# =========================================================
# 3. Добавление белого шума
# =========================================================
def add_noise(x, sigma=0.5):
    noise = np.random.normal(0, sigma, x.shape)
    return x + noise


# =========================================================
# 4. Построение спектра
# =========================================================
def plot_spectrum(X, fs, title="Спектр сигнала"):
    N = len(X)
    freq = np.fft.fftfreq(N, 1 / fs)
    mask = freq >= 0
    plt.figure()
    plt.plot(freq[mask], np.abs(X[mask]))
    plt.title(title)
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.show()


# =========================================================
# 5. Основная функция
# =========================================================
def main():
    # --- параметры
    fs = 1000  # частота дискретизации
    duration = 1  # длительность 1 секунда
    frequencies = [50, 150]
    amplitudes = [1, 0.5]

    # --- генерируем сигнал
    t, x = generate_cosine_signal(frequencies, amplitudes, fs, duration)

    # --- 5a: DFT slow
    start = time.time()
    X_slow = DFT_slow(x)
    t_slow = time.time() - start
    print(f"Время вычисления DFT_slow: {t_slow:.4f} сек")

    # --- график медленного преобразования
    plot_spectrum(X_slow, fs, "Спектр сигнала (DFT slow)")

    # --- 5a: FFT
    start = time.time()
    X_fft = np.fft.fft(x)
    t_fft = time.time() - start
    print(f"Время вычисления FFT: {t_fft:.4f} сек")




    # --- график FFT
    plot_spectrum(X_fft, fs, "Спектр сигнала (FFT)")

    # --- график исходного сигнала как сумма косинусоид
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="x(t) = cos(2π·50t) + 0.5·cos(2π·150t)")
    plt.title("Исходный сигнал: сумма двух косинусоид")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.legend()
    plt.show()
    # --- 5b: обратное преобразование FFT
    x_rec = np.fft.ifft(X_fft)
    plt.figure()
    plt.plot(t, x, label="Исходный сигнал")
    plt.plot(t, x_rec.real, '--', label="Восстановленный сигнал")
    plt.title("Обратное преобразование FFT")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 5c: добавление шума
    x_noise = add_noise(x, sigma=0.5)
    X_noise_fft = np.fft.fft(x_noise)

    # --- спектр зашумленного сигнала
    plot_spectrum(X_noise_fft, fs, "Спектр зашумленного сигнала")

    # --- обратное преобразование зашумленного сигнала
    x_rec_noise = np.fft.ifft(X_noise_fft)
    plt.figure()
    plt.plot(t, x_noise, label="Зашумленный сигнал")
    plt.plot(t, x_rec_noise.real, '--', label="Восстановленный сигнал")
    plt.title("Обратное преобразование зашумленного сигнала")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================================================
# запуск
# =========================================================
if __name__ == "__main__":
    main()
