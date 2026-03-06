# =========================================================
# Импорт библиотек
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# =========================================================
# Класс для разложения функции в ряд Фурье
# =========================================================

class FourierSeries:

    def __init__(self, func, T, N):
        self.func = func
        self.T = T
        self.N = N
        self.w0 = 2 * np.pi / T

    def a0(self):
        f = lambda t: self.func(t)
        return (2/self.T) * quad(f, -self.T/2, self.T/2)[0]

    def an(self, n):
        f = lambda t: self.func(t) * np.cos(n*self.w0*t)
        return (2/self.T) * quad(f, -self.T/2, self.T/2)[0]

    def bn(self, n):
        f = lambda t: self.func(t) * np.sin(n*self.w0*t)
        return (2/self.T) * quad(f, -self.T/2, self.T/2)[0]

    def approximate(self, t):

        result = self.a0()/2

        for n in range(1, self.N+1):
            result += self.an(n)*np.cos(n*self.w0*t)
            result += self.bn(n)*np.sin(n*self.w0*t)

        return result


# =========================================================
# Вспомогательные функции сигналов
# =========================================================

def rectangular_signal(t):
    return np.sign(np.sin(t))


def cosine_signal(A, f, t):
    w = 2 * np.pi * f
    return A * np.cos(w * t)


# =========================================================
# Раздел 1 — Аппроксимация прямоугольного сигнала
# =========================================================

def section1_fourier_approximation():

    T = 2 * np.pi
    N = 10

    t = np.linspace(-2*np.pi, 2*np.pi, 1000)

    fs = FourierSeries(rectangular_signal, T, N)

    x_real = np.array([rectangular_signal(i) for i in t])
    x_approx = np.array([fs.approximate(i) for i in t])

    error = x_real - x_approx

    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.plot(t, x_real, label="x(t)")
    plt.plot(t, x_approx, label="x*(t)")
    plt.title("Аппроксимация прямоугольного сигнала")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(t, error)
    plt.title("Погрешность e(t)")

    plt.tight_layout()
    plt.show()


# =========================================================
# Раздел 2 — Проверка алгоритма на косинусе
# =========================================================

def section2_cosine_test():

    A = 1
    f = 100

    T = 1
    fs = 5000

    t = np.linspace(0, T, fs)

    x = cosine_signal(A, f, t)

    # FFT
    X = np.fft.fft(x)
    freq = np.fft.fftfreq(len(x), d=t[1]-t[0])

    mask = freq >= 0

    freq = freq[mask]
    X = np.abs(X[mask])

    # коэффициент Фурье
    T_signal = 1 / f
    w0 = 2 * np.pi / T_signal

    a1 = (2/T_signal) * np.trapezoid(x * np.cos(w0*t), t)

    print("Коэффициент a1 =", a1)

    plt.figure()

    plt.plot(freq, X, label="FFT спектр")
    plt.axvline(f, linestyle="--", color="orange", label="Частота сигнала")

    plt.xlim(0,300)

    plt.title("Спектр сигнала cos")
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")

    plt.legend()

    plt.show()


# =========================================================
# Раздел 3 — Спектр прямоугольного сигнала
# =========================================================

def section3_rectangular_spectrum():

    f = 5
    t = np.linspace(0, 1, 5000)

    x = np.sign(np.sin(2*np.pi*f*t))

    X = np.fft.fft(x)
    freq = np.fft.fftfreq(len(x), d=t[1]-t[0])

    mask = freq >= 0

    plt.figure()

    plt.plot(freq[mask], np.abs(X[mask]))

    plt.title("Спектр прямоугольного сигнала")
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")

    plt.xlim(0,50)

    plt.show()


# =========================================================
# Раздел 4 — Добавление шума
# =========================================================

def section4_noise_effect():

    f = 5
    t = np.linspace(0, 1, 5000)

    x = np.sign(np.sin(2*np.pi*f*t))

    noise = 0.5 * np.random.randn(len(t))

    x_noise = x + noise

    X_noise = np.fft.fft(x_noise)
    freq = np.fft.fftfreq(len(x_noise), d=t[1]-t[0])

    mask = freq >= 0

    plt.figure()

    plt.plot(freq[mask], np.abs(X_noise[mask]))

    plt.title("Спектр сигнала с шумом")
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")

    plt.xlim(0,50)

    plt.show()


# =========================================================
# Главная функция
# =========================================================

def main():

    section1_fourier_approximation()
    section2_cosine_test()
    section3_rectangular_spectrum()
    section4_noise_effect()


if __name__ == "__main__":
    main()
