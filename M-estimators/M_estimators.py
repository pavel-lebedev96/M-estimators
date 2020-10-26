import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#линейная регрессионная модель
def f(x):
    return np.array([[1],
                     [x[0]],
                     [x[1]]])

#вычисление истинных факторов
def calc_xx(n, a, b):
    return np.random.uniform(a, b, size = (n,k))

#вычисление истинного отклика с истинными факторами
def calc_yy(xx, theta):
    X = calc_mtrx_X(xx)
    return X @ theta

#вычисление обычной ошибки
def calc_errors(var, e_size):
    return np.random.normal(0, np.sqrt(var), size = e_size)

#вычисление ошибок c mu долей выбросов
def calc_errors_with_outliers(var, var1, mu, e_size):
    bi = np.random.binomial(1, 1.0 - mu, n)
    err_res = calc_errors(var, e_size)
    err_outliers = calc_errors(var1, e_size)
    for i in range(n):
        if bi[i] == 0:
            #выброс
            err_res[i] = err_outliers[i]
    return err_res

#вычисление матрицы X
def calc_mtrx_X(x):
    n = len(x)
    X = np.empty((n,m))
    for i in range(n):
        X[i] = f(x[i]).T
    return X

#метод наименьших квадратов
def calc_least_squares(x, y):
    X = calc_mtrx_X(x)
    return np.linalg.inv(X.T @ X) @ X.T @ y

#метод ортогональной регрессии
def calc_OR(x, y):
    p = m - 1
    #выборочная ковариационная матрица
    xy = np.empty((n, p + 1))
    for j in range(p):
        xy[:, j] = (x[:, j] - np.mean(x[:, j])) / (n - 1)
    xy[:, p] = (y[:, 0] - np.mean(y)) / (n - 1)
    #сингулярное разложение матрицы ковариац. матрицы
    U, d, Vh = np.linalg.svd(xy, False)
    V = Vh.conj().T
    Vxy = V[:p, p]
    Vyy = V[p, p]
    theta_est = (-1.0 * Vxy) * (1.0 / Vyy)
    theta_est0 = np.mean(y) - np.sum([
        theta_est[j] * np.mean(x[:, j]) for j in range(p)])
    return np.insert(theta_est, 0, theta_est0).reshape((m, 1))

#метод наименьших квадратов синусов
def calc_LSS(x, y):
    p = m - 1
    #расстояние R от i-го наблюдения до центра масс
    temp = np.zeros(n)
    for j in range(p):
        temp += np.square(x[:, j] - np.mean(x[:, j]))
    R = np.sqrt(temp + np.square(y[:, 0] - np.mean(y)))
    #выборочная ковариационная матрица
    xy = np.empty((n, p + 1))
    for j in range(p):
        xy[:, j] = (x[:, j] - np.mean(x[:, j])) / R
    xy[:, p] = (y[:, 0] - np.mean(y)) / R
    #сингулярное разложение матрицы ковариац. матрицы
    U, d, Vh = np.linalg.svd(xy, False)
    V = Vh.conj().T
    Vxy = V[:p, p]
    Vyy = V[p, p]
    theta_est = (-1.0 * Vxy) * (1.0 / Vyy)
    theta_est0 = np.mean(y) - np.sum([
        theta_est[j] * np.mean(x[:, j]) for j in range(p)])
    return np.insert(theta_est, 0, theta_est0).reshape((m, 1))

#расчет остатков
def calc_r(mtrx_X, y, theta):
    y_est = mtrx_X @ theta
    return np.abs(y - y_est)

#MAD -оценка
def calc_MAD(r):
    return np.median(r) / 0.67449

#взвешенный МНК
def calc_weighted_least_squares(X, W, y):
    return np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

#вычисление весов Хьбера
def calc_Huber_weights(r, lamda, c):
    w = np.empty(n)
    for i in range(n):
        u = r[i, 0] / lamda
        if np.abs(u) < c:
            w[i] = 1
        if np.abs(u) >= c:
            w[i] = c * np.sign(u) / u
    return np.diag(w)

#M - оценка
def calc_Huber(x, y, c = 8):         
    X = calc_mtrx_X(x)
    theta0 = calc_least_squares(x, y)
    r_0 = calc_r(X, y, theta0)
    lamda = calc_MAD(r_0)
    s = 0
    tau = 1
    while (tau > 1.e-5) and (s < 1.e4):
        W = calc_Huber_weights(r_0, lamda, c)
        theta = calc_weighted_least_squares(X, W, y)
        tau = np.max(np.abs((theta - theta0) / theta0))        
        r_0 = calc_r(X, y, theta0)
        lamda = calc_MAD(r_0)
        theta0 = np.copy(theta)
        s += 1
    return theta0

#нахождение истинных факторов в i - наблюдении
def find_xx_i(x, y, theta, sigma_eps, sigma_delt, i):
    A = np.empty((k, k))
    b = np.empty(k)
    for p in range(k):
        for r in range(k):
            A[p, r] = theta[r + 1] * theta[p + 1] / np.square(sigma_eps)
            if r == p:
                A[p, r] += 1 / np.square(sigma_delt)
        b[p] = x[i, p] / np.square(sigma_delt) + (y[i] - theta[0]) * theta[p + 1] / np.square(sigma_eps)
    return np.linalg.solve(A, b)

#нахождение истинных факторов
def find_xx(x, y, theta, sigma_eps, sigma_delt):
    xx = np.empty((n, k))
    for i in range(n):
        xx[i] = find_xx_i(x, y[:, 0], theta[:, 0], sigma_eps, sigma_delt, i)
    return xx

#проверка, принадлежит ли области возможных значений
def check_in_region(xx, x, sigma_delt):
    for i in range(n):
        for j in range(k):
            if np.abs(x[i, j] - xx[i, j]) > (3 * sigma_delt):
                if (x[i, j] > xx[i, j]):
                    xx[i, j] = x[i, j] - (3 * sigma_delt)
                else:
                    xx[i, j] = x[i, j] + (3 * sigma_delt)

#модифицированная М - оценка
def calc_Huber_M(x, y, sigma_eps = 0.1, sigma_delt = 0.5):
    theta0 = calc_Huber(x, y)
    s = 0
    tau = 1   
    while (tau > 1.e-6) and (s < 1.e4):
        #нахождение истинных факторов
        xx = find_xx(x, y, theta0, sigma_eps, sigma_delt)
        #проверка на принадлежность области
        check_in_region(xx, x, sigma_delt)
        theta = calc_Huber(xx, y)
        tau = np.max(np.abs((theta - theta0) / theta0))
        theta0 = np.copy(theta)
        s += 1
    return theta0

#L2 - норма
def calc_L2_norm(theta, theta_est):
    return np.sum(np.abs((theta - theta_est) / theta))

#вычислительный эксперимент
def test(num, xx, yy, y_outliers, x_outliers, var, var1, mu):
    #инициализация
    delta = np.empty((n, k))
    eps = np.empty((n, 1))
    methods = [calc_least_squares, calc_OR, calc_LSS, calc_Huber, calc_Huber_M]
    method_names = ['LS','OR', 'LSS', 'Huber', 'HuberM']
    #таблица: столбцы - Оценки и норма, строки - методы
    res = pd.DataFrame([], columns = ['Estimators', 'Norm'])
    for name in method_names:
        res.loc[name] = [np.zeros((m,1)), 0]

    #эксперимент
    for i in range(num):
        #моделирование наблюдаемых данных
        if y_outliers:
            eps = calc_errors_with_outliers(var, var1, mu, (n, 1))
        else:
            eps = calc_errors(var, (n, 1))  
    
        if x_outliers:
            delta = calc_errors_with_outliers(var, var1, mu, (n, k))
        else:
            delta = calc_errors(var, (n, k))
        y_res = yy + eps
        x_res = xx + delta
        #оценка и вычисление нормы
        for name, func in zip(method_names, methods):
            theta_est = func(x_res, y_res)
            norm = calc_L2_norm(theta, theta_est)
            res['Estimators'][name] += theta_est
            res['Norm'][name] += norm
    return res / num

#рисование графиков отклонения регрессии при выбросах
def print_plots(x, y, out_indexes):
    if k > 1: return
    
    #с учетом выбросов
    theta_MNK = calc_least_squares(x, y)
    y_est = calc_yy(x, theta_MNK)
    plt.plot(x, y_est, 'g-', 
                label = "Линия регрессия МНК")

    #без учета выбросов
    reg_indexes = np.fromiter(set(np.arange(n)) - set(out_indexes), 
                              int)
    theta_MNK = calc_least_squares(x[reg_indexes], y[reg_indexes])
    y_est = y_est = calc_yy(x, theta_MNK)
    plt.plot(x, y_est, 'k-', 
                label = "Истинная линия регрессии")

    plt.plot(x[reg_indexes] , y[reg_indexes], 'bo', 
                label = "Регулярные наблюдения")
    plt.plot(x[out_indexes] , y[out_indexes], 'ro',
                label = "Выбросы")

    plt.ylabel('y')
    plt.xlabel('x')
    plt.grid()
    plt.legend(loc = 'best')
    plt.show()

#исходные данные
#-----------------------------#

#n - число наблюдений
n = 500

#m - число параметров
m = 3

#k - число факторов
k = 2

#вектор истинных параметров
theta = np.array([[1],
                  [1],
                  [1]])

#факторы
xx = calc_xx(n, 0, 1)
#отклик
yy = calc_yy(xx, theta)
#-----------------------------#
#Параметры эксперимента
#-----------------------------#

y_outliers = False
x_outliers = True

#Дисперсии регулярных наблюдений и выбросов
var = 0.01
var1 = 1
mu = 0.1

#-----------------------------#

#эксперимент
num = 1
res = test(num, xx, yy, y_outliers, x_outliers, var, var1, mu)

#вывод результатов
np.set_printoptions(precision = 3)
for name in res.index:
    print(name)    
    for col in res.columns:
        print(col)
        if col == "Estimators":
            print(res[col][name]) 
        if col == "Norm":
           print("%.3e" % (res[col][name]))
    print("-------------\n")
