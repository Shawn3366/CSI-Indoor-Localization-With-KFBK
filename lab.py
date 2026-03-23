#coding=UTF-8
import os
import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF
import GPy
import argparse
import torch
#import utils 这个是有的
from torch.utils.data import DataLoader
import time
# from bls import BLS  # 从bls模块导入BLS函数
from sklearn.preprocessing import Normalizer, QuantileTransformer
from scipy.interpolate import interp1d
from sklearn.neighbors import KNeighborsRegressor
# from pykalman import KalmanFilter
# from scipy import signal
# from scipy.signal import medfilt
import random
from data_utils import *
from deep_bls import DeepBLS
# from BLS_Regression import BLS
from myEx.kan_network import KAN
import torch.nn as nn
# 在文件开头添加
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
print(f"使用设备: {DEVICE}")


SEED = 1
# 设置全局随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import tensorflow as tf
tf.random.set_seed(SEED)

stateFile = "state-200_NN200.mat"
rewardFile = "reward-200_NN200.mat"

import pywt
def compute_threshold(coeffs):
    """ 计算自适应阈值（VisuShrink 方法） """
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    return 0.8 * sigma * np.sqrt(2 * np.log(len(coeffs[-1])))


def wavelet_denoising(data, wavelet='db4', level=5, threshold_method='soft'):

    denoised_data = np.zeros_like(data)

    # 确保分解层数不过高
    max_level = pywt.dwt_max_level(data.shape[0], pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)

    for i in range(data.shape[1]):  # 遍历每个特征通道
        coeffs = pywt.wavedec(data[:, i], wavelet, level=level)  # 小波分解


        # 仅对细节系数进行阈值去噪，保留近似系数
        coeffs_new = [coeffs[0]] + [
            (compute_threshold(coeffs), np.zeros_like(c))[1]
            for c in coeffs[1:]
        ]

        reconstructed = pywt.waverec(coeffs_new, 'db4')

        # 处理数据长度不匹配问题
        if len(reconstructed) > data.shape[0]:
            reconstructed = reconstructed[:data.shape[0]]
        elif len(reconstructed) < data.shape[0]:
            reconstructed = np.pad(reconstructed, (0, data.shape[0] - len(reconstructed)), mode='edge')

        denoised_data[:, i] = reconstructed

    return denoised_data


def getOriginalCSI():
    xLabel = getXlabel()
    yLabel = getYlabel()
    count = 0
    originalCSI = np.zeros((317, 3 * 30 * 1500), dtype=np.float64)
    newName = []
    label = np.empty((0, 2), dtype=np.int64)

    for i in range(21):
        for j in range(23):
            filePath = "D:/system_default/desk_top/he/BLS_KAN_Location/BLS_KAN_Location/47SwapData/coordinate" + xLabel[i] + yLabel[
                j] + ".mat"
            name = xLabel[i] + yLabel[j]
            if (os.path.isfile(filePath)):
                c = loadmat(filePath)
                CSI = np.reshape(c['myData'], (1, 3 * 30 * 1500))
                originalCSI[count, :] = CSI
                newName.append(name)
                label = np.append(label, [[int(xLabel[i]), int(yLabel[j])]], axis=0)
                count += 1

    return originalCSI, label, count

def getXlabel():
    xLabel = []
    for i in range(21):
        str = '%d' % (i + 1)
        xLabel.append(str)
    return xLabel


def getYlabel():
    yLabel = []
    for j in range(23):
        if (j < 9):
            num = 0
            str = '%d%d' % (num, j + 1)
            yLabel.append(str)
        else:
            yLabel.append('%d' % (j + 1))
    return yLabel
from sklearn.decomposition import PCA
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import pywt

def generatePilot():
    originalCSI, label, count = getOriginalCSI()
    # originalData = np.array(originalCSI[:, 0:3 * 30 * 900:5400], dtype='float')  # 317*15
    originalData = np.array(originalCSI[:, 0:2 * 40 * 800:2000], dtype=np.float64) #317*32
    originalData = SimpleImputer(strategy='mean').fit_transform(originalData)


    rng = np.random.RandomState(20)
    randomLabel = rng.randint(1, 317, size=32)
    labelIndex = np.sort(randomLabel)
    listCSI = originalData[labelIndex, :]
    return label[labelIndex], listCSI


def findIndex(label, pathPlan):
    index = []
    for i in range(len(pathPlan)):
        index1 = np.where(label[:, 0] == pathPlan[i][0])
        index2 = np.where(label[:, 1] == pathPlan[i][1])
        similar = list(set(index1[0]).intersection(set(index2[0])))
        index.append(similar)
    index = [x for x in index if x]  # 删除空元素
    return index


def filterProcess(mulGauProPrediction, n_iter):
    from pykalman import KalmanFilter
    from scipy import signal
    bufferCSI = np.zeros((len(mulGauProPrediction), len(mulGauProPrediction[0])), dtype=np.float64)
    b, a = signal.butter(2, 3 * 2 / 50, 'lowpass')
    for i in range(len(mulGauProPrediction)):
        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
        measurements = mulGauProPrediction[i]
        kf = kf.em(measurements, n_iter=n_iter)
        (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
        swap = filtered_state_means[:, 0]
        finalResult = signal.filtfilt(b, a, swap)
        bufferCSI[i, :] = finalResult
    return bufferCSI


def isOdd(n):
    if int(n) % 2 == 1:
        return int(n)
    else:
        return int(n) - 1


def find_close_fast(arr, e):
    low = 0
    high = len(arr) - 1
    idx = -1
    rng = np.random.RandomState(20)
    randomInt = rng.randint(len(errorBand[0]))

    while low <= high:
        mid = int((low + high) / 2)
        if e[randomInt, 0] == arr[mid] or mid == low:
            idx = mid
            break
        elif e[randomInt, 0] > arr[mid]:
            low = mid
        elif e[randomInt, 0] < arr[mid]:
            high = mid
    if idx + 1 < len(arr) and abs(e[randomInt, 0] - arr[idx]) > abs(e[randomInt, 0] - arr[idx + 1]):
        idx += 1

    return arr[idx]


def tensorData(professionalData, device):
    lengths = torch.tensor(list(map(len, professionalData)))
    lengths = lengths.to(device)

    data = []
    for i in range(len(professionalData)):
        TensorResult = torch.tensor(abs(professionalData[i]), dtype=torch.int64)
        data.append(TensorResult)
    inputs = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    inputs = inputs.to(device)
    return inputs, lengths


def completionData(rec, pilotCSI):
    recArray = np.zeros((len(rec), len(rec[0] - 1)), dtype=np.int)
    for i in range(len(rec)):
        transfer = np.array(rec[i])
        recArray[i, :] = transfer
    recArray = np.column_stack((recArray, pilotCSI[:, -1]))
    return recArray

def findPossiblePath(stateFile):
    possiblePath = []
    stateLabel = []
    state = loadmat(r"D:/system_default/desk_top/he/BLS_KAN_Location/BLS_KAN_Location/Fifth code/" + stateFile)
    stateList = np.reshape(state['array'], (100, 200, 2))
    for i in range(100):
        a = np.array(stateList[i]).tolist()
        list.append(a, [1, 1])
        new_list = [list(t) for t in set(tuple(xx) for xx in a)]
        new_list.sort()
        if [1, 1] and [21, 23] in new_list:
            possiblePath.append(new_list)
            stateLabel.append(i)
    return possiblePath, stateLabel


def OptimalPath(rewardFile):
    possiblePath, stateLabel = findPossiblePath(stateFile)
    reward = loadmat(r"D:/system_default/desk_top/he/BLS_KAN_Location/BLS_KAN_Location/Fifth code/" + rewardFile)
    rewardList = reward['array'][0]
    numOfpath = len(stateLabel)
    valueOfReward = []
    for i in range(numOfpath):
        valueOfReward.append(rewardList[stateLabel[i]])
    max_index = np.argmax(np.array(valueOfReward))
    OptimalPath = possiblePath[int(max_index)]
    return OptimalPath, np.max(valueOfReward)


def accuracyPre(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions - labels) ** 2, 1))) * 50 / 100


def accuracyStd(predictions, testLabel):
    error = np.asarray(predictions - testLabel)
    sample = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2) * 50 / 100
    return np.std(sample)



def saveTestErrorMat(predictions, testLabel, fileName):
    error = np.asarray(predictions - testLabel)
    sample = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2) * 50 / 100
    savemat(fileName + '.mat', {'array': sample})



if __name__ == '__main__':
    #main()

    "生成飞行数据"
    originalCSI, label, count = getOriginalCSI()
    pilotLabel, pilotCSI = generatePilot()
        # print(pilotLabel)
    "多元高斯回归过程"
    mean = np.mean(pilotLabel, axis=1)
    covMatrix = np.cov(pilotCSI)
    kernelRBF = GPy.kern.RBF(input_dim=2, variance=1)
    omiga = kernelRBF.K(pilotLabel, pilotLabel)
    kernelPilot = covMatrix * omiga
    np.random.seed(0)
    mulGauProPrediction = np.random.multivariate_normal(mean, kernelPilot, size=len(label))

    "滤波平滑处理"
    bufferCSI = filterProcess(mulGauProPrediction, n_iter=2)

    "state space model修正飞行数据"
    import statsmodels.api as sm
    meanError = np.mean(pilotCSI, axis=0)  #列平均
    newModel = sm.tsa.SARIMAX(meanError, order=(1,0,0), trend='c')
    results = newModel.fit(disp=False)
    predict_sari = results.get_prediction()

    "误差变量即CSI变动范围，3 Antenna * 5 Packet"
    errorBand = predict_sari.conf_int()

    "由errorBand的最小值对备选序列进行约束处理"
    from scipy.signal import savgol_filter
    filterMatrix = bufferCSI
    for i in range(len(bufferCSI)):
        sliding_window = isOdd(find_close_fast(bufferCSI[i], errorBand))    #寻找每个序列与ErrorBand最接近的元素作为滑动窗口
        sliding_window = 3  # 确保滑动窗口长度大于多项式阶数
        tmp_result = savgol_filter(bufferCSI[i], sliding_window, 2)
        filterMatrix[i,:] = tmp_result
        # plt.plot(filterMatrix)
        # plt.show()

    "A3C路径规划"
    pathPlan, maxReward = OptimalPath(rewardFile)

    "20%的真实数据"
    index_A3CPredict = np.array(findIndex(label, pathPlan)).flatten()
    index = np.array(findIndex(label, pilotLabel)).flatten()
    index_GaussianAndA3C = np.sort(list(set(np.append(index_A3CPredict, index, axis=0))))
    secondpilotCSI = originalCSI[index_GaussianAndA3C, ]
    "将0和inf替换为NaN，均值填充NaN值"
    secondpilotCSI_cl = secondpilotCSI
    secondpilotCSI_cl_df = pd.DataFrame(secondpilotCSI_cl)
    secondpilotCSI_cl_df.replace(0, np.nan, inplace=True)
    secondpilotCSI_cl_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 计算每列的均值，忽略NaN值
    mean_values = secondpilotCSI_cl_df.mean()
    # 用均值填充NaN值
    secondpilotCSI_cl_df.fillna(mean_values, inplace=True)
    # 将处理后的数据转换回 ndarray
    secondpilotCSI_cl_processed = secondpilotCSI_cl_df.values
    corresponding_labels = label[index_GaussianAndA3C, ]
    secondpilotCSI_cl_processed32 = np.array(secondpilotCSI_cl_processed[:, 0:2 * 40 * 800:2000], dtype=np.float64)  # 317*32

    "Deep——bls"
    train_x, test_x, train_y, test_y = train_test_split( secondpilotCSI_cl_processed32, corresponding_labels, test_size=0.2, random_state=10)
    model = DeepBLS(max_iter=50,
                 learn_rate=0.01,
                 new_BLS_max_iter=10,
                 NumFea=20,
                 NumWin=5,
                 NumEnhan=50,
                 s=0.8,
                 C=2 ** -30)
    model.fit(train_y, train_x)
    final_predictions = model.predict(label)
    print("finnal:",final_predictions)

    "生成对抗网络（GAN）"
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    import os
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from Lab_GAN import *
    from Lab_GAN import build_generator, build_discriminator, build_gan, train_gan
    import tensorflow as tf
    import random

    np.random.seed(123)
    tf.random.set_seed(123)
    random.seed(123)

    # 确保 TensorFlow 使用确定性操作
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


    # np.random.seed(1)

    # 对生成的数据进行线性变换，使其更贴近原始真实值分布
    def adjusted_generate(raw_data, generated_data):
        original_mean = np.mean(raw_data)
        original_std = np.std(raw_data)
        generated_mean = np.mean(generated_data)
        generated_std = np.std(generated_data)
        adjusted_generated_data = (generated_data - generated_mean) / generated_std * original_std + original_mean
        return adjusted_generated_data



    gauss_inputs = secondpilotCSI_cl_processed
    gauss_label = corresponding_labels

    kernel = 1.0 * RBF(length_scale=5.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    stacked_arrays = np.concatenate((label, gauss_label), axis=0)
    # stacked_arrays = np.concatenate((predictions, gauss_label), axis=0)
    unique_rows, counts = np.unique(stacked_arrays, axis=0, return_counts=True)
    remainder_label = unique_rows[counts == 1]  # residual sampling points

    original_dim = gauss_inputs.shape[1]
    label_dim = gauss_label.shape[1]
    latent_dim = 1000

    data_dim = gauss_inputs.shape[1]
    generator = build_generator(latent_dim, label_dim, data_dim)
    discriminator = build_discriminator(data_dim, label_dim)
    gan = build_gan(generator, discriminator)
    train_gan(generator, discriminator, gan, gauss_inputs, gauss_label, epochs=50, batch_size=32)

    # 生成 remainder_label 对应的数据
    np.random.seed(1)
    noise = np.random.normal(0, 1, (len(remainder_label), latent_dim))
    generated_data = generator.predict([noise, remainder_label])
    adjusted_generated_data = adjusted_generate(gauss_inputs, generated_data)

    merged_labels = np.concatenate((gauss_label, remainder_label), axis=0)
    sorted_indices = np.lexsort((merged_labels[:, 1], merged_labels[:, 0]))
    merged_labels_sorted = merged_labels[sorted_indices]

    merged_data = np.concatenate((gauss_inputs, adjusted_generated_data), axis=0)
    sort_data = merged_data[sorted_indices]
    GAN_data = sort_data.reshape((317, 3 * 30 * 1500))
    # 在生成数据后添加
    print(f"原始标签数量: {len(label)}")
    print(f"真实标签数量: {len(gauss_label)}")
    print(f"GAN补充生成数量: {len(remainder_label)}")


    GAN_data32 = np.array(GAN_data[:, 0:2 * 40 * 800:2000], dtype=np.float64)
    thirdFinger = 0.4 * GAN_data32 + 0.6 * final_predictions
    thirdFinger = thirdFinger[:252, :]
    label = label[:252, :]
    print(thirdFinger.shape)

    # 小波分解
    thirdFinger_denoised = wavelet_denoising(thirdFinger)


    X_temp, X_test, y_temp, y_test = train_test_split(
        thirdFinger_denoised, label, test_size=0.2, random_state=SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=SEED
    )

    # 第一个KAN网络：无监督特征提取
    print("=== 训练特征提取KAN网络 ===")

    feature_extractor = KAN(
        layers_hidden=[32, 16, 32],
        grid_size=5,
        spline_order=3
    )


    def train_autoencoder(model, data, epochs=200):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        data_tensor = torch.FloatTensor(data).to(DEVICE)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            reconstructed = model.forward(data_tensor)
            loss = criterion(reconstructed, data_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f"Autoencoder Epoch {epoch}, Reconstruction Loss: {loss.item():.6f}")


    # 训练特征提取器 (仅使用训练集)
    train_autoencoder(feature_extractor, X_train)

    # 提取特征 (三份数据)
    with torch.no_grad():
        train_features = feature_extractor.extract_features(torch.FloatTensor(X_train).to(DEVICE)).cpu().numpy()
        val_features = feature_extractor.extract_features(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy()
        test_features = feature_extractor.extract_features(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()

    # 特征标准化 (以 train 为基准)
    scaler = QuantileTransformer(output_distribution='normal')
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    print(f"特征维度: {train_features.shape[1]}")

    # BLS定位模型
    bls_model = DeepBLS(
        max_iter=1,
        learn_rate=0.005,
        new_BLS_max_iter=20,
        NumFea=10,
        NumWin=10,
        NumEnhan=200,
        s=0.6,
        C=2 ** -25
    )
    bls_model.fit(train_features, y_train)

    # 第二个KAN网络：监督定位预测
    print("=== 训练定位预测KAN网络 ===")
    locator_kan = KAN(
        layers_hidden=[16, 6, 2],
        grid_size=3,
        spline_order=3,
        scale_base=1.3,
        scale_spline=1.2,
        grid_eps=0.05
    )

    locator_kan.fit(
        torch.FloatTensor(train_features).to(DEVICE),
        torch.FloatTensor(y_train).to(DEVICE),
        epochs=300,
        batch_size=16,
        lr=5e-4,
        grad_clip=5.0,
        update_grid_freq=10
    )

    with torch.no_grad():
        # 1. 获取验证集上的预测结果
        val_bls_p = bls_model.predict(val_features)
        val_kan_p = locator_kan.predict(torch.FloatTensor(val_features).to(DEVICE)).cpu().numpy()

        # 2. 计算验证集预测误差
        err_bls = np.sqrt(np.sum((val_bls_p - y_val) ** 2, axis=1)) + 1e-6
        err_kan = np.sqrt(np.sum((val_kan_p - y_val) ** 2, axis=1)) + 1e-6

        # 3. 基于误差计算权重目标并求平均
        tau = 0.6
        weights_kan = np.exp(-err_kan / tau) / (np.exp(-err_bls / tau) + np.exp(-err_kan / tau))

        # 最终的全局静态权重
        final_w_kan = np.mean(weights_kan)
        final_w_bls = 1.0 - final_w_kan


    # 最终应用
    bls_predictions = bls_model.predict(test_features)
    kan_predictions = locator_kan.predict(torch.FloatTensor(test_features).to(DEVICE)).cpu().numpy()

    # 执行加权融合
    final_predictions = (final_w_bls * bls_predictions) + (final_w_kan * kan_predictions)

    # 评估结果

    print(f"Mean errors: {accuracyPre(final_predictions, y_test):.3f} m")
    print(f"Std: {accuracyStd(final_predictions, y_test):.3f} m")

