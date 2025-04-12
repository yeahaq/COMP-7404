from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import json
from sklearn.metrics import accuracy_score, recall_score
import matplotlib
matplotlib.use('TkAgg')

# 数据预处理部分保持不变
# ...

# 保存性能指标到文件
def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f)

# 加载性能指标从文件
def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# 训练 NB-SVM 模型并保存结果
def train_nb_svm(X, y):
    results = []

    print("开始训练 NB-SVM 模型...")
    # 随机划分数据集为 S1 和 S2
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # 假设 S1 和 S2 的大小各为一半，可以根据需要调整
    split_point = n_samples // 2
    S1_indices = indices[:split_point]
    S2_indices = indices[split_point:]

    X_S1 = X[S1_indices]
    y_S1 = y[S1_indices]
    X_S2 = X[S2_indices]
    y_S2 = y[S2_indices]

    # 估计朴素贝叶斯特征变换器
    n_features = X.shape[1]
    nb_transformers = []

    for j in range(n_features):
        X_S1_j = X_S1[:, j].reshape(-1, 1)
        n1 = np.sum(y_S1 == 1)
        n0 = np.sum(y_S1 == 0)

        pi_hat = n1 / (n1 + n0)
        kde_class1 = KernelDensity(bandwidth='silverman', kernel='gaussian').fit(X_S1_j[y_S1 == 1])
        kde_class0 = KernelDensity(bandwidth='silverman', kernel='gaussian').fit(X_S1_j[y_S1 == 0])
        nb_transformers.append((pi_hat, kde_class1, kde_class0))

    # 数据变换
    X_S2_transformed = np.zeros_like(X_S2)
    for j in range(n_features):
        pi_hat, kde_class1, kde_class0 = nb_transformers[j]
        X_S2_j = X_S2[:, j].reshape(-1, 1)
        log_density_class1 = kde_class1.score_samples(X_S2_j)
        log_density_class0 = kde_class0.score_samples(X_S2_j)
        transformed_feature = np.log(pi_hat) + log_density_class1 - np.log(1 - pi_hat) - log_density_class0
        X_S2_transformed[:, j] = transformed_feature

    # 定义 SVM 模型和参数网格
    svm_model = SVC()
    param_grid = {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100, 1000, 10000],
        'gamma': ['scale']
    }

    # 使用网格搜索寻找最佳参数
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_S2_transformed, y_S2)

    # 输出最佳参数
    print("最佳参数: ", grid_search.best_params_)

    # 使用最佳参数训练模型
    best_svm_model = grid_search.best_estimator_

    for i in range(10):
        print(f"nb_svm第{i}次评估")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42+i)
        # print("y_test 中 0 的数量:", np.sum(y_test == 0))
        # print("y_test 中 1 的数量:", np.sum(y_test == 1))
        # 测试集评估
        X_test_transformed = np.zeros_like(X_test)
        for j in range(n_features):
            pi_hat, kde_class1, kde_class0 = nb_transformers[j]
            X_test_j = X_test[:, j].reshape(-1, 1)
            log_density_class1 = kde_class1.score_samples(X_test_j)
            log_density_class0 = kde_class0.score_samples(X_test_j)
            transformed_feature = np.log(pi_hat) + log_density_class1 - np.log(1 - pi_hat) - log_density_class0
            X_test_transformed[:, j] = transformed_feature

        y_pred = best_svm_model.predict(X_test_transformed)
        # 计算性能指标
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_test, y_pred)
        detection_rate = recall_score(y_test, y_pred)

        false_alarm_rate = fp / (fp + tn)

        print(f"第{i}次评估accuracy:{accuracy}")
        print(f"第{i}次评估detection_rate:{detection_rate}")
        print(f"第{i}次评估false_alarm_rate:{false_alarm_rate}")

        results.append({
            'accuracy': accuracy,
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate
        })

    # 保存结果
    save_results(results, 'nb_svm_results.json')

    # 输出平均性能指标
    accuracies = [result['accuracy'] for result in results]
    detection_rates = [result['detection_rate'] for result in results]
    false_alarm_rates = [result['false_alarm_rate'] for result in results]

    print("\nNB-SVM 模型训练完成!")
    print(f"  准确率: {np.mean(accuracies)}")
    print(f"  检测率: {np.mean(detection_rates)}")
    print(f"  误报率: {np.mean(false_alarm_rates)}")

    return results

# 训练 NB-SVM2 模型并保存结果
def train_nb_svm2(X, y):
    results = []

    print("开始训练 NB-SVM2 模型...")
    # 随机划分数据集为 S1 和 S2
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # 假设 S1 和 S2 的大小各为一半，可以根据需要调整
    split_point = n_samples // 2
    S1_indices = indices[:split_point]
    S2_indices = indices[split_point:]

    X_S1 = X[S1_indices]
    y_S1 = y[S1_indices]
    X_S2 = X[S2_indices]
    y_S2 = y[S2_indices]

    # 估计朴素贝叶斯特征变换器
    n_features = X.shape[1]
    nb_transformers = []

    for j in range(n_features):
        X_S1_j = X_S1[:, j].reshape(-1, 1)
        n1 = np.sum(y_S1 == 1)
        n0 = np.sum(y_S1 == 0)

        pi_hat = n1 / (n1 + n0)
        # bandwidth = 0.5
        kde_class1 = KernelDensity(bandwidth='silverman', kernel='gaussian').fit(X_S1_j[y_S1 == 1])
        kde_class0 = KernelDensity(bandwidth='silverman', kernel='gaussian').fit(X_S1_j[y_S1 == 0])
        nb_transformers.append((pi_hat, kde_class1, kde_class0))

    # 数据变换
    X_S2_transformed = np.zeros_like(X_S2)
    for j in range(n_features):
        pi_hat, kde_class1, kde_class0 = nb_transformers[j]
        X_S2_j = X_S2[:, j].reshape(-1, 1)
        log_density_class1 = kde_class1.score_samples(X_S2_j)
        log_density_class0 = kde_class0.score_samples(X_S2_j)
        transformed_feature = np.log(pi_hat) + log_density_class1 - np.log(1 - pi_hat) - log_density_class0
        X_S2_transformed[:, j] = transformed_feature

    # print(X_S2_transformed)
    # print(f"X_S2_transformed.shape:{X_S2_transformed.shape}")
    # print(X_S2)
    # print(f"X_S2.shape:{X_S2.shape}")
    X_S2_combined = np.hstack((X_S2_transformed, X_S2))
    # 定义 SVM 模型和参数网格
    svm_model = SVC()
    param_grid = {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100, 1000, 10000],
        'gamma': ['scale']
    }

    # 使用网格搜索寻找最佳参数
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_S2_combined, y_S2)

    # 输出最佳参数
    print("最佳参数: ", grid_search.best_params_)

    # 使用最佳参数训练模型
    best_svm_model = grid_search.best_estimator_

    for i in range(10):
        print(f"nb_svm2第{i}次评估")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42+i)
        # 测试集评估
        X_test_transformed = np.zeros_like(X_test)
        for j in range(n_features):
            pi_hat, kde_class1, kde_class0 = nb_transformers[j]
            X_test_j = X_test[:, j].reshape(-1, 1)
            log_density_class1 = kde_class1.score_samples(X_test_j)
            log_density_class0 = kde_class0.score_samples(X_test_j)
            transformed_feature = np.log(pi_hat) + log_density_class1 - np.log(1 - pi_hat) - log_density_class0
            X_test_transformed[:, j] = transformed_feature

        X_test_combined = np.hstack((X_test_transformed, X_test))
        y_pred = best_svm_model.predict(X_test_combined)
        # 计算性能指标
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_test, y_pred)
        detection_rate = recall_score(y_test, y_pred)

        false_alarm_rate = fp / (fp + tn)

        print(f"第{i}次评估accuracy:{accuracy}")
        print(f"第{i}次评估detection_rate:{detection_rate}")
        print(f"第{i}次评估false_alarm_rate:{false_alarm_rate}")

        results.append({
            'accuracy': accuracy,
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate
        })

    # 保存结果
    save_results(results, 'nb_svm2_results.json')

    # 输出平均性能指标
    accuracies = [result['accuracy'] for result in results]
    detection_rates = [result['detection_rate'] for result in results]
    false_alarm_rates = [result['false_alarm_rate'] for result in results]

    print("\nNB-SVM2 模型训练完成!")
    print(f"  准确率: {np.mean(accuracies)}")
    print(f"  检测率: {np.mean(detection_rates)}")
    print(f"  误报率: {np.mean(false_alarm_rates)}")

    return results

# 训练 Single-SVM 模型并保存结果
def train_single_svm(X, y):
    results = []

    print("开始训练 Single-SVM 模型...")
    # 数据分割 S1 和 S2
    X_S1, X_S2, y_S1, y_S2 = train_test_split(X, y, test_size=0.5, random_state=42)

    # 训练模型
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_model.fit(X_S1, y_S1)

    for i in range (10):
        print(f"single_svm第{i}次评估")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42+i)
        # 测试集评估
        y_pred = svm_model.predict(X_test)
        # 计算性能指标
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = accuracy_score(y_test, y_pred)
        detection_rate = recall_score(y_test, y_pred)

        false_alarm_rate = fp / (fp + tn)

        results.append({
            'accuracy': accuracy,
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate
        })

    # 保存结果
    save_results(results, 'single_svm_results.json')

    # 输出平均性能指标
    accuracies = [result['accuracy'] for result in results]
    detection_rates = [result['detection_rate'] for result in results]
    false_alarm_rates = [result['false_alarm_rate'] for result in results]

    print("\nSINGLE-SVM 模型训练完成!")
    print(f"  准确率: {np.mean(accuracies)}")
    print(f"  检测率: {np.mean(detection_rates)}")
    print(f"  误报率: {np.mean(false_alarm_rates)}")

    return results

# 绘制箱型图
def plot_boxplots():
    nb_svm_results = load_results('nb_svm_results.json')
    nb_svm2_results = load_results('nb_svm2_results.json')
    single_svm_results = load_results('single_svm_results.json')

    accuracies_nb = [result['accuracy'] for result in nb_svm_results]
    detection_rates_nb = [result['detection_rate'] for result in nb_svm_results]
    false_alarm_rates_nb = [result['false_alarm_rate'] for result in nb_svm_results]

    accuracies_nb2 = [result['accuracy'] for result in nb_svm2_results]
    detection_rates_nb2 = [result['detection_rate'] for result in nb_svm2_results]
    false_alarm_rates_nb2 = [result['false_alarm_rate'] for result in nb_svm2_results]

    accuracies_single = [result['accuracy'] for result in single_svm_results]
    detection_rates_single = [result['detection_rate'] for result in single_svm_results]
    false_alarm_rates_single = [result['false_alarm_rate'] for result in single_svm_results]

    def plot_boxplot(metrics_nb, metrics_nb2, metrics_single, metric_name):
        plt.figure(figsize=(8, 6))
        plt.boxplot([metrics_nb, metrics_nb2, metrics_single], labels=['NB-SVM', 'NB-SVM2', 'Single-SVM'])
        plt.title(f'Distribution of {metric_name} across 10 runs')
        plt.ylabel(metric_name)
        plt.show()

    plot_boxplot(accuracies_nb, accuracies_nb2, accuracies_single, 'Accuracy')
    plot_boxplot(detection_rates_nb, detection_rates_nb2, detection_rates_single, 'Detection Rate')
    plot_boxplot(false_alarm_rates_nb, false_alarm_rates_nb2, false_alarm_rates_single, 'False Alarm Rate')

# 主程序
if __name__ == "__main__":
    # 加载数据
    raw_data_filename = "./data/clearData/totall_extend.csv"
    print("Loading raw data...")
    raw_data = pd.read_csv(raw_data_filename, header=None, low_memory=False)

    raw_data = raw_data.sample(frac=0.003)

    # 查看标签数据情况
    last_column_index = raw_data.shape[1] - 1
    print("print data labels:")
    print(raw_data[last_column_index].value_counts())

    # 将非数值型的数据转换为数值型数据
    # 将 "BENIGN" 类别设为 0，其他类别设为 1
    raw_data[raw_data.shape[1] - 1] = raw_data[raw_data.shape[1] - 1].apply(lambda x: 0 if x == 'BENIGN' else 1)

    # 分离出两类数据
    benign_data = raw_data[raw_data[raw_data.shape[1] - 1] == 0]
    attack_data = raw_data[raw_data[raw_data.shape[1] - 1] == 1]

    # 计算两类数据的数量
    benign_count = len(benign_data)
    attack_count = len(attack_data)

    # 确定抽取的数量，取两类中较小的数量
    sample_count = min(benign_count, attack_count)

    # 随机抽取指定数量的样本
    benign_sample = benign_data.sample(n=sample_count, random_state=42)
    attack_sample = attack_data.sample(n=sample_count, random_state=42)

    # 合并两类样本
    sampled_data = pd.concat([benign_sample, attack_sample])

    # 随机打乱合并后的数据
    sampled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # 查看标签数据情况
    last_column_index = sampled_data.shape[1] - 1
    print("print data labels:")
    print(sampled_data[last_column_index].value_counts())

    # 对原始数据进行切片，分离出特征和标签
    features = sampled_data.iloc[:, :sampled_data.shape[1] - 1]
    labels = sampled_data.iloc[:, sampled_data.shape[1] - 1]

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # 将多维的标签转为一维的数组
    labels = labels.values.ravel()
    # print(labels)
    # print(labels.shape)

    # 将数据分为训练集和测试集，并打印维数
    df = pd.DataFrame(features)
    df = df.values
    print(df)
    print(df.shape)

    # 分别训练三种模型
    train_nb_svm(df, labels)
    train_nb_svm2(df, labels)
    train_single_svm(df, labels)

    # 绘制箱型图
    plot_boxplots()