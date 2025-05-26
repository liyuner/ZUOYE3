# 环境准备
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, auc, accuracy_score)
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
# Iris数据集
from sklearn.datasets import load_iris
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.3)

# Breast Cancer数据集
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer, y_cancer, test_size=0.3)
scaler = StandardScaler()
X_cancer_train_scaled = scaler.fit_transform(X_cancer_train)
X_cancer_test_scaled = scaler.transform(X_cancer_test)

# 2. 分类与评估
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    time_cost = time.time() - start_time
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1": f1_score(y_test, y_pred, average="macro"),
        "Time (s)": round(time_cost, 4)
    }
    
    # 仅当能计算概率时添加ROC-AUC
    if y_pred_prob is not None:
        if len(np.unique(y_test)) == 2:  # 二分类
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_pred_prob[:, 1])
        else:  # 多分类
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_pred_prob, multi_class="ovo")
    else:
        metrics["ROC-AUC"] = "N/A"
    return metrics

# 绘制ROC曲线
def plot_roc_curve(model, X_test, y_test, model_name, dataset_name):
    if len(np.unique(y_test)) == 2:  # 二分类问题
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name} on {dataset_name}')
        plt.legend(loc="lower right")
        plt.show()
    else:  # 多分类问题
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_test_bin.shape[1]
        y_pred_prob = model.predict_proba(X_test)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 绘制所有类别的ROC曲线
        plt.figure()
        colors = ['blue', 'red', 'green']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name} on {dataset_name} (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.show()

# 3. 训练与结果存储
results = {}

# 定义模型配置（区分需要probability参数的模型）
model_configs = [
    ("Logistic Regression", LogisticRegression, {"max_iter": 1000, "random_state": 42}),
    ("Random Forest", RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
    ("SVM", SVC, {"max_iter": 1000, "probability": True, "random_state": 42})
]

for name, model_class, params in model_configs:
    # 初始化当前模型的存储字典
    results[name] = {}
    
    # Iris数据集
    model = model_class(**params)
    model.fit(X_iris_train, y_iris_train)
    results[name]["Iris"] = evaluate_model(model, X_iris_train, X_iris_test, y_iris_train, y_iris_test)
    plot_roc_curve(model, X_iris_test, y_iris_test, name, "Iris")
    
    # Breast Cancer数据集
    model = model_class(**params)
    model.fit(X_cancer_train_scaled, y_cancer_train)
    results[name]["Breast Cancer"] = evaluate_model(
        model, X_cancer_train_scaled, X_cancer_test_scaled, 
        y_cancer_train, y_cancer_test
    )
    plot_roc_curve(model, X_cancer_test_scaled, y_cancer_test, name, "Breast Cancer")


for dataset in ['Iris', 'Breast Cancer']:
    print(f"\n▶ {dataset}数据集性能对比")
    print("-"*50)
    
    # 收集当前数据集的所有模型结果
    dataset_results = {}
    for model_name in results:
        if dataset in results[model_name]:
            dataset_results[model_name] = results[model_name][dataset]
    
    if dataset_results:
        df = pd.DataFrame.from_dict(dataset_results, orient='index')
        print(df.round(4))
    else:
        print("无有效结果数据")
    
    print("-"*50)