import numpy as np
import os
from wordcloud import WordCloud
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from itertools import cycle


def generate_word_cloud(df, col):
    content = " ".join(str(cat) for cat in df[col])
    right_2_left = lambda w: get_display(reshape(f'{w}'))
    content_count = Counter(content.split())
    counts = {right_2_left(k): v for k, v in content_count.most_common(20)}
    font_file = "..\\Visualization\\Font\\NotoNaskhArabic-Regular.ttf"
    word_cloud = WordCloud(font_path=font_file,
                           background_color="white").generate_from_frequencies(counts)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def get_classes_freq(df, col):
    plt.figure(figsize=(12, 6))
    sns.countplot(df[col], palette='Set3')
    plt.show()


def get_classes_percentage(df, col):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    explode = list((np.array(list(df[col].dropna().value_counts())) / sum(list(df[col].dropna().value_counts()))))[::-1]
    labels = list(df[col].dropna().unique())
    sizes = df[col].value_counts()[::-1]
    ax2.pie(sizes, explode=explode, startangle=60, labels=labels, autopct='%1.0f%%', pctdistance=0.4)
    sns.countplot(y=col, data=df, ax=ax1)
    ax1.set_title("Count of each class")
    ax2.set_title("Percentage of each class")
    plt.show()


def most_freq_word(df, col):
    content = " ".join(cat for cat in df[col])
    content_count = Counter(content.split())
    right_2_left = lambda w: get_display(reshape(f'{w}'))
    y = [count for tag, count in content_count.most_common(20)]
    x = [right_2_left(tag) for tag, count in content_count.most_common(20)]
    plt.bar(x, y, color='crimson')
    plt.title("Term frequencies in Data")
    plt.ylabel("Frequency (log scale)")
    plt.yscale('log')  # optionally set a log scale for the y-axis
    plt.xticks(rotation=90)
    for i, (tag, count) in enumerate(content_count.most_common(20)):
        plt.text(i, count, f' {count} ', rotation=90,
                 ha='center', va='top' if i < 10 else 'bottom', color='white' if i < 10 else 'black')
    plt.xlim(-0.6, len(x) - 0.4)  # optionally set tighter x lims
    plt.tight_layout()  # change the whitespace such that all labels fit nicely
    plt.show()


def ROC_plot(y_true_ohe, y_hat_ohe, label_encoder, n_classes):
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_ohe[:, i], y_hat_ohe[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_ohe.ravel(), y_hat_ohe.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(20, 20))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(label_encoder.classes_[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("multiclass characteristic")
    plt.legend(loc="lower right")
    plt.show()