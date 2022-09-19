import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# digits = datasets.load_digits()

# print(digits.data.shape)
# # (1797, 64)

# print(digits.target.shape)
# # (1797,)

# X_reduced = TSNE(n_components=2, random_state=0).fit_transform(digits.data)

# print(X_reduced.shape)
# # (1797, 2)

# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=digits.target)
# plt.colorbar()
# plt.show()  # seabornでもmatplotlibでも表示されない．なぜか．(windowsだから？)
# # 学校のmac miniだとplt.show()を入れることで表示
# # <matplotlib.colorbar.Colorbar at 0x7ff21173ee90>


def t_sne(data, labels):
    """
    T-SNE法で多次元データを2次元に削減する
    data: 次元削減したいデータ
    labels: 色分けに使用される数字ラベル, 1次元配列
    """
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(data)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels)
    plt.colorbar()
    plt.show()

    return
