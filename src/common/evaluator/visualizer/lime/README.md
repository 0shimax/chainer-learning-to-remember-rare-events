# lime for image classification
このツールはarXiv:1602.04938(https://github.com/marcotcr/lime) の画像識別タスクを対象とした実装です。
##使い方
`import`して`LimeImage`オブジェクトを初期化します。

    from expalin_image import LimeImage
    lime = LimeImage(image, classifier, preprocessor)

* `image`オブジェクトはexplainしたい画像の`PIL.Image`オブジェクト
* `classifier`は引数として画像のベクター表現をサンプル数だけならべたものをとる関数です。返り値として各ラベルのスコアや確率値を返すものを想定します。返り値のフォーマットは`shape`が`(number_of_samples, number_of_labels)`である`numpy.array`であるとします
* `preprocessor`は引数として画像オブジェクトを受けとり`classifier`に渡すベクトル表現に変換する関数

100個のサンプルを生成して、`classifier`を使いスコアを計算します。

    lime.generate_samples(100)

与えられたラベル0に対して5superpixelを使った場合の各superpixelの重み生成します。

    lime.construct_explainer_with_label(0, 5)
    
与えられたラベル0に対して5superpixelを使った場合のサブ画像を生成します。

    import matplotlib.pyplot as plt
    plt.imshow(lime.constrcut_explainer_with_lable(0, 5))
    plt.show()



