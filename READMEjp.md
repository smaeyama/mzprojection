  mzprojection
=================

射影演算子法による統計データ解析


### Overview ###

森-Zwanzigの射影演算子法は解析したい変数の時系列データのアンサンブル <img src="https://latex.codecogs.com/gif.latex?f(t)^i" alt="f(t)^i" /> を
興味変数の時系列データ <img src="https://latex.codecogs.com/gif.latex?u(t)^i" alt="u(t)^i" /> に対する相関・無相関部分に分離する。


### Contents ###

    fortran/ - Fortran 版ソースコード。サブルーチンの使い方や理論の詳細は README.txt 参照  
    python/ - Python 版ソースコード。関数の使い方や理論の詳細は README.txt 参照  
    sample_data/ - サンプル時系列データとその射影結果  
    QUICK_START.txt - テストプログラムを実行するための簡易説明  

### Reference ###

[Shinya Maeyama and Tomo-Hiko Watanabe, "Extracting and Modeling the Effects of Small-Scale Fluctuations on Large-Scale Fluctuations by Mori-Zwanzig Projection operator method", J. Phys. Soc. Jpn. 89, 024401 (2020).](https://doi.org/10.7566/JPSJ.89.024401)

