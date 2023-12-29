# -*- coding: utf-8 -*-

#
# GMM-HMMを学習します．
#

# hmmfunc.pyからMonoPhoneHMMクラスをインポート
from hmmfunc import MonoPhoneHMM

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# os, sysモジュールをインポート
import sys
import os
from multiprocessing import Pool, Lock


##def train_subset(args):
##    start_idx, end_idx, feat_list, label_list = args
##   subset_feat_list = {utt: feat_list[utt] for utt in list(feat_list.keys())[start_idx:end_idx]}
##    subset_label_list = {utt: label_list[utt] for utt in list(label_list.keys())[start_idx:end_idx]}
##
##    # 学習(1iteration分)
##    hmm.train(subset_feat_list, subset_label_list)

def train_parallel(args):
    """Function to be run in parallel"""
    hmm, feat_list, label_list, report_interval , batch_size , trans_lock, pdf_lock = args
    hmm.train(feat_list, label_list, report_interval=report_interval)


#
# メイン関数
#
if __name__ == "__main__":
    # Pool of workers, specify the number of CPUs you want to use
    num_cpus = 64

    report_interval = 10
    pool = Pool(processes=num_cpus)
    trans_lock = Lock()
    pdf_lock = Lock()

    # 学習元のHMMファイル
    base_hmm = './exp/model_3state_1mix/0.hmm'

    # 訓練データの特徴量リストのファイル
    feat_scp = \
        '../01compute_features/mfcc/train_large/feats.scp'
        #'../01compute_features/mfcc/train_small/feats.scp'

    # 訓練データのラベルファイル
    label_file = \
        './exp/data/train_large/text_int'
        #'./exp/data/train_small/text_int'

    # 学習結果を保存していくフォルダ
    work_dir = './exp'

    # 更新回数
    num_iter = 100 #10

    # 混合数を増やす回数
    # 増やすたびに混合数は2倍になる
    # 最終的な混合数は2^(mixup_time)となる
    # 更新回数はnum_iter*(mixup_time+1)
    mixup_time = 5 #1

    # 学習に用いる発話数
    # 実際は全ての発話を用いるが，時間がかかるため
    # このプログラムでは一部の発話のみを使っている
    num_utters = 4500 #50

    # Assuming shared_trans and shared_pdf are meant to store float value. 

    #
    # 処理ここから
    #

    # MonoPhoneHMMクラスを呼び出す
    hmm = MonoPhoneHMM()

    # 学習前のHMMを読み込む
    hmm.load_hmm(base_hmm)

    # 学習元HMMファイルの状態数を得る
    num_states = hmm.num_states

    # 学習元HMMファイルの混合数を得る
    num_mixture = hmm.num_mixture

    # ラベルファイルを開き，発話ID毎の
    # ラベル情報を得る
    label_list = {}
    with open(label_file, mode='r') as f:
        for line in f:
            # 0列目は発話ID
            utt = line.split()[0]
            # 1列目以降はラベル
            lab = line.split()[1:]
            # 各要素は文字として読み込まれているので，
            # 整数値に変換する
            lab = np.int64(lab)
            # label_listに登録
            label_list[utt] = lab

    # 特徴量リストファイルを開き，
    # 発話ID毎の特徴量ファイルのパスを得る
    feat_list = {}
    with open(feat_scp, mode='r') as f:
        # 特徴量のパスをfeat_listに追加していく
        # このとき，学習に用いる発話数分だけ追加する
        # (全てのデータを学習に用いると時間がかかるため)
        for n, line in enumerate(f):
            if n >= num_utters:
                break
            # 0列目は発話ID
            utt = line.split()[0]
            # 1列目はファイルパス
            ff = line.split()[1]
            # 3列目は次元数
            nd = int(line.split()[3])
            # 発話IDがlabel_に存在しなければエラー
            if not utt in label_list:
                sys.stderr.write(\
                    '%s does not have label\n' % (utt))
                exit(1)
            # 次元数がHMMの次元数と一致しなければエラー
            if hmm.num_dims != nd:
                sys.stderr.write(\
                    '%s: unexpected #dims (%d)\n'\
                    % (utt, nd))
                exit(1)
            # feat_fileに登録
            feat_list[utt] = ff
    #
    # 学習処理
    #

    # 出力ディレクトリ名
    model_name = 'model_%dstate_%dmix' \
        % (num_states, num_mixture)
    out_dir = os.path.join(work_dir, model_name)

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_dir, exist_ok=True)

    # Split the data for parallel processing
    ##data_splits = []
    ##for i in range(num_cpus):
    ##    start_idx = i * (num_utters // num_cpus)
    ##    end_idx = (i+1) * (num_utters // num_cpus) if i < num_cpus - 1 else num_utters 
    ##    data_splits.append((start_idx, end_idx, feat_list, label_list))

    # Split utterances into batches
    utterances_per_batch = int(num_utters / num_cpus)
    batches = []
    start_idx = 0
    for _ in range(num_cpus):
        end_idx = start_idx + utterances_per_batch
        batch_feat_list = {k: v for k, v in list(feat_list.items())[start_idx:end_idx]}
        batch_label_list = {k: v for k,v in list(label_list.items())[start_idx:end_idx]}
        batches.append((hmm, batch_feat_list, batch_label_list, report_interval, utterances_per_batch, trans_lock, pdf_lock ))
        start_idx = end_idx
    print(batches)



    # 混合数の増加回数の分だけループ
    for m in range(mixup_time+1):
        #
        # 混合数増加の処理を行い，
        # 新しいフォルダに0.hmmという名前で保存する
        #
        if m > 0:
            # 混合数増加処理を行う
            hmm.mixup()
            # 混合数を2倍にする
            num_mixture *= 2

            # 出力ディレクトリ名
            model_name = 'model_%dstate_%dmix' \
                % (num_states, num_mixture)
            out_dir = os.path.join(work_dir, model_name)

            # 出力ディレクトリが存在しない場合は作成する
            os.makedirs(out_dir, exist_ok=True)
          
            # HMMを保存
            out_hmm = os.path.join(out_dir, '0.hmm')
            hmm.save_hmm(out_hmm)
            print('Increased mixtures %d -> %d' \
                % (num_mixture/2, num_mixture))
            print('saved model: %s' % (out_hmm))

        #
        # 現在のGMM混合数で規定回数学習を行う
        #
        # num_iterの数だけ更新を繰り返す
        for iter in range(num_iter):
            print('%d-th iterateion' % (iter+1))


            # Train on subsets in parallel
            #pool.map(train_subset, data_splits)
            pool.map(train_parallel, batches)

            # HMMのプロトタイプをjson形式で保存
            out_hmm = os.path.join(out_dir, 
                                   '%d.hmm' % (iter+1))
            # 学習したHMMを保存
            hmm.save_hmm(out_hmm)
            print('saved model: %s' % (out_hmm))

    pool.close()
    pool.join()



