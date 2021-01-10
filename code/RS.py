
# 留一法每个user最新的item作为test的正样本 为每个用户的一个test正样本 挑选999个负样本
# 计算hr和ndcg
def _eval_by_user(user):

    if _model.train_auc:
        # get predictions of positive samples in training set
        train_item_input = _dataset.trainList[user] # user在train中交互过的item集合
        train_user_input = np.full(len(train_item_input), user, dtype='int32')[:, None]
        train_item_input = np.array(train_item_input)[:, None]
        feed_dict = {_model.user_input: train_user_input, _model.item_input_pos: train_item_input}

        train_predict = _sess.run(_model.output, feed_dict)

    # get prredictions of data in testing set
    # user_input是1000个此user id  # item_input是999负样本+1待预测的正样本
    user_input, item_input = _feed_dicts[user] 
    feed_dict = {_model.user_input: user_input, _model.item_input_pos: item_input}

    # 输出这1000个样本的预测分
    predictions = _sess.run(_model.output, feed_dict)

    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum() # 大于等于正样本得分的数量 即正确数量排名-1

    # calculate AUC for training set
    train_auc = 0
    if _model.train_auc:
        # 每个train_predict的单项是一个正样本 训练集的所有正样本
        for train_res in train_predict:
            # 训练集中的1个正样本 得分高于999个负样本的概率
            train_auc += (train_res > neg_predict).sum() / len(neg_predict)
        # train中每个正样本都有1个概率 按照正样本个数来平均
        train_auc /= len(train_predict)

    # calculate HR@K, NDCG@K, AUC
    # 命中时，正样本排名=pos+1<=K 等价于 pos<=K-1 等价于pos<K
    hr = position < _K # 如果命中为True
    if hr:
        # 理想 idcg = 1/log_2^2
        # 实际 dcg = 1/log_2^(pos+2)
        # ndcg = dcg/idcg = log_2^2/log_2^(pos+2) = ln2/ln(pos+2)
        ndcg = math.log(2) / math.log(position+2)
    else:
        ndcg = 0
    # 每个负样本和此正样本比 此正样本得分高的概率 auc = 1-比正样本高的数量/负样本数量
    auc = 1 - (position * 1. / len(neg_predict))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]
    return hr, ndcg, auc, train_auc
