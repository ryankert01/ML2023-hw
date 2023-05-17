from sklearn import feature_selection as fs


def select_feat(config, train_data, valid_data, test_data):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,
                                                      :-1], valid_data[:, :-1], test_data

    feat_idx = []
    if config['select_all']:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        x_train_new = fs.SelectKBest(
            fs.f_regression, k=config['num_of_features']).fit_transform(raw_x_train, y_train)
        i = 0
        for it in x_train_new[1]:
            while raw_x_train[1][i] != it:
                i += 1
            feat_idx.append(i)
            i += 1
        print(feat_idx)
    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid
