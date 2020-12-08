import os
import pickle
import torch
from sklearn.metrics import classification_report
from model.Mymodel import PGAN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_dataset(task):
    print("task: ", task)

    A_us, A_uu = pickle.load(open("dataset/"+task+"/relations.pkl", 'rb'))
    X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, word_embeddings = pickle.load(open("dataset/"+task+"/train.pkl", 'rb'))
    X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev = pickle.load(open("dataset/"+task+"/dev.pkl", 'rb'))
    X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test = pickle.load(open("dataset/"+task+"/test.pkl", 'rb'))
    config['maxlen'] = len(X_train_source_wid[0])
    if task == 'twitter15':
        config['n_heads'] = 10
    elif task == 'twitter16':
        config['n_heads'] = 8
    else:
        config['n_heads'] = 7
        config['batch_size'] = 128
        config['num_classes'] = 2
        config['target_names'] = ['NR', 'FR']
    print(config)

    config['embedding_weights'] = word_embeddings
    config['A_us'] = A_us
    config['A_uu'] = A_uu
    return X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
           X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, \
           X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test


def train_and_test(model, task):
    model_suffix = model.__name__.lower().strip("text")
    config['save_path'] = 'checkpoint/weights.best.' + task + "." + model_suffix

    X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred, \
    X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, \
    X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid, y_test = load_dataset(task)

    nn = model(config)
    # nn.fit(X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred,
    #        X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev)  #

    print("================================")
    nn.load_state_dict(torch.load(config['save_path']))
    y_pred = nn.predict(X_test_source_wid, X_test_source_id, X_test_user_id, X_test_ruid)
    print(classification_report(y_test, y_pred, target_names=config['target_names'], digits=3))


config = {
    'lr':1e-3,
    'reg':1e-6,
    'embeding_size': 100,
    'batch_size':16,
    'nb_filters':100,
    'kernel_sizes':[3, 4, 5],
    'dropout':0.5,
    'epochs':18,
    'num_classes':4,
    'target_names':['NR', 'FR', 'TR', 'UR']
}


if __name__ == '__main__':
    task = 'twitter15'
    # task = 'twitter16'
    # task = 'weibo'
    model = PGAN
    train_and_test(model, task)



# Twitter15
#               precision    recall  f1-score   support
#
#           NR      0.865     0.988     0.922        84
#           FR      0.975     0.917     0.945        84
#           TR      0.938     0.893     0.915        84
#           UR      0.951     0.917     0.933        84
#
#     accuracy                          0.929       336
#    macro avg      0.932     0.929     0.929       336
# weighted avg      0.932     0.929     0.929       336



# Twitter16
#               precision    recall  f1-score   support
#
#           NR      0.936     0.957     0.946        46
#           FR      0.976     0.870     0.920        46
#           TR      0.857     0.933     0.894        45
#           UR      0.979     0.979     0.979        47
#
#     accuracy                          0.935       184
#    macro avg      0.937     0.935     0.935       184
# weighted avg      0.938     0.935     0.935       184


# Weibo  head=7
#               precision    recall  f1-score   support
#
#           NR      0.967     0.936     0.951       529
#           FR      0.937     0.967     0.952       521
#
#     accuracy                          0.951      1050
#    macro avg      0.952     0.952     0.951      1050
# weighted avg      0.952     0.951     0.951      1050

