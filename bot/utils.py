import numpy as np


def get_user_hist(df, user_id, h3, topk=20):
    old_items = df[(df.user_id == user_id) & (df.h3 == h3)].old_items
    assert len(old_items) == 3 # for each model
    old_items = np.array(old_items.head(1))[0][:topk]
    msg = f'Выбран user = {user_id}, h3 = {h3}\n'
    return msg + '\nИстория заказов: '+ ', '.join([s for s in old_items])


def get_user_pred(df, user_id, h3, model, topk=10):
    preds = df[(df.user_id == user_id) & (df.h3 == h3) & (df.model == model)].prediction
    assert len(preds) == 1
    preds = np.array(preds.head(1))[0][:topk]
    return '\nРекомендации:\n' + '\n'.join([f'    {i+1}. {s}' for i, s in enumerate(preds)])


def get_user_id(df, user_ids):
    return np.random.choice(user_ids)


def get_h3(df, user_id):
    h3s = np.array(df[df.user_id == user_id].h3.unique())
    return np.random.choice(h3s)


def next_top_k(top_k):
    if top_k == 5:
        return 10
    elif top_k == 10:
        return 30
    else:
        return 5