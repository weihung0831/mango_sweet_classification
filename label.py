import pretty_errors
import pandas as pd


def weight_labels(data):
    weightx = data['weight'].values.copy()

    for i in range(len(weightx)):
        if weightx[i] > 0.380:
            weightx[i] = 0
        elif weightx[i] > 0.330:
            weightx[i] = 1
        else:
            weightx[i] = 2

    data['label'] = weightx.astype('int')
    data.to_csv('data/mango_weight_label_sort_3_label_.csv')

    weight_sort_data = data.sort_values(by='label')
    # print(weight_sort_data.info())
    # weight_sort_data.to_csv('data/mango_weight_label_sort_2_label_.csv')

    print((weight_sort_data['label'] == 0).sum())
    print((weight_sort_data['label'] == 1).sum())
    print((weight_sort_data['label'] == 2).sum())


def sweet_labels(data):
    sweetx = data['sweet'].values.copy()

    for i in range(len(sweetx)):
        if sweetx[i] > 15:
            sweetx[i] = 0
        elif sweetx[i] > 13:
            sweetx[i] = 1
        elif sweetx[i] <= 13:
            sweetx[i] = 2
        else:
            sweetx[i] = 99

    data['label'] = sweetx.astype('int')
    data.to_csv('data/mango_sweet_label_sort_3_label_.csv')

    sweetx_sort_data = data.sort_values(by='label')
    # print(sweetx_sort_data)
    # sweetx_sort_data.to_csv('data/mango_sweet_label_sort_2_label_.csv')

    print((sweetx_sort_data['label'] == 0).sum())
    print((sweetx_sort_data['label'] == 1).sum())
    print((sweetx_sort_data['label'] == 2).sum())


if __name__ == '__main__':
    # weight_data = pd.read_csv('data/mango_weight.csv')
    # sweet_data = pd.read_csv('data/mango_sweet.csv')
    data = pd.read_csv('data/mango.csv')

    # weight_labels(data)
    sweet_labels(data)
