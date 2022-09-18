import torch
import utils
import numpy as np
import torch.nn as nn
import matplotlib as plt


train_text, train_label = utils.read_file("Dataset/reviews_train.tsv")
train_label = np.array(train_label[1:])
train_text = np.array(train_text[1:])
dictionary = utils.bag_of_words(train_text)
train_set = utils.extract_bow_feature_vectors(train_text, dictionary)
test_text, test_label = utils.read_file("Dataset/reviews_test.tsv")
test_label = np.array(test_label[1:])
test_label = np.array(test_label)
test_text = np.array(test_text[1:])
test_set = utils.extract_bow_feature_vectors(test_text, dictionary)


train_set = torch.from_numpy(train_set.astype(np.float32))
train_label = torch.from_numpy(train_label.astype(np.float32))
test_set = torch.from_numpy(test_set.astype(np.float32))
test_label = torch.from_numpy(test_label.astype(np.float32))
train_label = train_label.view(train_label.shape[0], 1)
test_label = test_label.view(test_label.shape[0], 1)
n_sample, n_features = train_set.shape
#################################### Model ############################################
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.tanh(self.linear(x))
        return y_pred

model = Model(n_features)

#################################### loss and optimizer ############################################
learning_rate = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#################################### training loop ############################################

num_epoch = 10000

for epoch in range(num_epoch):
    label_predicted = model(train_set)
    loss = criterion(train_label, label_predicted)

    loss.backward()
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 1000 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


with torch.no_grad():
    y_predicted = model(test_set)
    n = y_predicted.shape[0]
    sum = 0
    for i in range(n):
        if(y_predicted[i] * test_label[i] > 0):
            sum +=1
    print(sum/float(n))