from EFRingDataLoader import DataLoader,time_to_freq
from utils import numpy_to_tensor,check_accuracy,save_checkpoint,load_checkpoint,get_predict,print_reports
import numpy as np
from GetLoader import get_loader

## select the model here ##
from model.densenet import Model
# from model.efficientnet import Model
# from model.resnet50 import Model
# from model.seresnet import Model
# from model.vgg16_bn import Model
# from model.vit16 import Model

import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

# Set devices
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# label smoothing
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def get_train_test_val_loader():
    mDataLoader = DataLoader('../recording',
                             nb_windows=1,
                             aug_rate=0.5,
                             train_file=[
                                 "gestic_data_off_user1.csv",
                                 "gestic_data_off_user2.csv",
                                 "gestic_data_off_user3.csv",
                                 "gestic_data_off_user4.csv",
                                 "gestic_data_off_user5.csv",
                                 "gestic_data_off_user6.csv",
                                 "gestic_data_off_user7.csv",
                                 "gestic_data_off_user8.csv",
                                 "gestic_data_off_user9.csv",
                                 "gestic_data_off_user10.csv",
                                 "gestic_data_off_user11.csv",
                                 "gestic_data_off_user12.csv",
                                 "gestic_data_off_user13.csv",
                                 "gestic_data_off_user14.csv",
                                 "gestic_data_off_user15.csv",
                                 "gestic_data_off_user16.csv",
                             ],
                             val_file=[
                                 "gestic_data_off_user1_test.csv",
                                 "gestic_data_off_user2_test.csv",
                                 "gestic_data_off_user3_test.csv",
                                 "gestic_data_off_user4_test.csv",
                                 "gestic_data_off_user5_test.csv",
                                 "gestic_data_off_user6_test.csv",
                                 "gestic_data_off_user7_test.csv",
                                 "gestic_data_off_user8_test.csv",
                                 "gestic_data_off_user9_test.csv",
                                 "gestic_data_off_user10_test.csv",
                                 "gestic_data_off_user11_test.csv",
                                 "gestic_data_off_user12_test.csv",
                                 "gestic_data_off_user13_test.csv",
                                 "gestic_data_off_user14_test.csv",
                                 "gestic_data_off_user15_test.csv",
                                 "gestic_data_off_user16_test.csv",
                             ])
    X_train, X_val, y_train, y_val = mDataLoader.get_data()

    class_num = len(np.unique(y_train))

    train_loader,val_loader=get_loader(X_train,y_train,batch_size=4,divide=True,test_size=0.2)
    test_loader = get_loader(X_val, y_val, batch_size=4, divide=False)

    return train_loader, val_loader, test_loader, class_num

def train(filename="my_checkpoint.pth.tar"):

    train_loader, val_loader, test_loader, class_num = get_train_test_val_loader()
    torch.backends.cudnn.benchmark = True

    # Hyperparameters of model
    learning_rate = 1e-5
    num_epochs = 128

    # initialize the model
    classification_model = Model(output=class_num).to(device)

    # Loss and optimizer
    criterion = LabelSmoothing(0.1) 
    optimizer = optim.Adam(classification_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)

    # Train the model
    best_test_accuracy = 0
    step = 0

    for epoch in range(num_epochs):
        print("This is the %d epoch" % (epoch + 1))

        Train_losses_sum = 0
        num_losses = 0

        classification_model.train()
        for batch_idx, (data, targets) in tqdm(enumerate(train_loader),total=len(train_loader), leave=False):
            # Get data to cuda if possible
            data = data.type(torch.FloatTensor).to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = classification_model(data)
            loss = criterion(scores, targets)
            Train_losses_sum+=loss.item()
            num_losses+=1

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

        #_ = check_accuracy(train_loader, classification_model)
        test_accuracy = check_accuracy(val_loader, classification_model)

        if test_accuracy > best_test_accuracy:
            checkpoint = {'state_dict': classification_model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(state=checkpoint, filename=filename)
            best_test_accuracy = test_accuracy

        print("Now the best accuracy is %.4f" % (best_test_accuracy * 100))

        step += 1
        scheduler.step(Train_losses_sum/ num_losses)

def test(filename="my_checkpoint.pth.tar"):
    train_loader, val_loader, test_loader, class_num = get_train_test_val_loader()

    # initialize model
    learning_rate = 0.001
    classification_model = Model(output=class_num).to(device)
    optimizer = optim.Adam(classification_model.parameters(), lr=learning_rate)

    step = load_checkpoint(torch.load(filename), classification_model, optimizer)

    corrects,predictions=get_predict(test_loader, classification_model)
    print_reports(corrects, predictions)

if __name__ == "__main__":
    
    train(filename="model.pth.tar")
    test(filename="model.pth.tar")
