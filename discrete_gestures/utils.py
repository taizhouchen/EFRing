import torch
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from tqdm import tqdm
# Set devices
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def numpy_to_tensor(X,y):

    y = [int(num) for num in y]
    length = X.shape[0]

    X = torch.from_numpy(X).type(torch.float)
    label=torch.randn(length, ).type(torch.LongTensor)

    for i in range(length):
        label[i] = y[i]
    return X,label

# save_checkpoint and load_checkpoint
def save_checkpoint(state,filename="my_checkpoint.pth.tar"):

    print("=> Saving Checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint,model,optimizer):

    print("=> Loading Checkpoint")

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# check accuracy
def check_accuracy(loader,model):

    num_correct=0
    num_samples=0
    model.eval()

    with torch.no_grad():
        for batch_idx,(x,y) in tqdm(enumerate(loader),total=len(loader), leave=False):
            x=x.type(torch.FloatTensor).to(device=device)
            y=y.to(device=device)

            scores=model(x)
            # 64x6
            _,predictions=scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        return float(num_correct)/float(num_samples)

def get_predict(loader,model):

    corrects = torch.ones((1,)).to(device)
    predictions = torch.ones((1,)).to(device)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.type(torch.FloatTensor).to(device=device)
            y = y.to(device=device)

            scores = model(x)
            # 64x6
            _, prediction = scores.max(1)

            corrects = torch.cat((corrects, y), dim=0)
            predictions = torch.cat((predictions, prediction), dim=0)

    corrects = corrects[1:].cpu().detach().numpy()
    predictions = predictions[1:].cpu().detach().numpy()

    return corrects.astype(int),predictions.astype(int)

def print_reports(corrects,predictions):

    print("Accuracy:{:.4f}".format(accuracy_score(corrects, predictions)))
    print("Classification Report:\n{}".format(classification_report(corrects, predictions,digits=4)))
    print("Confusion matrix:\n{}".format(confusion_matrix(corrects, predictions)))
