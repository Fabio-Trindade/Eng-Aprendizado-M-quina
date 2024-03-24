import torch
from sklearn.metrics import accuracy_score, precision_score

class UtilTrain:
    @staticmethod
    def train(model,optim,loss_fn,features,labels):
        optim.zero_grad()
        predictions = model(features)
        loss = loss_fn(predictions,labels)
        loss.backward()
        optim.step()
        return loss.detach()
    
    def train_epochs(epochs,dataloader,val_dataset,model,optim,loss_fn):
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            count = 0
            for features, target in dataloader:
                loss = UtilTrain.train(model,optim,loss_fn,features,target)
                total_loss += loss
                count+=1
            prec,acc = UtilTrain.eval(model,val_dataset)
            print(f"epoch {epoch + 1} - loss = {total_loss/count} - val_precision = {prec:.2f} - val_acc = {acc:.2f}")
    
    def eval(model,dataset):
        model.eval()
        features, targets = dataset[:]
        predictions = model(features)
        for i in range(len(predictions.detach().numpy().ravel())):
            predictions[i] = 1 if predictions[i].item() > 0.5 else 0
        prec = precision_score(targets.detach().numpy().ravel(), predictions.detach().numpy().ravel(),  zero_division='warn')
        acc = accuracy_score(targets.detach().numpy().ravel(),predictions.detach().numpy().ravel())
        return prec,acc
