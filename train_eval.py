# import pandas as pd
# from src.datasets.spaceship_titanic_dataset import SpaceshipTitanicDataset
# from src.models.spaceship_model import SpaceShipModel
# from src.pre_processors.pre_processor_spaceship import PreProcessorSpaceship
# from src.transforms.to_torch_tensor import NumpyToTorchTensor
# from src.utils.util_path import UtilPath
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# import numpy as np
# # column_index,len_data
# root_path = UtilPath.get_root_path()
# def main():
#     vocab_columns = {
#         "PassengerId": [0,2],
#         "HomePlanet": [1,1],
#         "CryoSleep": [2,2],
#         "Cabin": [3,3],
#         "VIP": [6,2],
#         "Name": [12,2]
#     }

#     def process_features(features,transform_int:NumpyToTorchTensor,transform_float:NumpyToTorchTensor):
#         indexes = []
#         shift = 0
#         for k,v in vocab_columns.values():
#             for i in range(k+shift,k+v+shift):
#                 indexes.append(i)
#             shift += v-1
#         features_embedding = []
#         remain_features = []
#         with torch.no_grad():
#             for i,passenger_feature in enumerate(features):
#                 features_embedding.append([])
#                 remain_features.append([])
#                 for j,feature in enumerate(passenger_feature.tolist()):
#                     if j in indexes:
#                         features_embedding[i].append(feature)
#                     else:
#                         remain_features[i].append(feature)
#         return transform_int(np.array(features_embedding)),transform_float(np.array(remain_features))
                        
#     pre_processor = PreProcessorSpaceship(root_path+"/datasets/spaceship-titanic/train.csv")
#     pre_processed_data, vocab = pre_processor.pre_process()

#     len_vocab = len(vocab[0])
#     dim_embedd = 16
#     batch_size = 32 
#     model = SpaceShipModel(19,12,len_vocab,dim_embedd)
#     loss_fn = nn.MSELoss()
#     optim = torch.optim.Adam(model.parameters(),0.001)
#     epochs = 5
#     transform_float = NumpyToTorchTensor(dtype=float)
#     transform_int = NumpyToTorchTensor(dtype=int)

#     train_dataset = SpaceshipTitanicDataset(pre_processed_data,transform=transform_float)
    
#     dataloader = DataLoader(train_dataset,batch_size, shuffle=True, num_workers=2)

#     for epoch in range(epochs):
#         for features,target in dataloader:
#             embedding_features, remain_features = process_features(features, transform_int,transform_float)
#             target = transform_int(target)
#             optim.zero_grad()
#             predictions = model(embedding_features,remain_features,len(features))
#             embedd_target = model.embedding(target)
#             loss = loss_fn(predictions, embedd_target)
            
#             loss.backward()
            
#             optim.step()

#         calc = len(train_dataset)/batch_size
#         print(f"epoch {epoch + 1} - loss {loss/calc}")

#     torch.save(model.state_dict(), 'checkpoints/spaceship_titanic_model.pth')

#     model.eval()
    
#     pre_processed_test_data = pre_processor.pre_process_test(root_path+"/datasets/spaceship-titanic/test.csv")
#     test_dataset = SpaceshipTitanicDataset(pre_processed_test_data,transform=transform_float,type='eval')
#     df_sample_submission = pd.DataFrame({"PassengerId":[],"Transported":[]}) 
    
#     embedd_true = model.embedding(transform_int(np.array(vocab[0]['True'])))
#     embedd_true = embedd_true.reshape((1,dim_embedd))
    
#     embedd_false = model.embedding(transform_int(np.array(vocab[0]['False'])))
#     embedd_false = embedd_false.reshape((1,dim_embedd))
#     for i in range(len(test_dataset)):
#         x,_= test_dataset[i]
#         embedding_features, remain_features = process_features([x], transform_int,transform_float)
#         prediction = model(embedding_features,remain_features,1)
#         mse_true = loss_fn(prediction,embedd_true)
#         mse_false = loss_fn(prediction,embedd_false)
#         if mse_false < mse_true:
#             pred = "False"
#         else:
#             pred = "True"
#         string = f"{vocab[1][int(x[0].item())]}_{vocab[1][int(x[1].item())]}"
#         dict_data = {
#             "PassengerId":[string],
#             "Transported": pred
#         }
#         temp_df  = pd.DataFrame(dict_data)
#         df_sample_submission = pd.concat([temp_df,df_sample_submission],ignore_index =True)
#     df_sample_submission.to_csv("datasets/spaceship-titanic/sample_submission.csv",index=False)









