import time
import os
import json
import torch
from data_process.dataset import Task
from model.model import *

if __name__ == "__main__":

    epochs = 300


    for kfold in range(5):
        min_loss = 4
        net = DTImodel(64, 25)
        task_1 = Task(net, "../dataset/4PDBbind_train.csv")
        trainloader_lst, valid_loader_lst = task_1.load_data()
        train_loss_lst = []
        time_lst = []
        test_loss_lst = []
        pred_train = []
        pred_test = []
        train_mse = []
        train_rp = []
        test_mse = []
        test_rp = []
        start_time = time.time()
        train_loader = trainloader_lst[kfold]
        valid_loader = valid_loader_lst[kfold]
        num = 0
        for epoch in range(epochs):
            # ——————————————————————train————————————————————————
            train_loss, train_MSE, train_Rp = task_1.train(trainloader=train_loader)
            execution_time = time.time() - start_time
            # ——————————————————————validation———————————————————
            test_loss, test_MSE, test_Rp = task_1.test(valid_loader)
            time_lst.append(execution_time)
            # ——————————————————————correlation——————————————————
            train_loss_lst.append(train_loss)
            test_loss_lst.append(test_loss)

            train_mse.append(train_MSE)
            test_mse.append(test_MSE)

            train_rp.append(train_Rp)
            test_rp.append(test_Rp)

            # ——————————————————————save_model—————————————————————

            if test_loss < min_loss:
                min_loss = test_loss
                num += 1
                if num % 2 == 0:
                    torch.save(net.state_dict(), f'./data_process/cache/{kfold + 1}_1.pkl')
                else:
                    torch.save(net.state_dict(), f'./data_process/cache/{kfold + 1}_2.pkl')

            # print(f"[{epoch + 1}]\ntrain_loss: {train_loss:.4f}\ntest_loss: {test_loss:.4f}, train_time: {execution_time:.4f},train_MSE: {train_MSE:.4f}")
            print(
                '\n-------------------------------fold:%d------------epochs:%d------------------------------------' % (
                kfold + 1, epoch + 1))
            print(
                f"[train]    train_loss: {train_loss:.4f},  train_MSE: {train_MSE:.4f},  train_Rp:{train_Rp}，  train_time: {execution_time:.4f}")
            print(f"[test]     test_loss:  {test_loss:.4f},   test_MSE: {test_MSE:.4f},   test_Rp:{test_Rp}")

            # ——————————————————————performance—————————————————————
        save_path = "./data_process/data_cache/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        dict = {"train_loss": train_loss_lst, "time": time_lst, "test_loss": test_loss_lst}
        # dict = {"train_loss": train_loss_lst, "time": time_lst,"train_MSE":train_mse,"train_Rp":train_rp,"test_loss": test_loss_lst,"test_MSE":test_mse,"test_Rp":test_rp}
        with open(save_path + f"{kfold + 1}.json", "w") as f:
            json.dump(dict, f)


