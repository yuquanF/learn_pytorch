import torch
from torch import nn
from data import get_data_iter
from model import load_model_from_timm

device = torch.device("cuda:0")
device_ids = [0, 1]
img_label = './label.csv'
batch_size = 64
model_name = "vit_base_patch16_224"


def train(epochs):
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        batch_num = 0
        for x1, y1 in data_iter:
            # print(type(x1))  # <class 'torch.Tensor'>
            # print(x1.shape)  # torch.Size([64, 3, 224, 224])
            # print(type(y1))  # <class 'torch.Tensor'>
            # print(y1)
            # # tensor([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
            # #           0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            # #           0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1])
            # break
            batch_num += 64
            y1 = y1.to(device)
            x1 = x1.to(device)
            optimizer.zero_grad()
            pre = model(x1)
            # print(pre)      # tensor([[ 0.3571, -0.0912],...]])
            loss = loss_func(pre, y1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = torch.argmax(pre, axis=1)
            train_acc += (y1 == pred).sum().item()
            print("\r", "epoch:", epoch + 1, "acc:{:6.4f}".format(train_acc / batch_num),
                  "loss:{:6.4f}".format(train_loss),
                  end='')
        print("")


if __name__ == '__main__':
    # 加载模型
    print("加载模型中~~~~~")
    model = load_model_from_timm(model_name, device, device_ids, pretrained=True, num_classes=2)

    # 加载数据
    print("模型加载完成，数据加载中~~~~~")
    data_iter = get_data_iter(img_label, size=(224, 224), batch_size=batch_size)

    # 训练
    print("数据加载完成，开始训练~~~~~")
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()
    train(epochs=150)

    # 保存模型
    torch.save(model, 'model.pkl')
