from config import opts
from datasets.build_dataset import get_dataset
import torch
import os
import numpy as np
from models.yolo import Yolo, YoloWithLoss
from utils.autoanchors import kmean_anchors
from tqdm import tqdm

anchors = [ [[19.88362163, 43.27377509],
            [22.96717419, 46.93803082],
            [26.29045747, 50.63705496]],
            [[24.19947973, 58.73719134],
            [28.91440057, 60.22118699],
            [33.61075118, 62.88708615]],
            [[28.54104203, 75.82553078],
            [32.93518201, 83.20090047],
            [38.05241252, 89.77204744]] ]

if __name__ == "__main__":
    # print(kmean_anchors(opt))
    opt = opts().init()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    model = Yolo(opt, 21, anchors, 13).to(opt.device)
    if opt.load_model != "":
        model.load_state_dict(torch.load(opt.load_model, map_location=opt.device))
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay = opt.weight_decay)
    epochs = opt.num_epochs

    Dataset = get_dataset(opt.dataset)
    train_set = Dataset(opt, "train")
    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=False,
            drop_last=True,
            shuffle = True,
            collate_fn = train_set.collate_fn
    )

    val_set = Dataset(opt, "val")
    val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            pin_memory=False,
            drop_last=True,
            shuffle = True,
            collate_fn = val_set.collate_fn
    )
    exp = len(os.listdir(opt.output_dir)) if os.path.isdir(opt.output_dir) else 0
    os.makedirs(os.path.join(opt.output_dir, f"exp{exp}"), exist_ok=True)
    logging_file = open(os.path.join(opt.output_dir, f"exp{exp}", "log.txt"), 'w')

    modelwithloss = YoloWithLoss(model)
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total = len(train_loader))
        total_train = 0
        for i, (inp, bboxes) in pbar:
            modelwithloss.zero_grad()
            optimizer.zero_grad()
            out, loss = modelwithloss(inp.to(opt.device), bboxes)
            total_loss = loss['total_loss']
            
            total_train += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch: {epoch+1} | Total loss: {round(total_loss.item(), 4)} | Coord Loss: {round(loss['coord_loss'], 4)} | " + \
                                 f"Obj Loss: {round(loss['obj_loss'], 4)} | Cls Loss: {round(loss['cls_loss'], 4)}")
            exit()
        total_train /= len(train_loader)
        with torch.no_grad():
            pbar_val = tqdm(enumerate(val_loader), total = len(val_loader))
            pbar_val.set_description(f"Valid epoch {epoch+1}")
            total_val = 0
            for i, (inp, bboxes) in pbar_val:
                out, loss = modelwithloss(inp.to(opt.device), bboxes)
                total_loss = loss['total_loss']
                total_val+=total_loss.item()
            print(f"==========> Validation loss epoch {epoch+1}: {round(total_val/len(val_loader), 4)}")
        logging_file.write(f"Epoch {epoch+1}: Train loss: {round(total_train, 4)} | Val loss: {round(total_val/len(val_loader), 4)}\n")
        if epoch % opt.lr_step == 0:
            # power = 1.0
            # lr = opt.lr  * (1 - float(epoch/opt.num_epochs))**power
            lr = opt.lr / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        torch.save({
            "model": modelwithloss.model.state_dict(),
            "anchors": anchors
        }, os.path.join(opt.output_dir, f"exp{exp}", f"model_{epoch+1}.pth"))

    logging_file.close()