import torch

def decode_output(outputs, anchors, strides):
    dets = []
    anchors = torch.FloatTensor(anchors)
    anchor_grid = anchors.clone().view(len(outputs), 1, -1, 1, 1, 2).to(outputs[0].device)
    for i in range(len(outputs)):
        y = outputs[i].clone()
        bs, c, h, w, num_out = y.shape
        grid = make_grid(h, w).to(y.device)
        #Sigmoid the  centre_X, centre_Y. and object confidencce
        y[..., 0] = torch.sigmoid(y[..., 0])
        y[..., 1] = torch.sigmoid(y[..., 1])
        y[..., 4] = torch.sigmoid(y[..., 4])
        y[..., :2] = (y[..., :2] + grid) * strides[i]

        y[..., 2:4] = torch.exp(y[..., 2:4]) * anchor_grid[i]
        y[..., 5:] = torch.softmax(y[..., 5:], dim=-1)
        dets.append(y.view(bs, -1, num_out))
    dets = torch.cat(dets, 1)
    return dets

def make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()