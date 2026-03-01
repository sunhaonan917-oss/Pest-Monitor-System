import torch

@torch.no_grad()
def predict_5way5shot_one_query(model, x_support, y_support, x_query, device):
    """
    x_support: [25,3,H,W]
    y_support: [25] 值为 0..4
    x_query:   [1,3,H,W]
    return: logits [5]  (5类得分)
    """

    model.eval()

    x_support = x_support.to(device)
    y_support = y_support.to(device)
    x_query = x_query.to(device)

    n_way = model.n_way
    n_support = model.n_support

    # 1) 把 support 按类别整理成 [5,5,3,H,W]
    #    为了稳妥：根据 y_support 排序
    support_sorted = []
    for c in range(n_way):
        idx = (y_support == c).nonzero(as_tuple=True)[0]
        if idx.numel() != n_support:
            raise ValueError(f"类别 {c} 的 support 张数不是 {n_support}，而是 {idx.numel()}")
        support_sorted.append(x_support[idx])
    x_support_5x5 = torch.stack(support_sorted, dim=0)  # [5,5,3,H,W]

    # 2) 把 query 复制 5 份，拼成 episode: [5, (5+1), 3,H,W] = [5,6,3,H,W]
    x_query_rep = x_query.repeat(n_way, 1, 1, 1)  # [5,3,H,W]
    x_query_rep = x_query_rep.unsqueeze(1)        # [5,1,3,H,W]
    x_episode = torch.cat([x_support_5x5, x_query_rep], dim=1)  # [5,6,3,H,W]

    # 3) 扁平化喂给 backbone：一共 30 张图
    x_flat = x_episode.view(-1, *x_episode.size()[2:])  # [30,3,H,W]

    # 4) backbone -> [30,512]
    feats_512 = model.feature(x_flat)

    # 5) disentangle：不要用 forward（里面会随机采样）
    #    我们用 encode 的 a_mean（稳定）
    a_mean, a_logvar, b_mean, b_logvar = model.disentangle_model.encode(feats_512)
    a_code = a_mean  # [30,64]

    # 6) fc -> [30,128] -> reshape 回 [5,6,128]
    z = model.fc(a_code)                    # [30,128]
    z = z.view(n_way, n_support + 1, -1)    # [5,6,128]

    # 7) 设置 n_query=1，然后走 gnn
    model.n_query = 1
    z_stack = [z.view(1, -1, z.size(2))]    # [1, 30, 128]
    scores = model.forward_gnn(z_stack)     # 输出 [5,5]（5个“复制query”的预测）

    # 8) 取平均，变成最终一个 query 的 [5] 得分
    logits = scores.mean(dim=0)             # [5]
    return logits
