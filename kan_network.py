import random  # 导入整个random模块


import numpy as np
import torch
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
# 在文件最开头添加（所有import之后）


SEED = 42  # 可替换为其他固定数值

#SEED = int(time.time())

# 设置全局随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 如果使用TensorFlow（在GAN部分）
import tensorflow as tf
tf.random.set_seed(SEED)



class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        # 添加随机种子
            g = torch.Generator().manual_seed(SEED)

            # 修改权值初始化
            torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5)*self.scale_base, generator=g)

            # 修改噪声生成
            noise = (
                (torch.rand(self.grid_size+1, self.in_features, self.out_features, generator=g) - 1/2)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
         # 添加数值截断
        x = torch.clamp(x, min=-1e3, max=1e3)
        y = torch.clamp(y, min=-1e3, max=1e3)
        # 添加微小噪声
        x = x + 1e-6 * torch.randn_like(x)
        y = y + 1e-6 * torch.randn_like(y)
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("x contains NaN or Inf values")
            return None
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("y contains NaN or Inf values")
            return None
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    # 似乎是没用的
    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        #print(f"我是kanlinear 的 update ：Input x shape: {x.shape}")  # 打印输入形状
        assert x.dim() == 2 and x.size(1) == self.in_features, f"x 必须是二维张量，且第二维大小为 {self.in_features}，当前形状为 {x.shape}"
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)
        # 检查 unreduced_spline_output 是否包含 NaN 或 Inf
        if torch.isnan(unreduced_spline_output).any() or torch.isinf(unreduced_spline_output).any():
            print("unreduced_spline_output contains NaN or Inf values")
            return
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
        assert len(layers_hidden) >= 2, "至少需要输入层和输出层"

    def forward(self, x: torch.Tensor, update_grid=False):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x  # 直接返回最后一层输出


    def extract_features(self, x: torch.Tensor):
        """特征提取方法（返回最后一层前的特征）"""
        with torch.no_grad():
            for layer in self.layers[:-1]:
                x = layer(x)
        return x

    def predict(self, x: torch.Tensor):
        """预测方法（包含完整前向过程）"""
        with torch.no_grad():
            return self.forward(x)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

    def fit(self, traindata, trainlabel, epochs=100, lr=0.001,
        batch_size=32, val_ratio=0.2, patience=15,
        update_grid_freq=5, grad_clip=1.0,
        device=torch.device("cpu"),
        verbose=True):
        """
        增强版训练函数

        参数说明：
        traindata: 训练数据 (numpy数组或torch.Tensor)
        trainlabel: 训练标签 (numpy数组或torch.Tensor)
        batch_size: 批处理大小
        val_ratio: 验证集比例
        patience: 早停耐心值
        update_grid_freq: 网格更新频率(epoch)
        grad_clip: 梯度裁剪阈值
        device: 训练设备
        """

        # 1. 数据预处理和类型转换
        if not isinstance(traindata, torch.Tensor):
            traindata = torch.FloatTensor(traindata)
        if not isinstance(trainlabel, torch.Tensor):
            trainlabel = torch.FloatTensor(trainlabel)
        if torch.isnan(traindata).any() or torch.isinf(traindata).any():
            print("traindata contains NaN or Inf values")
            return [], []
        if traindata.dim() != 2:
            raise ValueError(f"我是fit traindata 必须是二维张量，当前维度为 {traindata.dim()}")
        if traindata.size(1) != self.layers[0].in_features:
            raise ValueError(f"我是fit traindata 的第二维大小必须为 {self.layers[0].in_features}，当前为 {traindata.size(1)}")
        # 移动到指定设备
        self.to(device)
        traindata = traindata.to(device)
        trainlabel = trainlabel.to(device)

        # 2. 划分训练集和验证集
        dataset = torch.utils.data.TensorDataset(traindata, trainlabel)
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        print("训练集大小：", len(train_dataset))
        print("验证集大小：", len(val_dataset))
        # 3. 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.parameters(), betas=(0.9, 0.999))
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2)

        # 4. 优化器和调度器配置
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        criterion = nn.MSELoss()

        # 5. 训练状态初始化
        best_val_loss = float('inf')
        no_improve_epochs = 0
        train_losses = []
        val_losses = []

        # 6. 主训练循环
        for epoch in range(epochs):
            # 训练模式
            self.train()
            epoch_train_loss = 0

            for batch_x, batch_y in train_loader:
                #print(f"Batch y shape: {batch_y.shape}")  # 添加打印语句
                # 前向传播
                outputs = self(batch_x)
                #print(f"Outputs shape: {outputs.shape}")  # 添加打印语句

                # 计算损失（MSE + 正则化）
                mse_loss = criterion(outputs, batch_y)
                reg_loss = self.regularization_loss()
                total_loss = mse_loss + 0.1 * reg_loss

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

                # 参数更新
                optimizer.step()

                epoch_train_loss += total_loss.item()

            # 计算平均训练损失
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证模式
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = self(batch_x)
                    val_loss += criterion(outputs, batch_y).item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            # 动态更新网格
            if (epoch+1) % update_grid_freq == 0:
                with torch.no_grad():
                    self.update_grid(traindata)

            # 早停机制
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_epochs = 0
                # 保存最佳模型
                torch.save(self.state_dict(), 'best_kan_model.pth')
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break


        # 加载最佳模型
        self.load_state_dict(torch.load('best_kan_model.pth'))
        return train_losses, val_losses

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):

        current_x = x  # 保存当前层输入
        for layer in self.layers:
            # 仅在该层处理数据
            layer.update_grid(current_x, margin)
            # 获取该层输出作为下一层输入
            current_x = layer(current_x)