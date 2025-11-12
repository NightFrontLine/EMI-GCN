class EMI_GCN2(nn.Module):
    """
    期望输入:
      data_dict = {"graph": [N,N] tensor, "flow_x": [B,N,H,1] tensor}
    输出:
      [B, N, H_out, 1]
    """
    def __init__(self, in_c, hid_c, out_c, kernel_size=3, num_nodes=170):
        super(EMI_GCN2, self).__init__()
        self.num_nodes = num_nodes
        self.linear_time = nn.Linear(1, hid_c)  # (B,N,H,1)->(B,N,H,D)

        # learnable graphs / gates
        Nn = num_nodes
        self.A_Dynamic = nn.Parameter(torch.clamp(torch.randn(Nn, Nn), 0, 1), requires_grad=True)
        self.A_W1 = nn.Parameter(torch.clamp(torch.randn(Nn, Nn), -1, 1), requires_grad=True)
        self.A_W2 = nn.Parameter(torch.clamp(torch.randn(Nn, Nn), -1, 1), requires_grad=True)
        self.A_Mutual_Fusion  = nn.Parameter(torch.clamp(torch.randn(Nn, Nn), 0, 1), requires_grad=True)
        self.W_Mutual_Flow    = nn.Parameter(torch.clamp(torch.randn(Nn, Nn), -1, 1), requires_grad=True)
        self.W_Mutual_Speed   = nn.Parameter(torch.clamp(torch.randn(Nn, Nn), -1, 1), requires_grad=True)
        self.W_Mutual_Occupation = nn.Parameter(torch.clamp(torch.randn(Nn, Nn), -1, 1), requires_grad=True)
        self.W_Dis = nn.Parameter(torch.clamp(torch.randn(Nn, Nn), -1, 1), requires_grad=True)
        self.W_Pearson = nn.Parameter(torch.clamp(torch.randn(Nn, Nn), -1, 1), requires_grad=True)
        self.W_TE = nn.Parameter(torch.clamp(torch.randn(Nn, Nn), -1, 1), requires_grad=True)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=hid_c, num_heads=1, batch_first=True)
        self.ETrans = nn.TransformerEncoderLayer(d_model=hid_c * 2, dropout=0.1, nhead=2, batch_first=True)
        self.ST_EDA = STEDA_Layer(in_features=hid_c * 2, hid_features=hid_c)

        self.conv1d_time = nn.Conv1d(in_channels=hid_c * 2, out_channels=hid_c * 2,
                                     kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.BiGRU = nn.GRU(input_size=hid_c * 2, hidden_size=hid_c, bidirectional=True, batch_first=True)

        self.linear_hidto2hid = nn.Linear(hid_c, hid_c * 2)
        self.linear_2hidtohid = nn.Linear(hid_c * 2, hid_c)
        self.linear_2hidtoout = nn.Linear(hid_c * 2, out_c)
        self.linear_ED4hidto2hid = nn.Linear(hid_c // 4, 2 * hid_c)

        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.k1 = nn.Parameter(torch.clamp(torch.randn(1), -1, 1), requires_grad=True)
        self.v1 = nn.Parameter(torch.clamp(torch.randn(1), -1, 1), requires_grad=True)
        self.v2 = nn.Parameter(torch.clamp(torch.randn(1), -1, 1), requires_grad=True)

        # buffers: 常量图（运行前由外部注入）
        self.register_buffer("A_Dis_const", None)
        self.register_buffer("A_Pearson_const", None)
        self.register_buffer("A_TE_const", None)
        self.register_buffer("A_Flow_Mutual_const", None)
        self.register_buffer("A_Speed_Mutual_const", None)
        self.register_buffer("A_Occupation_Mutual_const", None)

    @staticmethod
    def process_graph(graph_data):
        Nn = graph_data.size(0)
        I = torch.eye(Nn, dtype=graph_data.dtype, device=graph_data.device)
        graph_data = graph_data + I
        deg = torch.sum(graph_data, dim=-1, keepdims=False).pow(-1)
        deg[deg == float("inf")] = 0.
        Dinv = torch.diag(deg)
        return torch.mm(Dinv, graph_data)

    def forward(self, data, device, database_tag="GENERIC"):
        # 距离图
        flow_x = data["flow_x"].to(device)  # [B?,N,H,1] 或 [N,H,1] [32,170,12,1]
        print(f"flow_x.shape={flow_x.shape}")
        if flow_x.dim() == 3:
            flow_x = flow_x.unsqueeze(0)
        B, Nn, H, _ = flow_x.shape
        print(f"B={B}, Nn={Nn}, H={H}") # [32,170,12]
        X0 = self.linear_time(flow_x)  # [B,N,H,D] [32, 170, 12, 64]
        print(f"原始交通特征维度:X0.shape={X0.shape}")
        # ==========MCGE-GAttF模块=======================================================
        print("==============================================MCGE-GAttF模块=======================================================")
        print(f"X_Dis.shape={X_Dis.shape}")  # [32,170,12,64]
        print(f"X_Pearson.shape={X_Pearson.shape}")  # [32,170,12,64]
        print(f"池化后的X_Dis_Pool.shape={X_Dis_pool.shape}")  # [B,N,D] [32,170,64]
        print(f"池化后的X_Pearson_Pool.shape={X_Pearson_pool.shape}")  # [B,N,D] [32,170,64]
        print(f"S_Dis.shape={S_Dis.shape}")  # [B,N,D] [32,170,64]
        print(f"S_Pearson.shape={S_Pearson.shape}")  # [B,N,D] [32,170,64]
        print(f"S.shape={S.shape}")
        print(f"X_Multi.shape={X_Multi.shape}")  # [B,N,H,D] [32, 170, 12, 64]
        def pos(x): return F.softplus(x)

        # MIC-NTE
        print("==============================================MIC-NTE模块=======================================================")
        if database_tag in ("PEMSD8", "PEMSD4"):
            A_Flow_Mutual  = self.A_Flow_Mutual_const if self.A_Flow_Mutual_const is not None else torch.zeros_like(A_Dis)
            A_Speed_Mutual = self.A_Speed_Mutual_const if self.A_Speed_Mutual_const is not None else torch.zeros_like(A_Dis)
            A_Occ = self.A_Occupation_Mutual_const if self.A_Occupation_Mutual_const is not None else torch.zeros_like(A_Dis)
            num = pos(self.W_Mutual_Flow) * A_Flow_Mutual + pos(self.W_Mutual_Speed) * A_Speed_Mutual + pos(self.W_Mutual_Occupation) * A_Occ
            den = pos(self.W_Mutual_Flow) + pos(self.W_Mutual_Speed) + pos(self.W_Mutual_Occupation) + 1e-6
            A_mutual_fusion = num / den
        else:
            A_Speed_Mutual = self.A_Speed_Mutual_const if self.A_Speed_Mutual_const is not None else torch.zeros_like(A_Dis)
            A_mutual_fusion = pos(self.W_Mutual_Speed) * A_Speed_Mutual

        A_TE = self.A_TE_const if self.A_TE_const is not None else torch.zeros_like(A_Dis)


        print("==============================================EDGF模块=======================================================")
        A_dynamic_fused = pos(self.A_Dynamic) + self.A_W1 * A_Dis + self.A_W2 * A_Pearson + A_mutual_fusion + self.W_TE * A_TE

        # AGCE模块
        print("==============================================AGCE模块=======================================================")
        # X_Multi和A_Dynamic_fused输入到GCN
        print(f"X_Enhace.shape={X_Enhance.shape}")  # [32, 170, 12, 64]

        print(f"X_Enhance_2*维度和平均池化映射后={X_node.shape}")  # [32,170,12,64]->[32,170,12,128]->[32,170,128]

        # GCN的输出X_Enhance输入到TransformerEncoderLayer中
        print(f"X_Enhance_//2维度后={X_node.shape}")  # [32,170,64]

        print(f"AGGCE的输出X_Seq_Enhace.shape={X_for_seq.shape}") # [32,170,12,128]
        # 准备时序数据
        print(f"AGGCE输出reshape.shape={X_seq.shape}")  # [5440, 12, 128]

        # TSSE
        print("==================================TSSE=================================")
        print(f"GLU-CNN-LSTM输出: STRE.shape={X_steda.shape}")  # [5440, 12, 16]
        print(f"GLU-CNN-LSTM输出维度变换permute前: STRE.shape={X_STRE.shape}")

        print(f"X_Time_CNN_Input_1次permute后.shape={X_Time_CNN_Input.shape}")
        print(f"X_Time_CNN中.shape={X_Time_CNN.shape}")
        print(f"X_Time_CNN_Output2次permute后.shape={X_Time_CNN_Output.shape}")

        print(f"X_BiGRU.shape={X_gru.shape}")
        print(f"X_Last.shape={X_last.shape}")

        print(f"X_Dis残差.shape={gcn_dis_res.shape}")
        print(f"最后一层输出, 进过线性层到out.shape={out_map.shape}")
        return out_map.unsqueeze(-1)                              # [B,N,H_out,1] [32,170,12,1]



