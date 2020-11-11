class NaryTreeLSTM(nn.Module):
    def __init__(self, dim_h=500, vec_len=500, max_child_num=6):
        super(NaryTreeLSTM, self).__init__()

        self.dim_vec = vec_len
        self.dim_h = dim_h
        self.max_child_num = max_child_num
        self.device = torch.device("cuda")

        # input gate
        self.Wi = nn.parameter.Parameter(torch.randn(self.dim_h, self.dim_vec, device=self.device))
        self.bi = nn.parameter.Parameter(torch.zeros(self.dim_h, device=self.device))
        self.Uis = nn.parameter.Parameter(torch.randn(max_child_num, self.dim_h, self.dim_h, device=self.device))
        # self.register_parameter("Wi", self.Wi)

        # forget gate
        self.Wf = nn.parameter.Parameter(torch.randn(dim_h, self.dim_vec, device=self.device))
        self.bf = nn.parameter.Parameter(torch.zeros(dim_h, device=self.device))
        self.Ufs = nn.parameter.Parameter(torch.randn(max_child_num, dim_h, dim_h, device=self.device))

        # # output gate
        self.Wo = nn.parameter.Parameter(torch.randn(dim_h, self.dim_vec, device=self.device))
        self.bo = nn.parameter.Parameter(torch.zeros(dim_h, device=self.device))
        self.Uos = nn.parameter.Parameter(torch.randn(max_child_num, dim_h, dim_h, device=self.device))

        # # u
        self.Wu = nn.parameter.Parameter(torch.randn(dim_h, self.dim_vec, device=self.device))
        self.bu = nn.parameter.Parameter(torch.zeros(dim_h, device=self.device))
        self.Uus = nn.parameter.Parameter(torch.randn(max_child_num, dim_h, dim_h, device=self.device))

    def forward(self, tree, inputs):
        node_num = len(tree.treepositions())
        node_embedding = torch.zeros(node_num, 500, device=inputs.device)
        c, h, outputs = self.encode(tree, inputs, node_embedding, 0)
        return c, h, outputs

    def encode(self, tree, inputs, node_emb, idx_node=0):
        c_children = []
        h_children = []
        cur_idx = idx_node
        idx_node += 1
        _input = inputs[cur_idx]
        if isinstance(tree, Tree):
            for child_idx in range(len(tree)):
                if child_idx == self.max_child_num:
                    break
                child = tree[child_idx]
                c_subtree, h_subtree, node_emb = self.encode(child, inputs, node_emb, idx_node)
                idx_node += (len(child.treepositions()) if isinstance(child, Tree) else 1)
                c_children.append(c_subtree)
                h_children.append(h_subtree)
        else:
            c_children.append(_input)
            h_children.append(_input)

        c_q, h_q = self.encode_node(_input, c_children, h_children)
        node_emb[cur_idx] = h_q
        return c_q, h_q, node_emb

    def encode_node(self, x, cs, hs):
        x = torch.reshape(x, (self.dim_h,))
        _Ui = torch.zeros(self.dim_h, device=self.device)
        _Uo = torch.zeros(self.dim_h, device=self.device)
        _Uu = torch.zeros(self.dim_h, device=self.device)
        _Uf = [torch.zeros(self.dim_h, device=self.device) for _ in range(len(cs))]

        for idx in range(len(cs)):
            _Ui = torch.add(_Ui, torch.matmul(self.Uis[idx], hs[idx]))
            _Uo = torch.add(_Uo, torch.matmul(self.Uos[idx], hs[idx]))
            _Uu = torch.add(_Uu, torch.matmul(self.Uus[idx], hs[idx]))
            for j in range(len(cs)):
                _Uf[idx] = torch.add(_Uf[idx], torch.dot(self.Ufs[idx][j].data, hs[j]))

        i = torch.sigmoid(torch.add(torch.add(torch.matmul(self.Wi, x), _Ui), self.bi))
        o = torch.sigmoid(torch.add(torch.add(torch.matmul(self.Wo, x), _Uo), self.bo))
        f = [torch.sigmoid(torch.add(torch.add(torch.matmul(self.Wf, x), _Uf[idx]), self.bf)) for idx in range(len(cs))]
        u = torch.tanh(torch.add(torch.add(torch.matmul(self.Wu, x), _Uu), self.bu))

        c = torch.zeros(self.dim_h, device=self.device)
        for idx in range(len(cs)):
            c = torch.add(c, torch.mul(f[idx], cs[idx]))
        c = torch.add(torch.mul(i, u), c)
        h = torch.mul(o, torch.tanh(c))
        return c, h
