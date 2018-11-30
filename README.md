# N-aryTreeLSTM

An implementation (in **MXNet**) of the N-ary Tree LSTM described in the paper ***[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/abs/1503.00075),Kai Sheng Tai, Richard Socher, and Christopher Manning.***

![equation](https://github.com/mzx5464/N-aryTreeLSTM/blob/master/asset/equation.png)

**Note: "For large values of N (*the allowed maximum number of child*), these additional parameters are impractical and may be tied or fixed to zero."**

 Call function `c, h = self.encoder(tree, inputs, ctx)`
