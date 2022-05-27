# -*- coding:utf-8 -*-
# pylint: disable=no-member

from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon import rnn
# cnn-GRU


#空间卷积层
class Cnn(nn.Block):
    def __init__(self, **kwargs):
        super(Cnn, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(
            channels=64,
            kernel_size=(3, 2),
            padding=(1,0),
            strides=(1, 1)
        )
        self.conv2 = nn.Conv2D(
            channels=64,
            kernel_size=(5, 2),
            padding=(2, 0),
            strides=(1, 1)
        )
        self.conv3 = nn.Conv2D(
            channels=64,
            kernel_size=(7, 2),
            padding=(3, 0),
            strides=(1, 1)
        )
        self.convronghe = nn.Conv2D(
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 0),
            strides=(1, 1)
        )
    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        '''
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape
        #print("cnn x.shape")
        #print(x.shape)
        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            signal = x[:, :, :, time_step]
            #print("signal")
            #print(signal.shape)
            signal_SAt = signal.expand_dims(-1).transpose((0, 3, 1, 2))
            #print("signal_SAt")
            #print(signal_SAt.shape)
            c1 = self.conv1(signal_SAt)
            c2 = self.conv2(signal_SAt)
            c3 = self.conv3(signal_SAt)
            c = nd.concat(c1,c2,c3, dim=-1)
            ronghe = self.convronghe(c)
            outputs.append(ronghe)
        return nd.relu(nd.concat(*outputs, dim=-1))



class ASTGCN_block(nn.Block):
    def __init__(self, backbone, **kwargs):
        '''
        Parameters
        ----------
        backbone: dict, should have 6 keys,
                        "num_of_chev_filters",
                        "num_of_time_filters",
                        "time_conv_kernel_size",
                        "time_conv_strides",
        '''
        super(ASTGCN_block, self).__init__(**kwargs)
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']

        with self.name_scope():
            self.conv_SAt = Cnn()
            self.rong =nn.Conv2D(
            channels=1,
            kernel_size=(1, 1),
            strides=(1, 1)
            )
            self.time_LSTM = rnn.GRU(6,3)
            self.huiqu = nn.Conv2D(
                channels=64,
                kernel_size=(1, 1),
                strides=(1, 1)
            )
            '''
            self.time_conv = nn.Conv2D(
                channels=num_of_time_filters,
                kernel_size=(1, 3),
                padding=(0, 1),
                strides=(1, time_conv_strides))
            '''
            self.residual_conv = nn.Conv2D(
                channels=num_of_time_filters,
                kernel_size=(1, 1),
                strides=(1, time_conv_strides))
            self.ln = nn.LayerNorm(axis=2)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, num_of_time_filters, T_{r-1})

        '''
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape
        # shape is (batch_size, T, T)

        spatial_ccn = self.conv_SAt(x)
        #print("spatial_ccn.shape")
        #print(spatial_ccn.shape)
        time = self.rong(spatial_ccn).reshape(num_of_vertices, batch_size, num_of_timesteps)
        time_lstm_output = self.huiqu(self.time_LSTM(time).expand_dims(-1).transpose((1, 3, 0, 2)))

        '''
        # N,batch_size,Tr-1,num
        lstminput = spatial_ccn.transpose((2, 0, 3, 1))
        # print("log：spatial_gcn形状",spatial_gcn.shape)
        lstm_outputs = []
        for i in range(num):
            # N,batch_size,Tr-1
            lstmx = lstminput[:, :, :, i]
            # gru_output = nd.zeros(shape=(n,b,t), ctx=gruinput.context)
            lstm_output = (self.time_LSTM(lstmx))
            # print("log：gru_output形状", gru_output.shape)

            # -1 维度扩展  卷积核个数
            lstm_outputs.append(lstm_output.expand_dims(-1))
        # 根据卷积核个数连接  N,batch_size,Tr-1，num_filter

        time_lstm_output = nd.concat(*lstm_outputs, dim=-1)  #
        time_lstm_output = time_lstm_output.transpose((1, 0, 3, 2))
        '''
        # residual shortcut
        x_residual = (self.residual_conv(x.transpose((0, 2, 1, 3)))
                      .transpose((0, 2, 1, 3)))

        return self.ln(nd.relu(x_residual + time_lstm_output.transpose((0, 2, 1, 3))))


class STCNN_submodule(nn.Block):
    '''
    a module in ASTGCN
    '''
    def __init__(self, num_for_prediction, backbones, **kwargs):
        '''
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        backbones: list(dict), list of backbones

        '''
        super(STCNN_submodule, self).__init__(**kwargs)

        self.blocks = nn.Sequential()
        print("backbones")
        print(backbones)
        for backbone in backbones:
            self.blocks.add(ASTGCN_block(backbone))

        with self.name_scope():
            # use convolution to generate the prediction
            # instead of using the fully connected layer
            self.final_conv = nn.Conv2D(
                channels=num_for_prediction,
                kernel_size=(1, backbones[-1]['num_of_time_filters']))
            self.W = self.params.get("W", allow_deferred_init=True)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray,
           shape is (batch_size, num_of_vertices,
                     num_of_features, num_of_timesteps)

        Returns
        ----------
        mx.ndarray, shape is (batch_size, num_of_vertices, num_for_prediction)

        '''
        x = self.blocks(x)
        module_output = (self.final_conv(x.transpose((0, 3, 1, 2)))
                         [:, :, :, -1].transpose((0, 2, 1)))
        _, num_of_vertices, num_for_prediction = module_output.shape
        self.W.shape = (num_of_vertices, num_for_prediction)
        self.W._finish_deferred_init()
        return module_output * self.W.data()


class NWAGRU(nn.Block):
    '''
    ASTGCN, 3 sub-modules, for hour, day, week respectively
    '''
    def __init__(self, num_for_prediction, all_backbones, **kwargs):
        '''
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        all_backbones: list[list],
                       3 backbones for "hour", "day", "week" submodules
        '''
        super(NWAGRU, self).__init__(**kwargs)
        if len(all_backbones) <= 0:
            raise ValueError("The length of all_backbones "
                             "must be greater than 0")

        self.submodules = []
        with self.name_scope():
            for backbones in all_backbones:
                self.submodules.append(
                    STCNN_submodule(num_for_prediction, backbones))
                self.register_child(self.submodules[-1])

    def forward(self, x_list):
        '''
        Parameters
        ----------
        x_list: list[mx.ndarray],
                shape is (batch_size, num_of_vertices,
                          num_of_features, num_of_timesteps)

        Returns
        ----------
        Y_hat: mx.ndarray,
               shape is (batch_size, num_of_vertices, num_for_prediction)

        '''
        if len(x_list) != len(self.submodules):
            raise ValueError("num of submodule not equals to "
                             "length of the input list")

        num_of_vertices_set = {i.shape[1] for i in x_list}
        if len(num_of_vertices_set) != 1:
            raise ValueError("Different num_of_vertices detected! "
                             "Check if your input data have same size"
                             "at axis 1.")

        batch_size_set = {i.shape[0] for i in x_list}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have same batch size!")

        submodule_outputs = [self.submodules[idx](x_list[idx])
                             for idx in range(len(x_list))]

        return nd.add_n(*submodule_outputs)
