import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.NeuralNetwork import NeuralNetwork


class PGAN(NeuralNetwork):

    def __init__(self, config):
        super(PGAN, self).__init__()
        self.config = config
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        self.n_heads = config['n_heads']

        self.A_us = config['A_us']
        self.A_uu = config['A_uu']
        embeding_size = config['embeding_size']

        self.word_embedding = nn.Embedding(V, D, padding_idx=0, _weight=torch.from_numpy(embedding_weights))
        self.user_embedding = nn.Embedding(config['A_us'].shape[0], embeding_size, padding_idx=0)
        self.source_embedding = nn.Embedding(config['A_us'].shape[1], embeding_size)

        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=config['maxlen'] - K + 1) for K in config['kernel_sizes']])

        self.Wcm = [nn.Parameter(torch.FloatTensor(embeding_size, embeding_size)).cuda() for _ in
                    range(self.n_heads)]
        self.Wam = [nn.Parameter(torch.FloatTensor(embeding_size, embeding_size)).cuda() for _ in
                    range(self.n_heads)]
        self.scale = torch.sqrt(torch.FloatTensor([embeding_size])).cuda() #  // self.n_heads

        self.W1 = nn.Parameter(torch.FloatTensor(embeding_size * self.n_heads, embeding_size))
        self.W2 = nn.Parameter(torch.FloatTensor(embeding_size * self.n_heads, embeding_size))
        self.linear = nn.Linear(400, 200)

        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

        self.fc_out = nn.Sequential(
            nn.Linear(300 + 2 * embeding_size, 100),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(100, config["num_classes"])
        )
        self.fc_user_out = nn.Sequential(
            nn.Linear(embeding_size, 100),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(100, 3)
        )
        self.fc_ruser_out = nn.Sequential(
            nn.Linear(embeding_size, 100),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(100, 3)
        )
        print(self)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.user_embedding.weight)
        init.xavier_normal_(self.source_embedding.weight)
        for i in range(self.n_heads):
            init.xavier_normal_(self.Wcm[i])
            init.xavier_normal_(self.Wam[i])

        init.xavier_normal_(self.W1)
        init.xavier_normal_(self.W2)
        init.xavier_normal_(self.linear.weight)
        for name, param in self.fc_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)
        for name, param in self.fc_user_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)
        for name, param in self.fc_ruser_out.named_parameters():
            if name.__contains__("weight"):
                init.xavier_normal_(param)

    def user_multi_head(self, X_user, X_user_id, Wcm):
        M = self.source_embedding.weight
        linear1 = torch.einsum("bd,dd,sd->bs", X_user, Wcm, M) / self.scale
        linear1 = self.relu(linear1) # (batch, |S|)

        A_us = self.A_us[X_user_id.cpu(), :].todense()
        A_us = torch.FloatTensor(A_us).cuda()   # (batch, |S|)

        alpha = F.softmax(linear1 * A_us, dim=-1)
        alpha = self.dropout(alpha)
        return alpha.matmul(M)

    def retweet_user_multi_head(self, X_ruser, X_ruser_id, Wam):
        M = self.user_embedding.weight
        linear1 = torch.einsum("bnd,dd,md->bnm", X_ruser, Wam, M) / self.scale # m x bsz
        linear1 = self.relu(linear1)

        s1, s2 = X_ruser_id.size()
        idx = X_ruser_id.view(-1).cpu()
        A_uu = self.A_uu[idx, :].todense()
        A_uu = torch.FloatTensor(A_uu).view(s1, s2, -1).cuda()

        alpha = F.softmax(linear1 * A_uu, dim=-1)  # m x bsz
        alpha = self.dropout(alpha)
        return alpha.matmul(M)

    def publisher_encoder(self, X_user, X_user_id):
        m_hat = []
        for i in range(self.n_heads):
            m_hat.append(self.user_multi_head(X_user, X_user_id, self.Wcm[i]))

        m_hat = torch.cat(m_hat, dim=-1).matmul(self.W1)
        m_hat = self.elu(m_hat)
        m_hat = self.dropout(m_hat)

        U_hat = m_hat + X_user  # bsz x d
        return U_hat

    def retweet_user_encoder(self, X_ruser, X_ruser_id):  # 0.854  0.878
        '''
        :param X_ruser:  (bsz, num_users, d)
        :param X_ruser_id: (bsz, num_users)
        :return:
        '''
        m_hat = []
        for i in range(self.n_heads):
            m_hat.append(self.retweet_user_multi_head(X_ruser, X_ruser_id, self.Wam[i]))
        m_hat = torch.cat(m_hat, dim=-1).matmul(self.W2)
        m_hat = self.elu(m_hat)
        m_hat = self.dropout(m_hat)

        a_hat = m_hat + X_ruser  # bsz x 20 x d
        return a_hat

    def source_encoder(self, X_source, r_user_rep, user_rep):  #
        linear1 = torch.einsum("bd,bnd->bn", X_source, r_user_rep) # / self.scale
        alpha = F.softmax(linear1, dim=-1)
        retweet_rep = torch.einsum("bn,bnd->bd", alpha, r_user_rep)

        # beta = 0.5
        source_rep = torch.cat([retweet_rep, user_rep,
                                retweet_rep * user_rep,
                                retweet_rep - user_rep], dim=-1)  # .mm(self.W) #
        source_rep = self.linear(source_rep)
        source_rep = self.dropout(source_rep)
        return source_rep

    def text_representation(self, X_word):
        X_word = X_word.permute(0, 2, 1)
        conv_block = []
        for Conv, max_pooling in zip(self.convs, self.max_poolings):
            act = self.relu(Conv(X_word))
            pool = max_pooling(act).squeeze()
            conv_block.append(pool)

        features = torch.cat(conv_block, dim=1)
        features = self.dropout(features)
        return features

    def forward(self, X_source_wid, X_source_id, X_user_id, X_ruser_id):  # , X_composer_id, X_reviewer_id
        '''
        :param X_source_wid size: (batch_size, max_words)
                X_source_id size: (batch_size, )
                X_user_id  size: (batch_size, )
                X_retweet_id  size: (batch_size, max_retweets)
                X_retweet_uid  size: (batch_size, max_retweets)

        :return:
        '''
        X_word = self.word_embedding(X_source_wid)
        X_user = self.user_embedding(X_user_id)
        X_ruser = self.user_embedding(X_ruser_id)
        X_source = self.source_embedding(X_source_id)

        X_text = self.text_representation(X_word)

        user_rep = self.publisher_encoder(X_user, X_user_id)
        r_user_rep = self.retweet_user_encoder(X_ruser, X_ruser_id)  #
        source_rep = self.source_encoder(X_source, r_user_rep, user_rep)  #

        tweet_rep = torch.cat([X_text, source_rep], dim=-1)

        Xt_logit = self.fc_out(tweet_rep)
        Xu_logit = self.fc_user_out(user_rep)
        Xru_logit = self.fc_ruser_out(r_user_rep)

        return Xt_logit, Xu_logit, Xru_logit

