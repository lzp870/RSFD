import torch
import torch.nn as nn
from config import Constants
import math

class EmptyObject(object):
    def __init__(self):
        pass

def dict2obj(dict):
    obj = EmptyObject()
    obj.__dict__.update(dict)
    return obj

class attentional_pooling(nn.Module):
    def __init__(self, config):
        super(attentional_pooling, self).__init__()
        if isinstance(config, dict):
            config = dict2obj(config)

        self.num_attention_heads = config.num_attention_heads  # 8个头
        self.attention_head_size = int(config.dim_hidden / config.num_attention_heads)  # 512 / 8 = 64 每个头的维度是64维
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 64 * 8 = 512 所有头拼接后的维度是512维

        # self.one_embeded_vec = torch.LongTensor([0]).cuda()    #[0]
        # self.n_embeded_vec = torch.arange(1,17,1).cuda()       #[1,2,...,15,16]
        self.one_embeded_vec = nn.Parameter(torch.rand([1, config.dim_hidden]),requires_grad=True).cuda()
        self.n_embeded_vec = nn.Parameter(torch.rand([16, config.dim_hidden]),requires_grad=True).cuda()

        # self.embedding = nn.Embedding(17, config.dim_hidden)  # 17, 512, padding_idx=0

        self.query = nn.Linear(config.dim_hidden,self.all_head_size)  # Q: Linear(in_features=512, out_features=512, bias=True)
        self.key = nn.Linear(config.dim_hidden,self.all_head_size)  # K: Linear(in_features=512, out_features=512, bias=True)
        self.value = nn.Linear(config.dim_hidden,self.all_head_size)  # V: Linear(in_features=512, out_features=512, bias=True)
        self.dropout = nn.Dropout(0.5)  # Dropout(p=0.0, inplace=False)

        self.LayerNorm = torch.nn.LayerNorm(config.dim_hidden, eps=config.layer_norm_eps)  # LayerNorm((512,), eps=1e-05, elementwise_affine=True)

    def forward(self,enc_output):   #[64,16,512]
        # one_embeded_vec = self.embedding(self.one_embeded_vec).unsqueeze(0).repeat(enc_output.size(0),1,1)  #[1,512] --> [1,1,512] --> [64,1,512]
        # n_embeded_vec = self.embedding(self.n_embeded_vec).unsqueeze(0).repeat(enc_output.size(0),1,1)    #[16,512] --> [1,16,512] --> [64,16,512]
        one_embeded_vec = self.one_embeded_vec.unsqueeze(0).repeat(enc_output.size(0),1,1)  #[1,512] --> [1,1,512] --> [64,1,512]
        n_embeded_vec = self.n_embeded_vec.unsqueeze(0).repeat(enc_output.size(0),1,1)      #[16,512] --> [1,16,512] --> [64,16,512]
        # print(self.n_embeded_vec)
        result_one = self.multi_head_attention(one_embeded_vec,enc_output,enc_output)
        result_n = self.multi_head_attention(n_embeded_vec,enc_output,enc_output)

        return result_one,result_n

    def multi_head_attention(self,q,k,v):
        d_k, d_v, n_head = self.attention_head_size, self.attention_head_size, self.num_attention_heads  # 64 64 8

        sz_b, len_q, _ = q.size()  # [64,1,512]/[64,16,512]
        sz_b, len_k, _ = k.size()  # [64,16,512]
        sz_b, len_v, _ = v.size()  # [64,16,512]

        q = self.query(q).view(sz_b, len_q, n_head, d_k)    # [64,1,512] --> [64,1,512] --> [64,1,8,64]/[64,16,512] --> [64,16,512] --> [64,16,8,64]
        k = self.key(k).view(sz_b, len_k, n_head, d_k)      # [64,16,512] --> [64,16,512] --> [64,16,8,64]
        v = self.value(v).view(sz_b, len_v, n_head, d_v)    # [64,16,512] --> [64,16,512] --> [64,16,8,64]

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # [64,1,8,64] --> [8*64,1,64]/[64,16,8,64] --> [8*64,16,64]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # [64,16,8,64] --> [8*64,16,64]
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # [64,16,8,64] --> [8*64,16,64]

        attention_scores = torch.bmm(q, k.transpose(1, 2))  # [64*8,1,16]/[64*8,16,16]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        outputs = torch.bmm(attention_probs, v)

        outputs = outputs.view(n_head, sz_b, len_q, d_v)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)  [64,1,512]/[64,16,512]

        outputs = self.dropout(self.LayerNorm(outputs))

        return outputs
