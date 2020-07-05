import torch
import torch.nn as nn
import numpy as np
from rlcore.distributions import Categorical
import torch.nn.functional as F
import math
import time

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class MPNN(nn.Module):
    def __init__(self, action_space, num_agents, num_opp_agents, num_entities, input_size=16, hidden_dim=128, embed_dim=None,
                 pos_index=2, norm_in=False, nonlin=nn.ReLU, n_heads=1, mask_dist=None, entity_mp=False, policy_layers=1):
        super().__init__()

        self.h_dim = hidden_dim
        self.nonlin = nonlin
        self.num_agents = num_agents # number of agents
        self.num_opp_agents = num_opp_agents
        self.num_entities = num_entities # number of entities
        self.K = 3 # message passing rounds
        self.embed_dim = self.h_dim if embed_dim is None else embed_dim
        self.n_heads = n_heads
        self.mask_dist = mask_dist
        self.input_size = input_size
        self.entity_mp = entity_mp
        self.policy_layers = policy_layers
        # this index must be from the beginning of observation vector
        self.pos_index = pos_index

        self.encoder = nn.Sequential(nn.Linear(self.input_size,int(self.h_dim/2)),
                                     self.nonlin(inplace = True))

        self.oppEncoder = nn.Sequential(nn.Linear(self.input_size,int(self.h_dim/2)),
                                     self.nonlin(inplace = True)) ## encode the states of opponents

        self.oppAttn = MultiHeadOppAttention(n_heads=self.n_heads, input_dim=int(self.h_dim/2), opp_input_dim=int(self.h_dim/2), embed_dim=int(self.embed_dim/2))
        self.oppUpdate = nn.Sequential(nn.Linear(int(self.h_dim/2)+int(self.embed_dim/2),int(self.h_dim/2)),
                                    self.nonlin(inplace = True))

        self.messages = MultiHeadAttention(n_heads=self.n_heads,input_dim=self.h_dim,embed_dim=self.embed_dim)

        self.update = nn.Sequential(nn.Linear(self.h_dim+self.embed_dim,self.h_dim),
                                    self.nonlin(inplace = True))

        self.value_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                        self.nonlin(inplace = True),
                                        nn.Linear(self.h_dim,1))

        if self.policy_layers==1:
            self.policy_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                         self.nonlin(inplace = True))
        elif self.policy_layers==2:
            self.policy_head = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                        self.nonlin(inplace = True),
                                        nn.Linear(self.h_dim,self.h_dim))            

        if self.entity_mp:
            self.entity_encoder = nn.Sequential(nn.Linear(2,self.h_dim),
                                                self.nonlin(inplace = True))
            
            self.entity_messages = MultiHeadAttention(n_heads=1,input_dim=self.h_dim,embed_dim=self.embed_dim)
            
            self.entity_update = nn.Sequential(nn.Linear(self.h_dim+self.embed_dim,self.h_dim),
                                               self.nonlin())

        num_actions = action_space.shape[0] ##
        self.dist = Categorical(self.h_dim,num_actions)

        self.is_recurrent = False

        if norm_in:
            self.in_fn = nn.BatchNorm1d(self.input_size)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.apply(weights_init)

        self.attn_mat = np.ones((num_agents, num_agents))
        # self.opp_attn_mat = np.ones(())
        self.dropout_mask = None
        self.dead_mask = None

    def calculate_mask(self, inp):
        # inp is batch_size x self.input_size where batch_size is num_processes*num_agents
        
        pos = inp[:, self.pos_index:self.pos_index+2]
        bsz = inp.size(0)//self.num_agents
        mask = torch.full(size=(bsz,self.num_agents,self.num_agents),fill_value=0,dtype=torch.uint8)
        
        if self.mask_dist is not None and self.mask_dist > 0: 
            for i in range(1,self.num_agents):
                shifted = torch.roll(pos,-bsz*i,0)
                dists = torch.norm(pos-shifted,dim=1)
                restrict = dists > self.mask_dist
                for x in range(self.num_agents):
                    mask[:,x,(x+i)%self.num_agents].copy_(restrict[bsz*x:bsz*(x+1)])
        
        elif self.mask_dist is not None and self.mask_dist == -10:
           if self.dropout_mask is None or bsz!=self.dropout_mask.shape[0] or np.random.random_sample() < 0.1: # sample new dropout mask
               temp = torch.rand(mask.size()) > 0.85
               temp.diagonal(dim1=1,dim2=2).fill_(0)
               self.dropout_mask = (temp+temp.transpose(1,2))!=0
           mask.copy_(self.dropout_mask)

        return mask            


    # this is where attention is used for both self team and the opponents
    def _fwd(self, inp, oppInp, masks):
        # inp should be (batch_size,input_size)
        # inp - {iden, vel(2), pos(2), entities(...)}
        # agent_inp = inp[:,:self.input_size]
        # opp_inp = oppInp[:,:,self.input_size]          
        ## mask is alive/dead mask
        # mask = self.calculate_mask(agent_inp) # shape <batch_size/N,N,N> with 0 for comm allowed, 1 for restricted
        # print('inp and oppInp', inp.shape, oppInp.shape)
        # print('agent_inp', inp.shape, self.input_size)
        # print('communication mask', mask.shape, mask)
        h = self.encoder(inp) # should be (num_agents,self.h_dim)
        hOpp = self.oppEncoder(oppInp)  # (num_opp_agents,)
        # print('num_agents', self.num_agents)

        # reshape so that multiple time steps can be handled
        h = h.view(self.num_agents,-1,int(self.h_dim/2)).transpose(0,1)	# dimensions are (after the transpose) [time steps, num_agents, embedding dim]
        hOpp = hOpp.view(self.num_opp_agents, -1, int(self.h_dim/2)).transpose(0,1)


        # numAgents, numOpps = h.shape[0], hOpp.shape[0]  ## number of alive agents/opponents

        # print(h.shape, hOpp.shape)
        eOpp, opp_attn_mat = self.oppAttn(h, hOpp, masks, return_attn = True)   ## compute components from the opponents
        self.opp_attn_mat = opp_attn_mat.squeeze(0).squeeze(0).cpu().detach().numpy()
        # print('h', h.shape, 'eOpp', eOpp.shape)
        h = torch.cat((h, eOpp), dim = 2) 	# dims are [time steps, num agents, embed dim]
        # print('h concatenataed', h.shape)
        # hOpp = hOpp.view(numOpps,-1,int(self.h_dim/2)).transpose(0,1)
        
        # h = h.view(numAgents,-1,self.h_dim).transpose(0,1) # should be (batch_size/N,N,self.h_dim)
        
        # print('raw', inp.shape, oppInp.shape)
        # print('encoded', h.shape, hOpp.shape)
        # ## next we'll compute attention with the opponents
        # e = self.oppAttn(h, hOpp)
        # print(e.shape)

        # print('h and hOpp', h.size(), hOpp.size())

        ##  number of message passing rounds
        for k in range(self.K):
            m, attn = self.messages(h, mask=None, return_attn=True) # should be <batch_size/N,N,self.embed_dim>
            h = self.update(torch.cat((h,m),2)) # should be <batch_size/N,N,self.h_dim> ## update passes through a neural net
            # print('h', h, 'm', m, 'mpnn.py')
        h = h.transpose(0,1).contiguous().view(-1,self.h_dim)
        # print('own attn before squeezing')
        # print(attn.shape)
        # print('opp attn before squeezing')
        # print(opp_attn_mat.shape)
        self.attn_mat = attn.squeeze(0).squeeze(0).detach().cpu().numpy()
        # print('own attn')
        # print(self.attn_mat)
        # print('opponent attn')
        # print(self.opp_attn_mat)

        return h # should be <batch_size, self.h_dim> again

    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def _value(self, x):
        return self.value_head(x)

    def _policy(self, x):
        return self.policy_head(x)

    def act(self, inp, state, oppInp, mask=None, deterministic=False):
        x = self._fwd(inp, oppInp, mask)
        value = self._value(x)
        dist = self.dist(self._policy(x))
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action).view(-1,1)
        return value,action,action_log_probs,state

    def evaluate_actions(self, inp, state, oppInp, mask, action):
        x = self._fwd(inp, oppInp, mask)
        value = self._value(x)
        dist = self.dist(self._policy(x))
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()
        return value,action_log_probs,dist_entropy,state

    def get_value(self, inp, state, oppInp, mask):
        x = self._fwd(inp, oppInp, mask)
        value = self._value(x)
        return value


class MultiHeadAttention(nn.Module):
    # taken from https://github.com/wouterkool/attention-tsp/blob/master/graph_encoder.py
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None, return_attn=False):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """

        if h is None:
            h = q  # compute self-attention
        # print('h.size()', h.size())
        # h should be (batch_size, graph_size, input_dim)

        batch_size, graph_size, input_dim = h.size()
        
        # if only one agent, no need to do anything within the team
        if graph_size == 1:
        	out = torch.zeros((batch_size, graph_size, self.embed_dim))
        	attn = torch.zeros((1,batch_size, graph_size, graph_size))
        	if h.is_cuda:
        		out = out.cuda()
        		attn = attn.cuda()
        	if return_attn:
        		return out, attn
        	return out

        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)   ## almost like numpy matmul
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)
        # print('inside', Q.shape, K.shape, V.shape)
        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        # expComp = torch.exp(compatibility)
        for i in range(compatibility.shape[2]):
            compatibility[:,:,i,i] = -math.inf      # it's about paying attention to other teamates' messages, not own message
        # print('compatibility', compatibility.shape)
        # should only pay attention to alive team-mates
        


        # print('compatibility', compatibility)
        # print('compatibility', compatibility.shape)
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -math.inf

        # attn = torch.max()

        attn = F.softmax(compatibility, dim=-1)

        
        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)
        # print('original', 'heads', heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim).shape, 'W_out', self.W_out.view(-1, self.embed_dim).shape)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)
        # print('out', out.shape, 'attn', attn.shape, 'mpnn.opy') ## out torch.Size([num_heads = 1, num_agents = 3, embed_dim = 128]) attn torch.Size([num_heads = 1, batch_size = 1 or 4, num_agents = 3, num_agents = 3])
        if return_attn:
            return out, attn
        return out

class MultiHeadOppAttention(nn.Module):
    # taken from https://github.com/wouterkool/attention-tsp/blob/master/graph_encoder.py
    def __init__(
            self,
            n_heads,
            input_dim,
            opp_input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadOppAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.opp_input_dim = opp_input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h, hOpp, mask=None, return_attn=False):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """

        numSteps, numAgents, _= h.size()
        numSteps, numOpps, _ = hOpp.size()

        ## left here!!
        # batch_size, graph_size, input_dim = h.size()
        # n_query = q.size(1)
        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim, "Wrong embedding dimension of input"

       	# print('before vectorizing')
       	# print('h')
       	# print(h.shape)
       	# print('hOpp')
       	# print(hOpp.shape)
        hflat = h.contiguous().view(-1, self.input_dim)
        hOppFlat = hOpp.contiguous().view(-1, self.opp_input_dim)
        # qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        # shp = (self.n_heads, batch_size, graph_size, -1)
        # shp_q = (self.n_heads, batch_size, n_query, -1)
        # print('hflat', hflat.shape, 'hOppFlat', hOppFlat.shape, 'W_query', self.W_query.shape)
        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(hOppFlat, self.W_query).view(numSteps, numOpps, -1)   ## torch matmul is almost like numpy matmul
        # print('inside message passing', qflat.shape, self.W_query.shape, Q.shape)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        V = torch.matmul(hOppFlat, self.W_val).view(numSteps, numOpps, -1)
        K = torch.matmul(hflat, self.W_key).view(numSteps, numAgents, -1)

        # print('matrix shapes', Q.shape, V.shape, K.shape)
        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(K, Q.transpose(1,2))
        # print('compatibility', compatibility.shape)
        # print('compatibility', compatibility.shape)
        # # Optionally apply mask to prevent attention
        # if mask is not None:
        #     mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
        #     compatibility[mask] = -math.inf
        # print('opp compatibility', compatibility.shape)
        attn = F.softmax(compatibility, dim=-1)
        # print('inside opp attn computation')
        # print(attn.shape)
        # # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        # if mask is not None:
        #     attnc = attn.clone()
        #     attnc[mask] = 0
        #     attn = attnc

        heads = torch.matmul(attn, V)
        out = torch.mm(
            heads.contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)).view(numSteps, numAgents, self.embed_dim)
        # print('shapes')
        # print('out', out.shape, 'attn', attn.shape)
        # print('numSteps', numSteps, 'numAgents', numAgents, 'embed_dim', self.embed_dim)
        if return_attn:
            return out, attn
        return out
        