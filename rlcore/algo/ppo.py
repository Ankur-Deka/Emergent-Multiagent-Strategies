import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import time

## this is not being used, check JointPPO below
class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False):

        self.actor_critic = actor_critic    ## it is MPNN instance

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    def update(self, rollouts, time_mask):      ## time_mask used to mask out unnecessary time steps
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(obs_batch,
                                             recurrent_hidden_states_batch, masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped=value_preds_batch+(values-value_preds_batch).clamp(-self.clip_param,self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = .5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(return_batch, values)
                
                self.optimizer.zero_grad()                
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)    ## gradient clipping
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


class JointPPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    def update(self, rollouts_list, opp_rollouts_list):
        # rollouts_list - list of rollouts of agents which share self.actor_critic policy
        # print('rollouts_list', rollouts_list, 'ppo.py')
        # print('opp_rollouts_list', opp_rollouts_list, 'ppo.py')
        advantages_list = []
        for rollout in rollouts_list:
            advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            advantages_list.append(advantages)
        # print('advantages_list', advantages_list, 'ppo.py')
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                raise NotImplementedError('sampler not implemented for recurrent policies')
            else:
                # print('magent_feed_forward_generator')
                data_generator = magent_feed_forward_generator(rollouts_list, opp_rollouts_list, advantages_list, self.num_mini_batch)
            
            
            for sample in data_generator:
                obs_batch, mask, obs_opp_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample   ## I've added the mask and mask_opp to handle dead agents
                # print(obs_batch)    # dims are [time_steps*num agents x obs dim]. Each row first row is obs for first agent at time t, second row is for first agent at t+1, .... then second agent at t, t+1.... and so on
                
                # print('mask')
                # print(mask)
                # print('PPO.py')
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(obs_batch,
                                 recurrent_hidden_states_batch, obs_opp_batch, masks_batch, actions_batch)
                

                dist_entropy = dist_entropy*mask[:,0]
                dist_entropy = dist_entropy.mean() 
                if mask.mean() != 0:
                    dist_entropy = dist_entropy/mask.mean()

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                # print('ratio')
                # print(ratio)
                ratio = mask*ratio
                # print('ratio masked')
                # print(ratio)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                # print('surr1')
                # print(surr1)
                # print('surr2')
                # print(surr2)
                action_loss = -torch.min(surr1, surr2)
                action_loss = mask*action_loss ## wherever mask[i] = 0, gradient cannot flow back through action_loss[i] there
                action_loss = action_loss.mean()
                if mask.mean() != 0:
                    action_loss = action_loss/mask.mean()
                if self.use_clipped_value_loss:
                    value_pred_clipped=value_preds_batch+(values-value_preds_batch).clamp(-self.clip_param,self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = .5 * torch.max(value_losses, value_losses_clipped)
                    value_loss = value_loss*mask
                    value_loss = value_loss.mean()
                    if mask.mean() != 0:
                        value_loss = value_loss/mask.mean()
                
                else:
                    value_loss = 0.5 * F.mse_loss(return_batch, values)
                    value_loss = value_loss*mask
                    value_loss = value_loss.mean()
                    if mask.mean() != 0:
                        value_loss = value_loss/mask.mean()
                
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch        


def magent_feed_forward_generator(rollouts_list, opp_rollouts_list, advantages_list, num_mini_batch):
    num_steps, num_processes = rollouts_list[0].rewards.size()[0:2]
    batch_size = num_processes * num_steps              ## total samples
    mini_batch_size = int((batch_size/num_mini_batch))  ## size of minibatch for each agent
    # print('batch_size', batch_size, 'mini_batch_size', mini_batch_size)
    # time.sleep(2)
    sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)    ## selects set of random indices
    # print('rollouts_list', len(rollouts_list))
    # print('opp_rollouts_list', len(opp_rollouts_list))

    for i, indices in enumerate(sampler):
        # print('indices')
        # print(indices)
        # print('Thing 1')
        # print([rollout.obs[:-1].view(-1,*rollout.obs.size()[2:])[indices] for rollout in rollouts_list])
        obs_batch = torch.cat([rollout.obs[:-1].view(-1,*rollout.obs.size()[2:])[indices] for rollout in rollouts_list],0) ## obs_batch: each row corresponds to one agent's observation at one time step. first mini_batch_size rows are for first agent at different times steps, next mini_batch_size rows are for second agent at the same set of time steps, ....

        mask = obs_batch[:,0].clone().view(-1,1)
        obs_opp_batch = torch.cat([rollout.obs[:-1].view(-1,*rollout.obs.size()[2:])[indices] for rollout in opp_rollouts_list],0)
        recurrent_hidden_states_batch = torch.cat([rollout.recurrent_hidden_states[:-1].view(-1, 
                    rollout.recurrent_hidden_states.size(-1))[indices] for rollout in rollouts_list],0)
        actions_batch = torch.cat([rollout.actions.view(-1,
                    rollout.actions.size(-1))[indices] for rollout in rollouts_list],0)
        value_preds_batch=torch.cat([rollout.value_preds[:-1].view(-1, 1)[indices] for rollout in rollouts_list],0)
        return_batch = torch.cat([rollout.returns[:-1].view(-1, 1)[indices] for rollout in rollouts_list],0)
        masks_batch = torch.cat([rollout.masks[:-1].view(-1, 1)[indices] for rollout in rollouts_list],0)
        old_action_log_probs_batch=torch.cat([rollout.action_log_probs.view(-1,1)[indices] for rollout in rollouts_list],0)
        adv_targ = torch.cat([advantages.view(-1, 1)[indices] for advantages in advantages_list],0)
        # print('i', i)
        # print('obs_batch')
        # print(obs_batch)
        # print('obs_opp_batch')
        # print(obs_opp_batch)
        # print('actions_batch')
        # print(actions_batch)
        # print('masks_batch')
        # print(masks_batch)

        yield obs_batch, mask, obs_opp_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch,\
              masks_batch, old_action_log_probs_batch, adv_targ