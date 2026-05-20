import torch
import random

class SAM(torch.optim.Optimizer):
    def __init__(self, params,  base_optimizer,named_params, rho=0.05, peft_weight="all",random_weight=False, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.param_names = {p: name for name, p in named_params}  # 关键修改
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.peft_weight=peft_weight
        self.random_weight=random_weight
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        # 获取参数名称映射（假设通过named_parameters()初始化）
        #param_names = {p: name for name, p in self.model.named_parameters()}  # 需要连接模型参数
        # 获取所有参数及其梯度范数
        param_grad_norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    name = self.param_names.get(p, "")
                    if self.peft_weight == "all":
                        grad_norm_p = p.grad.norm()  # 计算当前参数的梯度范数
                        # 精确匹配q_proj参数（避免匹配到其他含q的层）
                        param_grad_norms.append((p, grad_norm_p.item()))
                        #print("all")
                    elif self.peft_weight == "q" and '.q_proj.' in name:
                        grad_norm_p = p.grad.norm()  # 计算当前参数的梯度范数
                        param_grad_norms.append((p, grad_norm_p.item()))
                        #print("loraq")
                    elif self.peft_weight == "v" and '.v_proj.' in name:  
                        grad_norm_p = p.grad.norm()  # 计算当前参数的梯度范数
                        param_grad_norms.append((p, grad_norm_p.item()))
                        #print("lorav")
                    elif self.peft_weight == "v" and 'prefix_values' in name: 
                        grad_norm_p = p.grad.norm()  # 计算当前参数的梯度范数
                        param_grad_norms.append((p, grad_norm_p.item())) 
                        #print("prefixv")
                    if self.peft_weight == "k" and 'prefix_keys' in name:     
                        grad_norm_p = p.grad.norm()  # 计算当前参数的梯度范数
                        param_grad_norms.append((p, grad_norm_p.item())) 
                        #print("prefixk")                    
                        #print("enter")
                        #grad_norm_p = p.grad.norm()  # 计算当前参数的梯度范数
                        #param_grad_norms.append((p, grad_norm_p.item()))
                    #print("grad_norm_p:")
                    #print(grad_norm_p.item())
        
        if self.random_weight==False:    
            param_grad_norms.sort(key=lambda x: x[1], reverse=True)  # 按照梯度范数降序排序
            num_sensitive_params = int(len(param_grad_norms) * 0.05)  # 选择 5% 的参数
            sensitive_params = param_grad_norms[:num_sensitive_params]
            #print("randomF")
        elif self.random_weight==True:
        # 随机选择5%的参数（至少选1个）
            num_sensitive_params = max(1, int(len(param_grad_norms) * 0.05))
            sensitive_params = random.sample(param_grad_norms, num_sensitive_params)
            #print("randomT")
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                #if p.requires_grad == False: continue
                
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                #if (p, p.grad.norm().item()) in sensitive_params:
                grad_norm_p = p.grad.norm().item()
                if grad_norm_p in sensitive_params:
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"
                    
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
