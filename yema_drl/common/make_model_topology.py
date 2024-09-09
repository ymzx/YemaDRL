import torch
from ppo_agent import ActorGaussian  # 假设模型类在这个文件中定义
from types import SimpleNamespace

# 1. 重新创建模型实例
args = {'max_action': 2, 'hidden_width': 64, 'state_dim': 3, 'action_dim': 1, 'use_tanh': True, 'use_orthogonal_init': False}
model = ActorGaussian(args=SimpleNamespace(**args))

# 2. 加载模型参数
state_dict = torch.load(r'D:\project\YemaDRL\yema_drl\ppo\model\Pendulum-v1_actor_35.pth')
model.load_state_dict(state_dict)

# 3. 创建一个输入张量
state = torch.randn(1, args['state_dim'])  # 输入尺寸根据模型要求调整

# 4. 获取模型的输出
output = model(state)

# 5. 可视化模型的计算图
from torchviz import make_dot
dot = make_dot(output, params=dict(model.named_parameters()))

# 6. 保存为PDF或PNG
dot.format = 'png'  # 或者 'png'
dot.render("model_topology")

