from icl.analysis.activation_analysis import start_save, end_save, get_result, add_activation, add_activation_grad

import torch
log_dir = './tmp_log'
start_save(log_dir, save_activation=True, save_activation_grad=True, debug=True, cover = True)
a = torch.randn(2, 3, 4, 5)
c = a*2
c = add_activation(c, 'c')
c = add_activation_grad(c, 'c')
e = c.sum()
e.backward() # 必须手动加上backward（但是requires_grad之类的会自动处理），否则grad不会被记录
end_save()
print('here')
print(get_result(log_dir, 'c'))
print(get_result(log_dir, 'c_grad'))
