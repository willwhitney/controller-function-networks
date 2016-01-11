require 'nn'
require 'ScheduledWeightSharpener'

-- parameters
local precision = 1e-5
local jac = nn.Jacobian
iteration = 100000

-- define inputs and module
local input = torch.rand(1,10)
local module = nn.ScheduledWeightSharpener()

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
    print('==> module OK')
else
    print('==> error too large, incorrect implementation')
end
