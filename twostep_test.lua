require 'nn'
require 'gnuplot'
require 'optim'
require 'nngraph'
require 'distributions'

require 'vis'

require 'IIDCF_meta'
require 'utils'
require 'OneHot'
require 'ExpectationCriterion'
require 'ProperJacobian'

-- parameters
local precision = 1e-5
local jac = nn.ProperJacobian
iteration = 100000

opt = {}
opt.sharpening_rate = 0
opt.batch_size = 1
iteration = 1


-- define inputs and module
local input = torch.rand(1,26)

local network = nn.Sequential()

local par = nn.ConcatTable()

primitivePipe = nn.Sequential()
primitivePipe:add(nn.Narrow(2, 1, 16))
primitivePipe:add(nn.Reshape(2, 8, false))
par:add(primitivePipe)

par:add(nn.Narrow(2, 17, 10))
network:add(par)

print(network:forward(input))


local module = nn.IIDCFNetwork({
        input_dimension = 8 + 10,
        encoded_dimension = 10,
        num_functions = 8,
        controller_units_per_layer = 10,
        controller_num_layers = 1,
        controller_dropout = 0,
        steps_per_output = 2,
        controller_nonlinearity = 'softmax',
        function_nonlinearity = 'prelu',
        controller_type = 'scheduled_sharpening',
        controller_noise = 0,
    })
network:add(module)

print(network)
test_input = {torch.rand(2,8), torch.rand(1,10)}
module:forward(test_input)
print(module:backward(test_input, torch.rand(1,10)))
-- print(network:forward(input))


-- test backprop, with Jacobian
local err = jac.testJacobian(network, input)
print('==> error: ' .. err)
if err<precision then
    print('==> module OK')
else
    print('==> error too large, incorrect implementation')
end
