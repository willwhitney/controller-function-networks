require 'nn'
require 'gnuplot'
require 'optim'
require 'nngraph'
require 'distributions'

require 'vis'

require 'utils'
require 'OneHot'
require 'ExpectationCriterion'
require 'ProperJacobian'

require 'FF_IIDCF_meta'
-- require 'IIDCF_meta'

torch.manualSeed(1)

-- parameters
local precision = 1e-5
local jac = nn.ProperJacobian
iteration = 100000

opt = {}
opt.sharpening_rate = 0
opt.batch_size = 1
iteration = 1

primitives = 8
timesteps = 10
vector_size = 10

-- define inputs and module
local input = torch.rand(1, primitives * timesteps + vector_size)

local network = nn.Sequential()

local par = nn.ConcatTable()

primitivePipe = nn.Sequential()
primitivePipe:add(nn.Narrow(2, 1, primitives * timesteps))
primitivePipe:add(nn.Reshape(timesteps, primitives, false))
par:add(primitivePipe)

par:add(nn.Narrow(2, primitives * timesteps, vector_size))
network:add(par)


local module = nn.IIDCFNetwork({
        input_dimension = primitives + vector_size,
        encoded_dimension = vector_size,
        num_functions = primitives,
        controller_units_per_layer = vector_size,
        controller_num_layers = 1,
        controller_dropout = 0,
        steps_per_output = timesteps,
        controller_nonlinearity = 'softmax',
        function_nonlinearity = 'prelu',
        controller_type = 'scheduled_sharpening',
        controller_noise = 0,
    })
network:add(module)

-- print(network)
-- test_input = {torch.rand(2,8), torch.rand(1,10)}
-- module:forward(test_input)
-- print(module:backward(test_input, torch.rand(1,10)))
-- print(network:forward(input))


-- test backprop, with Jacobian
local err = jac.testJacobian(network, input)
print('==> error: ' .. err)
if err<precision then
    print('==> module OK')
else
    print('==> error too large, incorrect implementation')
end
