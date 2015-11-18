require 'nn'
require 'optim'
require 'OneHot'
require 'vis'

seed = torch.rand(1):mul(100000):round()[1]
-- seed = 51651
-- seed = 76829
torch.manualSeed(seed)

print("Using seed " .. seed)

opt = {
    -- input_dimension = 10,
    -- encoded_dimension = 5,
    -- num_primitives = 8,
    batch_size = 1,
    seq_length = 1,
    -- num_functions = 4,
    -- rnn_size = 5,
    -- controller_num_layers = 2,
    -- controller_dropout = 0,
    -- steps_per_output = 1,
    -- controller_nonlinearity = 'sigmoid',
    -- function_nonlinearity = 'sigmoid',
}

--
-- function finiteDiff(model, input, target, p, gp)
--     local epsilon = 1e-6
--     local p_backup = p:clone()
--     local gp_backup = gp:clone()
--     local fd_grad = torch.zeros(gp:size())
--
--     for i = 1, p:size(1) do
--         p:copy(p_backup)
--         model:reset()
--         p[i] = p[i] - epsilon
--
--         local loss_negative = 0
--         if type(input) == 'table' then
--             local output_negative = model:forward(input)
--             for step = 1, #inputs do
--                 loss_negative = loss_negative + criterion:forward(output_negative[step], targets[step])
--             end
--         else
--             local output_negative = model:forward(input):clone()
--             loss_negative = criterion:forward(output_negative, target)
--         end
--
--         p:copy(p_backup)
--         model:reset()
--         p[i] = p[i] + epsilon
--
--         local loss_positive = 0
--         if type(input) == 'table' then
--             local output_positive = model:forward(input)
--             for step = 1, #inputs do
--                 loss_positive = loss_positive + criterion:forward(output_positive[step], targets[step])
--             end
--         else
--             local output_positive = model:forward(input):clone()
--             loss_positive = criterion:forward(output_positive, target)
--         end
--
--         fd_grad[i] = (loss_positive - loss_negative) / (2 * epsilon)
--     end
--
--     p:copy(p_backup)
--     gp:copy(gp_backup)
--     return fd_grad
-- end
--
-- function finiteDiffLSTMInput(model, input, target)
--     local epsilon = 1e-6
--     -- local input_backup = input:clone()
--     local fd_grad_input = torch.zeros(input:size())
--
--     for i = 1, input:size(2) do
--         model:reset()
--         input_negative = input:clone()
--         input_negative[1][i] = input_negative[1][i] - epsilon
--
--         local output_negative = model:forward({input_negative})[1]:clone()
--         local loss_negative = criterion:forward(output_negative, target)
--
--         model:reset()
--         input_positive = input:clone()
--         input_positive[1][i] = input_positive[1][i] + epsilon
--
--         local output_positive = model:forward({input_positive})[1]:clone()
--         local loss_positive = criterion:forward(output_positive, target)
--
--         fd_grad_input[1][i] = (loss_positive - loss_negative) / (2 * epsilon)
--     end
--
--     -- input:copy(input_backup)
--     return fd_grad_input
-- end

function finiteDiff(model, input, target, p, gp)
    local epsilon = 1e-6
    local p_backup = p:clone()
    local gp_backup = gp:clone()
    local fd_grad = torch.zeros(gp:size())

    for i = 1, p:size(1) do
        p:copy(p_backup)
        -- model:reset()
        p[i] = p[i] - epsilon

        local loss_negative = 0
        -- if type(input) == 'table' then
        --     local output_negative = model:forward(input)
        --     for step = 1, #inputs do
        --         loss_negative = loss_negative + criterion:forward(output_negative[step], targets[step])
        --     end
        -- else
            local output_negative = model:forward(input):clone()
            loss_negative = criterion:forward(output_negative, target)
        -- end

        p:copy(p_backup)
        -- model:reset()
        p[i] = p[i] + epsilon

        local loss_positive = 0
        -- if type(input) == 'table' then
        --     local output_positive = model:forward(input)
        --     for step = 1, #inputs do
        --         loss_positive = loss_positive + criterion:forward(output_positive[step], targets[step])
        --     end
        -- else
            local output_positive = model:forward(input):clone()
            loss_positive = criterion:forward(output_positive, target)
        -- end

        fd_grad[i] = (loss_positive - loss_negative) / (2 * epsilon)
    end

    p:copy(p_backup)
    gp:copy(gp_backup)
    return fd_grad
end
-- function finiteDiff(model, input, target)
--     local epsilon = 1e-6
--     -- local input_backup = input:clone()
--     local fd_grad_input = {
--             torch.zeros(input[1]:size()),
--             torch.zeros(input[2]:size()),
--         }
--
--     for i = 1, input[1]:size(1) do
--         model:reset()
--         input_negative = {
--                 input[1]:clone(),
--                 input[2]:clone(),
--             }
--         input_negative[1][i] = input_negative[1][i] - epsilon
--
--         local output_negative = model:forward(input_negative):clone()
--         local loss_negative = criterion:forward(output_negative, target)
--
--         model:reset()
--         input_positive = {
--                 input[1]:clone(),
--                 input[2]:clone(),
--             }
--         input_positive[1][i] = input_positive[1][i] - epsilon
--
--         local output_positive = model:forward(input_positive):clone()
--         local loss_positive = criterion:forward(output_positive, target)
--
--         fd_grad_input[1][1][i] = (loss_positive - loss_negative) / (2 * epsilon)
--     end
--
--     for i = 1, input[2]:size(1) do
--         model:reset()
--         input_negative = {
--                 input[1]:clone(),
--                 input[2]:clone(),
--             }
--         input_negative[2][i] = input_negative[2][i] - epsilon
--
--         local output_negative = model:forward(input_negative):clone()
--         local loss_negative = criterion:forward(output_negative, target)
--
--         model:reset()
--         input_positive = {
--                 input[1]:clone(),
--                 input[2]:clone(),
--             }
--         input_positive[2][i] = input_positive[2][i] - epsilon
--
--         local output_positive = model:forward(input_positive):clone()
--         local loss_positive = criterion:forward(output_positive, target)
--
--         fd_grad_input[2][1][i] = (loss_positive - loss_negative) / (2 * epsilon)
--     end
--
--     -- input:copy(input_backup)
--     print(fd_grad_input)
--     return fd_grad_input
-- end


-- require 'NewIIDCF'
-- model = nn.IIDCFNetwork({
--     input_dimension = 10,
--     encoded_dimension = 10,
--     num_functions = 8,
--     controller_units_per_layer = 10,
--     controller_num_layers = 1,
--     controller_dropout = 0,
--     steps_per_output = 1,
--     controller_nonlinearity = 'sigmoid',
--     function_nonlinearity = 'sigmoid',
-- })
require 'IIDCF_meta'
model = nn.IIDCFNetwork({
    input_dimension = 18,
    encoded_dimension = 10,
    num_functions = 8,
    controller_units_per_layer = 10,
    controller_num_layers = 1,
    controller_dropout = 0,
    steps_per_output = 1,
    controller_nonlinearity = 'sigmoid',
    function_nonlinearity = 'sigmoid',
})

-- model = nn.Linear(10, 10)
p, gp = model:getParameters()
p_backup = p:clone()
gp_backup = gp:clone()

criterion = nn.MSECriterion()
model:evaluate()
-- model:reset()

inputs = torch.rand(opt.batch_size, 10)
inputs = {}
inputs[1] = torch.rand(opt.batch_size, 8)
inputs[2] = torch.rand(opt.batch_size, 10)
target = torch.zeros(opt.batch_size, 10)

-- print(inputs)
output = model:forward(inputs)
loss = criterion:forward(output, target)
grad_output = criterion:backward(output, target)


model:backward(inputs, grad_output)
backprop_grad = gp:clone()

-- model:reset()
fd_grad = finiteDiff(model, inputs, target, p, gp)

-- print(backprop_grad)
-- print(fd_grad)

print("Gradient error:", (backprop_grad - fd_grad):norm())
print("Average true gradient size:", fd_grad:norm(1) / fd_grad:size(1))
print("Average backprop gradient size:", backprop_grad:norm(1) / backprop_grad:size(1))
-- print("Average gradient error per param:", (backprop_grad - fd_grad):norm(1) / p:size(1))
print("Average gradient percent error:", (backprop_grad - fd_grad):norm(1) / fd_grad:norm(1) * 100)
-- print(vis.simplestr(backprop_grad))
-- print(vis.simplestr(fd_grad))
