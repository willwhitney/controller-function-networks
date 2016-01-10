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
    -- num_functions = 4,
    -- rnn_size = 5,
    -- controller_num_layers = 2,
    -- controller_dropout = 0,
    -- steps_per_output = 1,
    -- controller_nonlinearity = 'sigmoid',
    -- function_nonlinearity = 'sigmoid',
}

function finiteDiff(model, input, target, p, gp)
    local epsilon = 1e-6
    local p_backup = p:clone()
    local gp_backup = gp:clone()
    local fd_grad = torch.zeros(gp:size())

    for i = 1, p:size(1) do
        p:copy(p_backup)
        p[i] = p[i] - epsilon

        local loss_negative = 0
        local output_negative = model:forward(input):clone()
        loss_negative = criterion:forward(output_negative, target)

        p:copy(p_backup)
        p[i] = p[i] + epsilon

        local loss_positive = 0
        local output_positive = model:forward(input):clone()
        loss_positive = criterion:forward(output_positive, target)

        fd_grad[i] = (loss_positive - loss_negative) / (2 * epsilon)
    end

    p:copy(p_backup)
    gp:copy(gp_backup)
    return fd_grad
end

function finiteDiffSampling(model, input, target, p, gp)
    local epsilon = 1e-6
    local p_backup = p:clone()
    local gp_backup = gp:clone()
    local fd_grad = torch.zeros(gp:size())

    for i = 1, p:size(1) do
        p:copy(p_backup)
        p[i] = p[i] - epsilon
        local loss_negative = model:forward(input, target)

        p:copy(p_backup)
        p[i] = p[i] + epsilon
        local loss_positive = model:forward(input, target)

        fd_grad[i] = (loss_positive - loss_negative) / (2 * epsilon)
    end

    p:copy(p_backup)
    gp:copy(gp_backup)
    return fd_grad
end

test = "sampling"
if test == "sampling" then
    require 'SamplingIID'
    model = nn.SamplingCFNetwork({
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

    p, gp = model:getParameters()
    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()

    inputs = torch.rand(opt.batch_size, 10)
    inputs = {}
    inputs[1] = torch.rand(opt.batch_size, 8)
    inputs[2] = torch.rand(opt.batch_size, 10)
    target = torch.zeros(opt.batch_size, 10)

    model:forward(inputs, target)
    model:backward(inputs, target)
    backprop_grad = gp:clone()

    fd_grad = finiteDiffSampling(model, inputs, target, p, gp)
else
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

    p, gp = model:getParameters()
    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()

    inputs = torch.rand(opt.batch_size, 10)
    inputs = {}
    inputs[1] = torch.rand(opt.batch_size, 8)
    inputs[2] = torch.rand(opt.batch_size, 10)
    target = torch.zeros(opt.batch_size, 10)

    output = model:forward(inputs)
    loss = criterion:forward(output, target)
    grad_output = criterion:backward(output, target)


    model:backward(inputs, grad_output)
    backprop_grad = gp:clone()

    fd_grad = finiteDiff(model, inputs, target, p, gp)
end

print("Gradient error:", (backprop_grad - fd_grad):norm())
print("Average true gradient size:", fd_grad:norm(1) / fd_grad:size(1))
print("Average backprop gradient size:", backprop_grad:norm(1) / backprop_grad:size(1))
-- print("Average gradient error per param:", (backprop_grad - fd_grad):norm(1) / p:size(1))
print("Average gradient percent error:", (backprop_grad - fd_grad):norm(1) / fd_grad:norm(1) * 100)
-- print(vis.simplestr(backprop_grad))
-- print(vis.simplestr(fd_grad))
