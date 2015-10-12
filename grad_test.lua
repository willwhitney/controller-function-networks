require 'optim'
require 'OneHot'
require 'vis'

torch.manualSeed(1)

opt = {
    input_dimension = 10,
    batch_size = 1,
    num_functions = 5,
    controller_units_per_layer = 10,
    controller_num_layers = 10,
    controller_dropout = 0,
    steps_per_output = 1,
}

function finiteDiff(model, input, target)
    local epsilon = 1e-7
    local p, gp = model:getParameters()
    local p_backup = p:clone()
    local gp_backup = gp:clone()
    local fd_grad = torch.zeros(gp:size())

    for i = 1, p:size(1) do
        p:copy(p_backup)
        p[i] = p[i] + epsilon
        local output_negative = model:forward(input):clone()
        local loss_negative = criterion:forward(output_negative, target)

        p:copy(p_backup)
        p[i] = p[i] - epsilon
        local output_positive = model:forward(input):clone()
        local loss_positive = criterion:forward(output_positive, target)

        fd_grad[i] = (loss_positive - loss_negative) / (2 * epsilon)
    end

    p:copy(p_backup)
    gp:copy(gp_backup)
    return fd_grad:mul(-1)
end

function finiteDiffStep(model, input, target)
    local epsilon = 1e-5
    local p, gp = model:getParameters()
    local p_backup = p:clone()
    local gp_backup = gp:clone()
    local fd_grad = torch.zeros(gp:size())

    for i = 1, p:size(1) do
        p:copy(p_backup)
        model:reset()
        p[i] = p[i] + epsilon
        local output_negative = model:step(input):clone()
        local loss_negative = criterion:forward(output_negative, target)

        p:copy(p_backup)
        model:reset()
        p[i] = p[i] - epsilon
        local output_positive = model:step(input):clone()
        local loss_positive = criterion:forward(output_positive, target)

        fd_grad[i] = (loss_positive - loss_negative) / (2 * epsilon)
    end

    p:copy(p_backup)
    gp:copy(gp_backup)
    return fd_grad:mul(-1)
end

function finiteDiffMultistep(model, inputs, targets)
    local epsilon = 1e-5
    local p, gp = model:getParameters()
    local p_backup = p:clone()
    local gp_backup = gp:clone()
    local fd_grad = torch.zeros(gp:size())

    for i = 1, p:size(1) do
        p:copy(p_backup)
        model:reset()
        p[i] = p[i] + epsilon
        local output_negative = model:step(input):clone()
        local loss_negative = criterion:forward(output_negative, target)

        p:copy(p_backup)
        model:reset()
        p[i] = p[i] - epsilon
        local output_positive = model:step(input):clone()
        local loss_positive = criterion:forward(output_positive, target)

        fd_grad[i] = (loss_positive - loss_negative) / (2 * epsilon)
    end

    p:copy(p_backup)
    gp:copy(gp_backup)
    return fd_grad:mul(-1)
end


input = torch.rand(opt.batch_size, opt.input_dimension)
target = torch.zeros(opt.batch_size, opt.input_dimension)



mode = 'step'
if mode == 'step' then
    require 'CFNetwork'
    model = nn.CFNetwork({
        input_dimension = opt.input_dimension,
        num_functions = opt.num_functions,
        controller_units_per_layer = opt.controller_units_per_layer,
        controller_num_layers = opt.controller_num_layers,
        controller_dropout = opt.controller_dropout,
        steps_per_output = opt.steps_per_output,
    })
    p, gp = model:getParameters()

    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()
    model:reset()

    output = model:step(input):clone()
    loss = criterion:forward(output, target)
    model:backstep(input, criterion:backward(output, target))
    backprop_grad = gp:clone()

    model:reset()
    fd_grad = finiteDiffStep(model, input, target)

    -- print(vis.simplestr(backprop_grad))
    -- print(vis.simplestr(fd_grad))

    print("Average param size:", p:norm(1) / p:size(1))
    print("Average gradient error per param:", (backprop_grad - fd_grad):norm(1) / p:size(1))
    print("Gradient error:", (backprop_grad - fd_grad):norm())

elseif mode == 'multistep' then
    require 'CFNetwork_multistep'
    model = nn.CFNetwork({
        input_dimension = opt.input_dimension,
        num_functions = opt.num_functions,
        controller_units_per_layer = opt.controller_units_per_layer,
        controller_num_layers = opt.controller_num_layers,
        controller_dropout = opt.controller_dropout,
        steps_per_output = opt.steps_per_output,
    })
    p, gp = model:getParameters()

    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()
    model:reset()

    output = model:step(input):clone()
    loss = criterion:forward(output, target)
    model:backstep(input, criterion:backward(output, target))
    backprop_grad = gp:clone()

    model:reset()
    fd_grad = finiteDiffStep(model, input, target)

    -- print(vis.simplestr(backprop_grad))
    -- print(vis.simplestr(fd_grad))

    print("Average param size:", p:norm(1) / p:size(1))
    print("Average gradient error per param:", (backprop_grad - fd_grad):norm(1) / p:size(1))
    print("Gradient error:", (backprop_grad - fd_grad):norm())

else
    require 'IIDCFNetwork'
    model = nn.IIDCFNetwork({
        input_dimension = opt.input_dimension,
        num_functions = opt.num_functions,
        controller_units_per_layer = opt.controller_units_per_layer,
        controller_num_layers = opt.controller_num_layers,
        controller_dropout = opt.controller_dropout,
        steps_per_output = opt.steps_per_output,
    })
    p, gp = model:getParameters()

    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()
    model:reset()

    output = model:forward(input):clone()
    loss = criterion:forward(output, target)
    model:backward(input, criterion:backward(output, target))
    backprop_grad = gp:clone()

    model:reset()
    fd_grad = finiteDiff(model, input, target)

    -- print(vis.simplestr(backprop_grad))
    -- print(vis.simplestr(fd_grad))

    print("Average gradient size:", gp:norm(1) / gp:size(1))
    print("Average gradient error per param:", (backprop_grad - fd_grad):abs():sum() / p:size(1))
    print("Gradient error:", (backprop_grad - fd_grad):norm())
end
