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
    input_dimension = 10,
    encoded_dimension = 5,
    batch_size = 5,
    seq_length = 2,
    num_functions = 4,
    rnn_size = 5,
    controller_num_layers = 2,
    controller_dropout = 0,
    steps_per_output = 5,
    controller_nonlinearity = 'sigmoid',
    function_nonlinearity = 'sigmoid',
}
mode = 'multistep'


function finiteDiff(model, input, target, p, gp)
    local epsilon = 1e-6
    local p_backup = p:clone()
    local gp_backup = gp:clone()
    local fd_grad = torch.zeros(gp:size())

    for i = 1, p:size(1) do
        p:copy(p_backup)
        model:reset()
        p[i] = p[i] - epsilon

        local loss_negative = 0
        if type(input) == 'table' then
            local output_negative = model:forward(input)
            for step = 1, #inputs do
                loss_negative = loss_negative + criterion:forward(output_negative[step], targets[step])
            end
        else
            local output_negative = model:forward(input):clone()
            loss_negative = criterion:forward(output_negative, target)
        end

        p:copy(p_backup)
        model:reset()
        p[i] = p[i] + epsilon

        local loss_positive = 0
        if type(input) == 'table' then
            local output_positive = model:forward(input)
            for step = 1, #inputs do
                loss_positive = loss_positive + criterion:forward(output_positive[step], targets[step])
            end
        else
            local output_positive = model:forward(input):clone()
            loss_positive = criterion:forward(output_positive, target)
        end

        fd_grad[i] = (loss_positive - loss_negative) / (2 * epsilon)
    end

    p:copy(p_backup)
    gp:copy(gp_backup)
    return fd_grad
end

function finiteDiffLSTMInput(model, input, target)
    local epsilon = 1e-6
    -- local input_backup = input:clone()
    local fd_grad_input = torch.zeros(input:size())

    for i = 1, input:size(2) do
        model:reset()
        input_negative = input:clone()
        input_negative[1][i] = input_negative[1][i] - epsilon

        local output_negative = model:forward({input_negative})[1]:clone()
        local loss_negative = criterion:forward(output_negative, target)

        model:reset()
        input_positive = input:clone()
        input_positive[1][i] = input_positive[1][i] + epsilon

        local output_positive = model:forward({input_positive})[1]:clone()
        local loss_positive = criterion:forward(output_positive, target)

        fd_grad_input[1][i] = (loss_positive - loss_negative) / (2 * epsilon)
    end

    -- input:copy(input_backup)
    return fd_grad_input
end

function finiteDiffIIDInput(model, input, target)
    local epsilon = 1e-6
    -- local input_backup = input:clone()
    local fd_grad_input = torch.zeros(input:size())

    for i = 1, input:size(2) do
        model:reset()
        input_negative = input:clone()
        input_negative[1][i] = input_negative[1][i] - epsilon

        local output_negative = model:forward(input_negative)[1]:clone()
        local loss_negative = criterion:forward(output_negative, target)

        model:reset()
        input_positive = input:clone()
        input_positive[1][i] = input_positive[1][i] + epsilon

        local output_positive = model:forward(input_positive)[1]:clone()
        local loss_positive = criterion:forward(output_positive, target)

        fd_grad_input[1][i] = (loss_positive - loss_negative) / (2 * epsilon)
    end

    -- input:copy(input_backup)
    return fd_grad_input
end


if mode == 'step' then
    require 'CFNetwork_multistep'
    model = nn.CFNetwork({
        input_dimension = opt.input_dimension,
        num_functions = opt.num_functions,
        controller_units_per_layer = opt.rnn_size,
        controller_num_layers = opt.controller_num_layers,
        controller_dropout = opt.controller_dropout,
        steps_per_output = opt.steps_per_output,
        controller_nonlinearity = opt.controller_nonlinearity,
        function_nonlinearity = opt.function_nonlinearity,
        encoded_dimension = opt.encoded_dimension,
    })
    p, gp = model:getParameters()
    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()
    model:reset()

    inputs = {}
    targets = {}
    for i = 1, opt.seq_length do
        inputs[i] = torch.rand(opt.batch_size, opt.input_dimension)
        targets[i] = torch.zeros(opt.batch_size, opt.input_dimension)
    end

    outputs = model:forward(inputs)
    grad_outputs = {}
    for i = opt.seq_length, 1, -1 do
        loss = criterion:forward(outputs[i], targets[i])
        grad_outputs[i] = criterion:backward(outputs[i], targets[i])
    end

    model:backward(inputs, grad_outputs)
    backprop_grad = gp:clone()

    model:reset()
    fd_grad = finiteDiff(model, inputs, targets, p, gp)
    -- fd_grad = finiteDiffStep(model, inputs, targets, p, gp)

elseif mode == 'functions' then
    require 'CFNetwork_multistep'
    model = nn.CFNetwork({
        input_dimension = opt.input_dimension,
        num_functions = opt.num_functions,
        controller_units_per_layer = opt.rnn_size,
        controller_num_layers = opt.controller_num_layers,
        controller_dropout = opt.controller_dropout,
        steps_per_output = opt.steps_per_output,
        controller_nonlinearity = opt.controller_nonlinearity,
        function_nonlinearity = opt.function_nonlinearity,
        encoded_dimension = opt.encoded_dimension,
    })
    p, gp = model:getFunctionParameters()
    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()
    model:reset()

    inputs = {}
    targets = {}
    for i = 1, opt.seq_length do
        inputs[i] = torch.rand(opt.batch_size, opt.input_dimension)
        targets[i] = torch.zeros(opt.batch_size, opt.input_dimension)
    end

    outputs = model:forward(inputs)
    grad_outputs = {}
    for i = opt.seq_length, 1, -1 do
        loss = criterion:forward(outputs[i], targets[i])
        grad_outputs[i] = criterion:backward(outputs[i], targets[i])
    end

    model:backward(inputs, grad_outputs)
    backprop_grad = gp:clone()

    model:reset()
    fd_grad = finiteDiff(model, inputs, targets, p, gp)


elseif mode == 'controller' then
    require 'CFNetwork_multistep'
    model = nn.CFNetwork({
        input_dimension = opt.input_dimension,
        num_functions = opt.num_functions,
        controller_units_per_layer = opt.rnn_size,
        controller_num_layers = opt.controller_num_layers,
        controller_dropout = opt.controller_dropout,
        steps_per_output = opt.steps_per_output,
        controller_nonlinearity = opt.controller_nonlinearity,
        function_nonlinearity = opt.function_nonlinearity,
        encoded_dimension = opt.encoded_dimension,
    })
    p, gp = model:getControllerParameters()
    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()
    model:reset()

    inputs = {}
    targets = {}
    for i = 1, opt.seq_length do
        inputs[i] = torch.rand(opt.batch_size, opt.input_dimension)
        targets[i] = torch.zeros(opt.batch_size, opt.input_dimension)
    end

    outputs = model:forward(inputs)
    grad_outputs = {}
    for i = opt.seq_length, 1, -1 do
        loss = criterion:forward(outputs[i], targets[i])
        grad_outputs[i] = criterion:backward(outputs[i], targets[i])
    end

    model:backward(inputs, grad_outputs)
    backprop_grad = gp:clone()

    model:reset()
    fd_grad = finiteDiff(model, inputs, targets, p, gp)

elseif mode == 'lstm' then
    require 'SteppableLSTM'
    model = nn.SteppableLSTM(opt.input_dimension, opt.input_dimension, opt.controller_units_per_layer, opt.controller_num_layers, opt.controller_dropout)
    p, gp = model:getParameters()
    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()
    model:reset()

    inputs = {}
    targets = {}
    for i = 1, opt.seq_length do
        inputs[i] = torch.rand(opt.batch_size, opt.input_dimension)
        targets[i] = torch.zeros(opt.batch_size, opt.input_dimension)
    end

    outputs = model:forward(inputs)
    grad_outputs = {}
    for i = opt.seq_length, 1, -1 do
        loss = criterion:forward(outputs[i], targets[i])
        grad_outputs[i] = criterion:backward(outputs[i], targets[i])
    end

    model:backward(inputs, grad_outputs)
    backprop_grad = gp:clone()

    model:reset()
    fd_grad = finiteDiff(model, inputs, targets, p, gp)

elseif mode == 'multistep' then
    require 'CFNetwork_multistep'
    model = nn.CFNetwork({
        input_dimension = opt.input_dimension,
        num_functions = opt.num_functions,
        controller_units_per_layer = opt.rnn_size,
        controller_num_layers = opt.controller_num_layers,
        controller_dropout = opt.controller_dropout,
        steps_per_output = opt.steps_per_output,
        controller_nonlinearity = opt.controller_nonlinearity,
        function_nonlinearity = opt.function_nonlinearity,
        encoded_dimension = opt.encoded_dimension,
    })
    p, gp = model:getParameters()
    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()
    model:reset()

    inputs = {}
    targets = {}
    for i = 1, opt.seq_length do
        inputs[i] = torch.rand(opt.batch_size, opt.input_dimension)
        targets[i] = torch.zeros(opt.batch_size, opt.input_dimension)
    end

    outputs = model:forward(inputs)
    grad_outputs = {}
    for i = opt.seq_length, 1, -1 do
        loss = criterion:forward(outputs[i], targets[i])
        grad_outputs[i] = criterion:backward(outputs[i], targets[i])
    end

    model:backward(inputs, grad_outputs)
    backprop_grad = gp:clone()

    model:reset()
    fd_grad = finiteDiff(model, inputs, targets, p, gp)


elseif mode == 'lstm-input' then
    require 'SteppableLSTM'
    model = nn.SteppableLSTM(opt.input_dimension, opt.input_dimension, opt.controller_units_per_layer, opt.controller_num_layers, opt.controller_dropout)
    criterion = nn.MSECriterion()
    model:evaluate()

    input = torch.rand(opt.batch_size, opt.input_dimension)
    target = torch.zeros(opt.batch_size, opt.input_dimension)

    output = model:forward({input})[1]
    loss = criterion:forward(output, target)
    grad_output = criterion:backward(output, target)

    backprop_gradInput = model:backward({input}, {grad_output})[1][1]:clone()

    model:reset()
    fd_gradInput = finiteDiffLSTMInput(model, input, target)[1]
    print(vis.simplestr(backprop_gradInput))
    print(vis.simplestr(fd_gradInput))

    print("gradInput error:", (backprop_gradInput - fd_gradInput):norm())
    print("Average true gradInput size:", fd_gradInput:norm(1) / fd_gradInput:size(1))
    print("Average backprop gradInput size:", backprop_gradInput:norm(1) / backprop_gradInput:size(1))
    -- print("Average gradient error per param:", (backprop_grad - fd_grad):norm(1) / p:size(1))
    print("Average gradInput percent error:", (backprop_gradInput - fd_gradInput):norm(1) / fd_gradInput:norm(1) * 100)
    error("Done")


elseif mode == 'iid-functions' then
    require 'IIDCFNetwork'
    model = nn.IIDCFNetwork({
        input_dimension = opt.input_dimension,
        num_functions = opt.num_functions,
        controller_units_per_layer = opt.controller_units_per_layer,
        controller_num_layers = opt.controller_num_layers,
        controller_dropout = opt.controller_dropout,
        steps_per_output = opt.steps_per_output,
    })
    p, gp = model:getFunctionParameters()
    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()
    model:reset()

    input = torch.rand(opt.batch_size, opt.input_dimension)
    target = torch.zeros(opt.batch_size, opt.input_dimension)

    output = model:forward(input):clone()
    loss = criterion:forward(output, target)
    model:backward(input, criterion:backward(output, target))
    backprop_grad = gp:clone()

    model:reset()
    fd_grad = finiteDiff(model, input, target, p, gp)

elseif mode == 'iid-input' then
    require 'IIDCFNetwork'
    model = nn.IIDCFNetwork({
        input_dimension = opt.input_dimension,
        num_functions = opt.num_functions,
        controller_units_per_layer = opt.controller_units_per_layer,
        controller_num_layers = opt.controller_num_layers,
        controller_dropout = opt.controller_dropout,
        steps_per_output = opt.steps_per_output,
    })

    criterion = nn.MSECriterion()
    model:evaluate()
    model:reset()

    input = torch.rand(opt.batch_size, opt.input_dimension)
    target = torch.zeros(opt.batch_size, opt.input_dimension)

    output = model:forward(input):clone()
    loss = criterion:forward(output, target)
    backprop_gradInput = model:backward(input, criterion:backward(output, target)):clone()

    model:reset()
    fd_gradInput = finiteDiffIIDInput(model, input, target)

    print("gradInput error:", (backprop_gradInput - fd_gradInput):norm())
    print("Average true gradInput size:", fd_gradInput:norm(1) / fd_gradInput:size(1))
    print("Average backprop gradInput size:", backprop_gradInput:norm(1) / backprop_gradInput:size(1))
    -- print("Average gradient error per param:", (backprop_grad - fd_grad):norm(1) / p:size(1))
    print("Average gradInput percent error:", (backprop_gradInput - fd_gradInput):norm(1) / fd_gradInput:norm(1) * 100)
    error("Done")


elseif mode == 'iid-controller' then
    require 'IIDCFNetwork'
    model = nn.IIDCFNetwork({
        input_dimension = opt.input_dimension,
        num_functions = opt.num_functions,
        controller_units_per_layer = opt.controller_units_per_layer,
        controller_num_layers = opt.controller_num_layers,
        controller_dropout = opt.controller_dropout,
        steps_per_output = opt.steps_per_output,
    })
    p, gp = model:getControllerParameters()
    p_backup = p:clone()
    gp_backup = gp:clone()

    criterion = nn.MSECriterion()
    model:evaluate()
    model:reset()

    input = torch.rand(opt.batch_size, opt.input_dimension)
    target = torch.zeros(opt.batch_size, opt.input_dimension)

    output = model:forward(input):clone()
    loss = criterion:forward(output, target)
    model:backward(input, criterion:backward(output, target))
    backprop_grad = gp:clone()

    model:reset()
    fd_grad = finiteDiff(model, input, target, p, gp)


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

    input = torch.rand(opt.batch_size, opt.input_dimension)
    target = torch.zeros(opt.batch_size, opt.input_dimension)

    output = model:forward(input):clone()
    loss = criterion:forward(output, target)
    model:backward(input, criterion:backward(output, target))
    backprop_grad = gp:clone()

    model:reset()
    fd_grad = finiteDiff(model, input, target, p, gp)
end

print("Gradient error:", (backprop_grad - fd_grad):norm())
print("Average true gradient size:", fd_grad:norm(1) / fd_grad:size(1))
print("Average backprop gradient size:", backprop_grad:norm(1) / backprop_grad:size(1))
-- print("Average gradient error per param:", (backprop_grad - fd_grad):norm(1) / p:size(1))
print("Average gradient percent error:", (backprop_grad - fd_grad):norm(1) / fd_grad:norm(1) * 100)
-- print(vis.simplestr(backprop_grad))
-- print(vis.simplestr(fd_grad))
