require 'nn'
require 'Controller'
require 'Constant'
require 'vis'

CFNetwork, parent = torch.class('nn.CFNetwork', 'nn.Module')

function CFNetwork:__init(options)
    self.controller = nn.Controller(
            options.encoded_dimension, -- needs to look at the whole input
            options.num_functions, -- outputs a weighting over all the functions
            options.controller_units_per_layer,
            options.controller_num_layers,
            options.controller_dropout,
            options.controller_nonlinearity )

    self.steps_per_output = options.steps_per_output or 1

    self.encoder = nn.Linear(options.input_dimension, options.encoded_dimension)
    self.decoder = nn.Linear(options.encoded_dimension, options.input_dimension)

    self.functions = {}
    for i = 1, options.num_functions do

        -- local const = torch.zeros(opt.batch_size, options.encoded_dimension)
        -- const[{{}, {i}}] = 1
        -- local layer = nn.Constant(const)

        -- local layer = nn.Sequential()
        -- layer:add(nn.Linear(options.encoded_dimension, options.encoded_dimension))
        -- layer:add(nn.Sigmoid())
        -- layer:add(nn.PReLU())

        local layer = nn.Sequential()
        layer:add(nn.Linear(options.encoded_dimension, options.encoded_dimension))
        if options.function_nonlinearity == 'sigmoid' then
            layer:add(nn.Sigmoid())
        elseif options.function_nonlinearity == 'tanh' then
            layer:add(nn.Tanh())
        elseif options.function_nonlinearity == 'relu' then
            layer:add(nn.ReLU())
        elseif options.function_nonlinearity == 'prelu' then
            layer:add(nn.PReLU())
        elseif options.function_nonlinearity == 'none' then

        else
            error("Must specify a nonlinearity for the functions.")
        end

        -- local layer = KarpathyLSTM.lstm(options.encoded_dimension, options.encoded_dimension, 1, dropout)

        table.insert(self.functions, layer)
    end

    for i = 1, #self.functions do
        print(self.functions[i])
    end

    self.mixtable = nn.MixtureTable()
    self:reset()
end

function CFNetwork:forward(inputs)
    -- print(inputs)
    self:reset()
    if type(inputs) == 'table' then
        -- the input should be a table of sequential inputs such that
        -- inputs[2] is chronologically after inputs[1]
        local outputs = {}
        for i = 1, #inputs do
            outputs[i] = self:step(inputs[i]):clone()
        end
        self.output = outputs
    else
        -- if given a tensor, the input should be indexed as
        -- inputs[batch_index][sequence_timestep][one_hot_index]
        local outputs = {}
        for i = 1, inputs:size(2) do
            local current_input = inputs[{{}, {i}}]:reshape(inputs:size(1), inputs:size(3))
            outputs[i] = self:step(current_input):clone()
        end
        self.output = outputs
    end
    -- print(self.output)
    return self.output
end

function CFNetwork:updateOutput(inputs)
    return self:forward(inputs)
end

function CFNetwork:step(input)
    local next_input = self.encoder:forward(input)
    local step_trace = {}
    for substep = 1, self.steps_per_output do
        local controller_output = self.controller:step(next_input)

        local function_outputs = {}
        for i = 1, #self.functions do
            local function_output = self.functions[i]:forward(next_input):clone()
            table.insert(function_outputs, function_output)
        end

        local current_output = self.mixtable:forward({controller_output, function_outputs}):clone()
        local substep_trace = {
                input = next_input,
                output = current_output,
            }
        table.insert(step_trace, substep_trace)
        next_input = current_output:clone()
    end

    table.insert(self.trace, step_trace)

    self.output = self.decoder:forward(next_input):clone()
    return self.output
end

function CFNetwork:backstep(input, gradOutput)
    local timestep = #self.trace
    local step_trace = self.trace[timestep]

    local encoded_input = self.encoder:forward(input)
    if step_trace[1].input:sum() ~= encoded_input:sum() then
        error("CFNetwork:backstep has been called in the wrong order.")
    end

    self.decoder:forward(step_trace[#step_trace].output)
    local current_gradInput = self.decoder:backward(step_trace[#step_trace].output, gradOutput)

    for substep = self.steps_per_output, 1, -1 do
        local substep_trace = step_trace[substep]
        local substep_input = substep_trace.input

        local controller_step_trace = self.controller.trace[#self.controller.trace]
        local controller_output = controller_step_trace[#controller_step_trace].output

        -- forward the functions to guarantee correct operation
        local function_outputs = {}
        for i = 1, #self.functions do
            table.insert(function_outputs, self.functions[i]:forward(substep_input))
        end
        self.mixtable:forward({controller_output, function_outputs})

        local grad_table = self.mixtable:backward(
                {controller_output, function_outputs},
                current_gradInput)

        local grad_controller_output = grad_table[1]
        local grad_function_outputs = grad_table[2]

        current_gradInput = self.controller:backstep(nil, grad_controller_output):clone()
        for i = 1, #self.functions do
            current_gradInput = current_gradInput + self.functions[i]:backward(substep_input, grad_function_outputs[i])
        end
    end
    self.gradInput = self.encoder:backward(input, current_gradInput)

    -- pop this timestep from our stack
    self.trace[timestep] = nil
    return self.gradInput
end

function CFNetwork:backward(inputs, grad_outputs)
    if type(inputs) == 'table' then
        for timestep = #inputs, 1, -1 do
            self:backstep(inputs[timestep], grad_outputs[timestep])
        end
    else
        for timestep = #grad_outputs, 1, -1 do
            local input = inputs[{{}, {timestep}}]:reshape(inputs:size(1), inputs:size(3))
            self:backstep(input, grad_outputs[timestep])
        end
    end
end

function CFNetwork:reset(batch_size)
    self.trace = {}
    batch_size = batch_size or opt.batch_size
    self.controller:reset(batch_size)
end

-- taken from nn.Container
function CFNetwork:parameters()
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i=1,#from do
                tinsert(to,from[i])
            end
        else
            table.insert(to,from)
        end
    end
    local w = {}
    local gw = {}
    for i=1,#self.functions do
        local mw,mgw = self.functions[i]:parameters()
        if mw then
            tinsert(w,mw)
            tinsert(gw,mgw)
        end
    end
    local mw,mgw = self.controller:parameters()
    if mw then
        tinsert(w,mw)
        tinsert(gw,mgw)
    end
    local mw,mgw = self.encoder:parameters()
    if mw then
        tinsert(w,mw)
        tinsert(gw,mgw)
    end
    local mw,mgw = self.decoder:parameters()
    if mw then
        tinsert(w,mw)
        tinsert(gw,mgw)
    end
    return w,gw
end


-- taken from nn.Container
function CFNetwork:function_parameters()
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i=1,#from do
                tinsert(to,from[i])
            end
        else
            table.insert(to,from)
        end
    end
    local w = {}
    local gw = {}
    for i=1,#self.functions do
        local mw,mgw = self.functions[i]:parameters()
        if mw then
            tinsert(w,mw)
            tinsert(gw,mgw)
        end
    end
    return w,gw
end

function CFNetwork:getFunctionParameters()
    local f_parameters, f_gradParameters = self:function_parameters()
    return parent.flatten(f_parameters), parent.flatten(f_gradParameters)
end

-- taken from nn.Container
function CFNetwork:controller_parameters()
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i=1,#from do
                tinsert(to,from[i])
            end
        else
            table.insert(to,from)
        end
    end
    local w = {}
    local gw = {}

    local mw,mgw = self.controller:parameters()
    if mw then
        tinsert(w,mw)
        tinsert(gw,mgw)
    end
    return w,gw
end

function CFNetwork:getControllerParameters()
    local c_parameters, c_gradParameters = self:controller_parameters()
    return parent.flatten(c_parameters), parent.flatten(c_gradParameters)
end

function CFNetwork:training()
    for i = 1, #self.functions do
        self.functions[i]:training()
    end
    self.mixtable:training()
    self.encoder:training()
    self.decoder:training()
    self.controller:training()
end

function CFNetwork:evaluate()
    for i = 1, #self.functions do
        self.functions[i]:evaluate()
    end
    self.mixtable:evaluate()
    self.encoder:evaluate()
    self.decoder:evaluate()
    self.controller:evaluate()
end

function CFNetwork:cuda()
    for i = 1, #self.functions do
        self.functions[i]:cuda()
    end
    self.mixtable:cuda()
    self.encoder:cuda()
    self.decoder:cuda()
    self.controller:cuda()
end

function CFNetwork:float()
    for i = 1, #self.functions do
        self.functions[i]:float()
    end
    self.mixtable:float()
    self.encoder:float()
    self.decoder:float()
    self.controller:float()
end

function CFNetwork:double()
    for i = 1, #self.functions do
        self.functions[i]:double()
    end
    self.mixtable:double()
    self.encoder:double()
    self.decoder:double()
    self.controller:double()
end
