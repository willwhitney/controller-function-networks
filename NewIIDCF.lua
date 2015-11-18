require 'nn'
require 'Controller'
require 'Constant'
require 'vis'

--[[
This version is reserved for data that is iid between each :forward().
It has no memory across calls to :forward; it performs multiple internal steps
for each :forward, but does not backpropagate over multiple inputs.
--]]


IIDCFNetwork, parent = torch.class('nn.IIDCFNetwork', 'nn.Module')

function IIDCFNetwork:__init(options)
    self.controller = nn.Controller(
            options.encoded_dimension, -- needs to look at the whole input
            options.num_functions, -- outputs a weighting over all the functions
            options.controller_units_per_layer,
            options.controller_num_layers,
            options.controller_dropout,
            options.controller_nonlinearity )

    self.steps_per_output = options.steps_per_output or 1

    self.functions = {}
    for i = 1, options.num_functions do

        -- local const = torch.zeros(opt.batch_size, options.input_dimension)
        -- const[{{}, {i}}] = 1
        -- local layer = nn.Constant(const)

        local layer = nn.Sequential()
        layer:add(nn.Linear(options.encoded_dimension, options.encoded_dimension))
        -- layer:add(nn.Sigmoid())
        if options.function_nonlinearity == 'sigmoid' then
            layer:add(nn.Sigmoid())
        elseif options.function_nonlinearity == 'tanh' then
            layer:add(nn.Tanh())
        elseif options.function_nonlinearity == 'relu' then
            layer:add(nn.ReLU())
        elseif options.function_nonlinearity == 'none' then

        else
            error("Must specify a nonlinearity for the functions.")
        end

        print(layer)

        -- local layer = nn.Sequential()
        -- layer:add(nn.Linear(options.input_dimension, options.input_dimension))
        -- layer:add(nn.Tanh())

        table.insert(self.functions, layer)
    end

    -- for i = 1, #self.functions do
    --     print(self.functions[i])
    -- end

    self.encoder = nn.Linear(options.input_dimension, options.encoded_dimension)
    self.decoder = nn.Linear(options.encoded_dimension, options.input_dimension)
    self.mixtable = nn.MixtureTable()
    self:reset()
end

function IIDCFNetwork:step(input)
    local controller_output = self.controller:step(input)
    -- print(vis.simplestr(controller_output[1]))

    local function_outputs = {}
    for i = 1, #self.functions do
        local function_output = self.functions[i]:forward(input):clone()
        table.insert(function_outputs, function_output)
    end

    local current_output = self.mixtable:forward({controller_output, function_outputs}):clone()
    local step_trace = {
            input = input:clone(),
            output = current_output:clone(),
        }
    table.insert(self.trace, step_trace)
    return current_output
end

function IIDCFNetwork:forward(input)
    self:reset()
    local next_input = self.encoder:forward(input):clone()
    for t = 1, self.steps_per_output do
        next_input = self:step(next_input):clone()
    end

    self.output = self.decoder:forward(next_input):clone()
    return self.output
end

function IIDCFNetwork:updateOutput(input)
    return self:forward(input)
end

function IIDCFNetwork:backstep(t, gradOutput)
    local step_trace = self.trace[t]
    local step_input = step_trace.input

    local controller_step_trace = self.controller.trace[#self.controller.trace]
    local controller_output = controller_step_trace[#controller_step_trace].output

    -- forward the functions to guarantee correct operation
    local function_outputs = {}
    for i = 1, #self.functions do
        table.insert(function_outputs, self.functions[i]:forward(step_input):clone())
    end
    self.mixtable:forward({controller_output, function_outputs})

    local grad_table = self.mixtable:backward(
            {controller_output, function_outputs},
            gradOutput)

    local grad_controller_output = grad_table[1]
    local grad_function_outputs = grad_table[2]

    current_gradOutput = self.controller:backstep(nil, grad_controller_output):clone()
    for i = 1, #self.functions do
        current_gradOutput = current_gradOutput + self.functions[i]:backward(step_input, grad_function_outputs[i])
    end
    return current_gradOutput
end

function IIDCFNetwork:backward(input, gradOutput)
    print(self.trace)
    local timestep = #self.trace
    local step_trace = self.trace[timestep]
    -- print(step_trace.input)
    -- print(self.encoder:forward(input))
    if step_trace.input:norm() ~= self.encoder:forward(input):norm() then
        error("IIDCFNetwork:backstep has been called in the wrong order.")
    end
    local current_gradOutput = self.decoder:backward(step_trace.output, gradOutput)

    for t = self.steps_per_output, 1, -1 do
        current_gradOutput = self:backstep(t, current_gradOutput)

        -- pop this timestep from our stack
        self.trace[t] = nil
    end
    self.gradInput = self.encoder:backward(input, current_gradOutput)

    return self.gradInput
end

function IIDCFNetwork:reset(batch_size)
    self.trace = {}
    batch_size = batch_size or opt.batch_size
    self.controller:reset(batch_size)
end

-- taken from nn.Container
function IIDCFNetwork:parameters()
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
    return w,gw
end


-- taken from nn.Container
function IIDCFNetwork:function_parameters()
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

function IIDCFNetwork:getFunctionParameters()
    local f_parameters, f_gradParameters = self:function_parameters()
    return parent.flatten(f_parameters), parent.flatten(f_gradParameters)
end

-- taken from nn.Container
function IIDCFNetwork:controller_parameters()
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

function IIDCFNetwork:getControllerParameters()
    local c_parameters, c_gradParameters = self:controller_parameters()
    return parent.flatten(c_parameters), parent.flatten(c_gradParameters)
end

function IIDCFNetwork:training()
    for i = 1, #self.functions do
        self.functions[i]:training()
    end
    self.mixtable:training()
    self.controller:training()
end

function IIDCFNetwork:evaluate()
    for i = 1, #self.functions do
        self.functions[i]:evaluate()
    end
    self.mixtable:evaluate()
    self.controller:evaluate()
end

function IIDCFNetwork:cuda()
    for i = 1, #self.functions do
        self.functions[i]:cuda()
    end
    self.mixtable:cuda()
    self.controller:cuda()
end

function IIDCFNetwork:float()
    for i = 1, #self.functions do
        self.functions[i]:float()
    end
    self.mixtable:float()
    self.controller:float()
end

function IIDCFNetwork:double()
    for i = 1, #self.functions do
        self.functions[i]:double()
    end
    self.mixtable:double()
    self.controller:double()
end
