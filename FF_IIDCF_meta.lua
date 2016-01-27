require 'nn'
require 'Controller'
require 'SharpeningController'
require 'ScheduledSharpeningController'
-- require 'ExpectationCriterion'
require 'Constant'
require 'vis'

--[[
This version is reserved for data that is iid between each :forward().
It has no memory across calls to :forward; it performs multiple internal steps
for each :forward, but does not backpropagate over multiple inputs.
--]]


IIDCFNetwork, parent = torch.class('nn.IIDCFNetwork', 'nn.Module')

function IIDCFNetwork:__init(options)
    self.controller = nn.Sequential()
    self.controller:add(nn.Linear(options.input_dimension, options.num_functions))
    self.controller:add(nn.Sigmoid())
    if options.controller_noise > 0 then
        self.controller:add(nn.Noise(options.controller_noise))
    end
    self.controller:add(nn.ScheduledWeightSharpener())
    self.controller:add(nn.AddConstant(1e-20))
    self.controller:add(nn.Normalize(1, 1e-100))

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
        elseif options.function_nonlinearity == 'prelu' then
            layer:add(nn.PReLU())
        elseif options.function_nonlinearity == 'none' then

        else
            error("Must specify a nonlinearity for the functions.")
        end

        -- local layer = nn.Sequential()
        -- layer:add(nn.Linear(options.input_dimension, options.input_dimension))
        -- layer:add(nn.Tanh())
        print(layer)
        table.insert(self.functions, layer)
    end

    -- for i = 1, #self.functions do
    --     print(self.functions[i])
    -- end

    self.mixtable = nn.MixtureTable()
    -- self.criterion = nn.ExpectationCriterion(nn.MSECriterion())
    self.jointable = nn.JoinTable(2)
    self:reset()
end

function IIDCFNetwork:step(input)
    local controller_metadata, input_vector = table.unpack(input)
    -- print("step info: ")
    -- print(controller_metadata)
    -- print(input_vector)
    local controller_input = self.jointable:forward(input):clone()
    local controller_output = self.controller:forward(controller_input):clone()
    -- print(controller_output)
    print('weights:', vis.simplestr(controller_output[1]))

    local temp = torch.zeros(#self.functions, input_vector:size(2))
    local function_outputs = {}
    for i = 1, #self.functions do
        local function_output = self.functions[i]:forward(input_vector):clone()
        table.insert(function_outputs, function_output)
        temp[i] = function_output

        -- local function_output = self.functions[i]:forward(input_vector):clone()[1]
        -- function_outputs[i] = function_output:clone()
    end
    print(temp)
    -- local current_output = {controller_output, function_outputs}
    -- print({controller_output[1], function_outputs})
    local current_output = self.mixtable:forward({controller_output, function_outputs}):clone()
    local step_trace = {
            input = {
                    input[1]:clone(),
                    input[2]:clone(),
                },
            -- output = current_output:clone(),
            output = current_output:clone(),
        }
    table.insert(self.trace, step_trace)
    return current_output
end

function IIDCFNetwork:forward(input)
    -- print("forward")
    self:reset()
    local controller_metadata, input_vector = table.unpack(input)
    -- print("input[1]: ", input[1])
    local next_input = input_vector
    for t = 1, self.steps_per_output do
        -- print("controller_metadata[t]: ", controller_metadata[t])
        local step_controller_metadata = controller_metadata[t]:reshape(1, controller_metadata[t]:size(1))
        next_input = self:step({step_controller_metadata, next_input}):clone()
    end

    self.output = next_input
    return self.output
end

function IIDCFNetwork:updateOutput(input)
    return self:forward(input)
end

function IIDCFNetwork:updateGradInput(input, gradOutput)
    return self:backward(input, gradOutput)
end

function IIDCFNetwork:backstep(t, gradOutput)
    -- print(self.trace)
    print(t, gradOutput)
    local step_trace = self.trace[t]
    local step_input = step_trace.input

    local controller_metadata, input_vector = table.unpack(step_input)
    -- local grad_probs, grad_outputs = table.unpack(gradOutput)
    local controller_input = self.jointable:forward(input)
    local controller_output = self.controller:forward(controller_input)

    -- forward the functions to guarantee correct operation
    -- local function_outputs = torch.zeros(#self.functions, input_vector:size(2))
    local function_outputs = {}
    for i = 1, #self.functions do
        table.insert(function_outputs, self.functions[i]:forward(input_vector):clone())
        -- local function_output = self.functions[i]:forward(input_vector):clone()[1]
        -- function_outputs[i] = function_output
    end
    -- print({controller_output, function_outputs})
    self.mixtable:forward({controller_output, function_outputs})
    -- print("controller_output: ", controller_output)
    -- print("function_outputs: ", function_outputs)
    -- print("gradOutput: ", gradOutput)
    local grad_table = self.mixtable:backward(
            {controller_output, function_outputs},
            gradOutput)

    local grad_controller_output = grad_table[1]
    local grad_function_outputs = grad_table[2]

    local grad_controller_input = self.controller:backward(controller_input, grad_controller_output):clone()
    local grad_input_table = self.jointable:backward(step_input, grad_controller_input)

    -- ^ yields a table of form {grad_controller_metadata, grad_input_vector}

    -- print(grad_outputs)
    for i = 1, #self.functions do
        -- add the gradients from each of the functions to grad_input_vector
        grad_input_table[2] = grad_input_table[2] + self.functions[i]:backward(input_vector, grad_function_outputs[i])
    end
    return grad_input_table
end

function IIDCFNetwork:backward(input, gradOutput)
    -- print("backward")
    self.gradInput = {torch.zeros(input[1]:size()), torch.zeros(input[2]:size())}
    -- print("input\n", input)
    -- print("target\n", target)
    local timestep = #self.trace
    if self.trace[1].input[2]:norm() ~= input[2]:norm() then
        error("IIDCFNetwork:backstep has been called in the wrong order.")
    end
    -- local gradOutput = self.criterion:backward(self.trace[timestep].output, target)
    -- print("gradOutput\n", gradOutput)
    local current_gradOutput = gradOutput

    for t = self.steps_per_output, 1, -1 do
        local current_gradInput = self:backstep(t, current_gradOutput)
        current_gradOutput = current_gradInput[2]
        self.gradInput[1][t] = current_gradInput[1]

        -- pop this timestep from our stack
        -- self.trace[t] = nil
    end
    self.gradInput[2] = current_gradOutput

    return self.gradInput
end

function IIDCFNetwork:reset(batch_size)
    self.trace = {}
    batch_size = batch_size or opt.batch_size
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
