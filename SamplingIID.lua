require 'nn'
require 'Controller'
require 'ExpectationCriterion'
require 'Constant'
require 'vis'

--[[
This version is reserved for data that is iid between each :forward().
It has no memory across calls to :forward; it performs multiple internal steps
for each :forward, but does not backpropagate over multiple inputs.
--]]


SamplingCFNetwork, parent = torch.class('nn.SamplingCFNetwork', 'nn.Module')

function SamplingCFNetwork:__init(options)
    self.controller = nn.Controller(
            options.input_dimension, -- needs to look at the whole input
            options.num_functions, -- outputs a weighting over all the functions
            options.controller_units_per_layer,
            options.controller_num_layers,
            options.controller_dropout,
            'softmax' )

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

        -- local layer = nn.Sequential()
        -- layer:add(nn.Linear(options.input_dimension, options.input_dimension))
        -- layer:add(nn.Tanh())

        table.insert(self.functions, layer)
    end

    for i = 1, #self.functions do
        print(self.functions[i])
    end

    self.mixtable = nn.MixtureTable()
    self.criterion = nn.ExpectationCriterion(nn.MSECriterion())
    self.jointable = nn.JoinTable(2)
    self:reset()
end

function SamplingCFNetwork:step(input)
    local controller_metadata, input_vector = table.unpack(input)
    local controller_input = self.jointable:forward(input)
    local controller_output = self.controller:step(controller_input)
    print(vis.simplestr(controller_output[1]))

    local function_outputs = torch.zeros(#self.functions, input_vector:size(2))
    for i = 1, #self.functions do
        local function_output = self.functions[i]:forward(input_vector):clone()[1]
        function_outputs[i] = function_output
    end
    local current_output = {controller_output, function_outputs}
    -- local current_output = self.mixtable:forward({controller_output, function_outputs}):clone()
    local step_trace = {
            input = {
                    input[1]:clone(),
                    input[2]:clone(),
                },
            -- output = current_output:clone(),
            output = current_output,
        }
    table.insert(self.trace, step_trace)
    return current_output
end

function SamplingCFNetwork:forward(input, target)
    self:reset()
    -- print(input)
    local next_input = input
    for t = 1, self.steps_per_output do
        next_input = self:step(next_input)
    end

    self.output = self.criterion:forward(next_input, target)
    return self.output
end

function SamplingCFNetwork:backstep(t, gradOutput)
    local step_trace = self.trace[t]
    local step_input = step_trace.input

    local controller_metadata, input_vector = table.unpack(step_input)
    local grad_probs, grad_outputs = table.unpack(gradOutput)
    local controller_input = self.jointable:forward(input)

    local controller_step_trace = self.controller.trace[#self.controller.trace]
    local controller_output = controller_step_trace[#controller_step_trace].output

    -- forward the functions to guarantee correct operation
    local function_outputs = {}
    for i = 1, #self.functions do
        table.insert(function_outputs, self.functions[i]:forward(input_vector):clone())
    end
    -- self.mixtable:forward({controller_output, function_outputs})

    -- local grad_table = self.mixtable:backward(
            -- {controller_output, function_outputs},
            -- current_gradOutput)

    -- local grad_controller_output = grad_table[1]
    -- local grad_function_outputs = grad_table[2]

    local grad_controller_input = self.controller:backstep(controller_input, grad_probs):clone()
    local grad_input_table = self.jointable:backward(step_input, grad_controller_input)
    -- ^ yields a table of form {grad_controller_metadata, grad_input_vector}

    -- print(grad_outputs)
    for i = 1, #self.functions do
        -- add the gradients from each of the functions to grad_input_vector
        grad_input_table[2] = grad_input_table[2] + self.functions[i]:backward(input_vector, grad_outputs[i])
    end
    return current_gradOutput
end

function SamplingCFNetwork:backward(input, target)
    -- print("input\n", input)
    -- print("target\n", target)
    local timestep = #self.trace
    if self.trace[1].input[2]:norm() ~= input[2]:norm() then
        error("SamplingCFNetwork:backstep has been called in the wrong order.")
    end
    local gradOutput = self.criterion:backward(self.trace[timestep].output, target)
    -- print("gradOutput\n", gradOutput)
    local current_gradOutput = gradOutput

    for t = self.steps_per_output, 1, -1 do
        current_gradOutput = self:backstep(t, current_gradOutput)

        -- pop this timestep from our stack
        -- self.trace[t] = nil
    end
    self.gradInput = current_gradOutput

    return self.gradInput
end

function SamplingCFNetwork:reset(batch_size)
    self.trace = {}
    batch_size = batch_size or opt.batch_size
    self.controller:reset(batch_size)
end

-- taken from nn.Container
function SamplingCFNetwork:parameters()
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
function SamplingCFNetwork:function_parameters()
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

function SamplingCFNetwork:getFunctionParameters()
    local f_parameters, f_gradParameters = self:function_parameters()
    return parent.flatten(f_parameters), parent.flatten(f_gradParameters)
end

-- taken from nn.Container
function SamplingCFNetwork:controller_parameters()
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

function SamplingCFNetwork:getControllerParameters()
    local c_parameters, c_gradParameters = self:controller_parameters()
    return parent.flatten(c_parameters), parent.flatten(c_gradParameters)
end

function SamplingCFNetwork:training()
    for i = 1, #self.functions do
        self.functions[i]:training()
    end
    self.mixtable:training()
    self.controller:training()
end

function SamplingCFNetwork:evaluate()
    for i = 1, #self.functions do
        self.functions[i]:evaluate()
    end
    self.mixtable:evaluate()
    self.controller:evaluate()
end

function SamplingCFNetwork:cuda()
    for i = 1, #self.functions do
        self.functions[i]:cuda()
    end
    self.mixtable:cuda()
    self.controller:cuda()
end

function SamplingCFNetwork:float()
    for i = 1, #self.functions do
        self.functions[i]:float()
    end
    self.mixtable:float()
    self.controller:float()
end

function SamplingCFNetwork:double()
    for i = 1, #self.functions do
        self.functions[i]:double()
    end
    self.mixtable:double()
    self.controller:double()
end
