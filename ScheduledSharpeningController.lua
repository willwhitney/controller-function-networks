require 'nn'
require 'nngraph'

require 'ScheduledWeightSharpener'
require 'Noise'
require 'Print'
require 'Renormalize'
LSTM = require 'LSTM'
KarpathyLSTM = require 'KarpathyLSTM'

ScheduledSharpeningController, parent = torch.class('nn.ScheduledSharpeningController', 'nn.Module')


function ScheduledSharpeningController:__init(
            input_dimension,
            output_dimension,
            num_units_per_layer,
            num_layers,
            dropout,
            nonlinearity,
            noise )
    -- print("input size:", input_dimension)
    self.input_dimension = input_dimension
    self.output_dimension = output_dimension
    self.num_units_per_layer = num_units_per_layer
    print("Units per layer: ", self.num_units_per_layer)

    self.network = {}

    -- create the input layer with different input size
    table.insert(self.network,
        KarpathyLSTM.lstm(self.input_dimension, self.num_units_per_layer, 1, dropout))

    -- and the other n_units -> n_units layers
    for i = 2, num_layers do
        table.insert(self.network,
            KarpathyLSTM.lstm(self.num_units_per_layer, self.num_units_per_layer, 1, dropout))
    end

    if nonlinearity ~= 'sigmoid' then
        print("Warning: overriding controller nonlinearity with sigmoid.")
        nonlinearity = 'sigmoid'
    end

    weightNonlinearity = nn.Identity
    if nonlinearity == 'sigmoid' then
        weightNonlinearity = nn.Sigmoid
    elseif nonlinearity == 'tanh' then
        weightNonlinearity = nn.Tanh
    elseif nonlinearity == 'relu' then
        weightNonlinearity = nn.ReLU
    elseif nonlinearity == 'prelu' then
        weightNonlinearity = nn.PReLU
    elseif nonlinearity == 'softmax' then
        weightNonlinearity = nn.SoftMax
    else
        error("Must specify a nonlinearity for the controller.")
    end


    -- last layer smushes back down to output domain, then outputs (0-1) weights
    self.decoder = nn.Sequential()
    self.decoder:add(nn.Linear(self.num_units_per_layer, self.output_dimension))
    self.decoder:add(weightNonlinearity())

    if noise > 0 then
        self.decoder:add(nn.Noise(noise))
    end

    -- local postParallel = nn.ConcatTable()
    --
    -- -- make a tensor with output_dimension components
    -- local weightPipe = nn.Sequential()
    --     weightPipe:add(nn.SplitTable(2))
    --     weightPipe:add(nn.NarrowTable(2, self.output_dimension))
    --     weightPipe:add(nn.JoinTable(1))
    --     weightPipe:add(weightNonlinearity())
    -- postParallel:add(weightPipe)
    --
    -- -- pull out the last component for the exponent
    -- local sharpeningPipe = nn.Sequential()
    --     sharpeningPipe:add(nn.SplitTable(2))
    --     sharpeningPipe:add(nn.SelectTable(1))
    --     sharpeningPipe:add(nn.SoftPlus())
    -- postParallel:add(sharpeningPipe)
    --
    -- self.decoder:add(postParallel)

    self.decoder:add(nn.ScheduledWeightSharpener())
    -- self.decoder:add(nn.Renormalize())
    -- self.decoder:add(nn.Print("Before Normalize"))
    self.decoder:add(nn.AddConstant(1e-20))
    self.decoder:add(nn.Normalize(1, 1e-100))
    -- self.decoder:add(nn.Print("After Decoder"))

    -- if nonlinearity == 'sigmoid' then
    --     self.decoder:add(nn.Sigmoid())
    -- elseif nonlinearity == 'tanh' then
    --     self.decoder:add(nn.Tanh())
    -- elseif nonlinearity == 'relu' then
    --     self.decoder:add(nn.ReLU())
    -- elseif nonlinearity == 'prelu' then
    --     self.decoder:add(nn.PReLU())
    -- elseif nonlinearity == 'softmax' then
    --     self.decoder:add(nn.SoftMax())
    -- elseif nonlinearity == 'none' then
    --
    -- else
    --     error("Must specify a nonlinearity for the controller.")
    -- end
    -- self.decoder:add(nn.Sigmoid())

    -- self.decoder = nn.Linear(self.num_units_per_layer, self.output_dimension)

    self:reset()
end

function ScheduledSharpeningController:reset(batch_size)
    batch_size = batch_size or opt.batch_size
    self.trace = {}
    self.backtrace = {}

    -- create a first state with all previous outputs & cells set to zeros
    self.state = {}
    for i = 1, #self.network do
        local layer_state
        if self.decoder.modules[1].weight:type() == "torch.CudaTensor" then
            layer_state = {
                    torch.zeros(batch_size, self.num_units_per_layer):cuda(),  -- prev_c
                    torch.zeros(batch_size, self.num_units_per_layer):cuda(),  -- prev_h
                }
        else
            layer_state = {
                    torch.zeros(batch_size, self.num_units_per_layer),  -- prev_c
                    torch.zeros(batch_size, self.num_units_per_layer),  -- prev_h
                }
        end

        table.insert(self.state, layer_state)
    end
end

-- take one timestep with this input
-- if using the model this way, make sure to call reset() between sequences
function ScheduledSharpeningController:step(input)
    local current_input = input:clone()
    local step_trace = {}
    local output

    -- #self.network is the number of layers
    for i = 1, #self.network do
        local inputs = {
                current_input,
                self.state[i][1],  -- prev_c for this layer
                self.state[i][2],  -- prev_h for this layer
            }
        -- print(inputs)

        output = self.network[i]:forward(inputs)

        -- the trace for this layer at this step is just a table of its
        -- inputs and outputs
        local layer_step_trace = {
            inputs = inputs,
            output = output,
        }

        -- the input for the next layer is next_h from this one
        current_input = output[2]:clone() -- cloning defensively, TODO: remove

        -- the output was {next_c, next_h}, which is
        -- the state for this layer at the next step
        self.state[i] = {
            output[1]:clone(), -- cloning defensively, TODO: remove
            output[2]:clone(), -- cloning defensively, TODO: remove
        }
        table.insert(step_trace, layer_step_trace)
    end

    -- last one is just a Linear
    local decoder_output = self.decoder:forward(current_input)
    table.insert(step_trace, {
        inputs = current_input:clone(),
        output = decoder_output:clone()
    })

    table.insert(self.trace, step_trace)
    self.output = decoder_output:clone()
    return self.output
end

-- step forward on a table of inputs representing the sequence
function ScheduledSharpeningController:forward(inputs)
    self:reset()
    local outputs = {}
    for i = 1, #inputs do
        outputs[i] = self:step(inputs[i]):clone()
    end
    self.output = outputs
    return self.output
end


-- backpropagate on a table of inputs and a table of grad_outputs
function ScheduledSharpeningController:backward(inputs, grad_outputs)
    local current_gradOutput

    -- make a set of zero gradients for the timestep after the last one
    -- allows us to use the same code for the last timestep as for the others
    -- self.backtrace[#self.trace + 1] = self:buildFinalGradient()

    for timestep = #grad_outputs, 1, -1 do
        self:backstep(self.trace[timestep].input, grad_outputs[timestep])
    end
    self.gradInput = current_gradOutput
    return self.gradInput
end


-- this should only be used after the system has been run to completion
-- at that point, it should be called in the reverse order of computation
function ScheduledSharpeningController:backstep(input, gradOutput)
    local timestep = #self.trace

    -- if this is the last timestep, and it hasn't been done already,
    -- make a set of zero gradients for the timestep after the last one.
    -- this allows us to use the same code for the last timestep as for the others
    if type(self.backtrace[timestep + 1]) == "nil" then
        self.backtrace[#self.trace + 1] = self:buildFinalGradient()
    end

    -- make sure that (with dummy in place for (#timesteps + 1)) we have gradients
    -- for the timestep after this
    assert(type(self.backtrace[timestep + 1]) ~= nil)

    local step_trace = self.trace[timestep]
    local step_backtrace = {}

    -- in Torch, a network must have been run forward() before backward()
    -- otherwise operation is not guaranteed to be correct

    -- #self.network is the number of layers
    for i = 1, #self.network do
        local layer_input = step_trace[i].inputs
        self.network[i]:forward(layer_input)
    end

    local decoder_input = step_trace[#self.network+1].inputs
    self.decoder:forward(decoder_input)

    current_gradOutput = self.decoder:backward(decoder_input, gradOutput)

    for i = #self.network, 1, -1 do
        local layer_input = step_trace[i].inputs

        -- now we'll build a table of the form
        --      { grad(next_c), grad(next_h) }
        -- that we can run backward through this layer
        local layer_grad_output = {}

        -- grad(next_c) is grad_prev_c from the next timestep
        layer_grad_output[1] = self.backtrace[timestep + 1][i][2]

        -- grad(next_h) contribution from grad_prev_h from the next timestep
        layer_grad_output[2] = self.backtrace[timestep + 1][i][3]

        -- grad(next_h) contribution from next_h as this layer's output
        layer_grad_output[2] = layer_grad_output[2] + current_gradOutput


        local gradInput = self.network[i]:backward(layer_input, layer_grad_output)
        local layer_step_backtrace = {
            -- cloning defensively, TODO: remove
            gradInput[1]:clone(), -- grad_input
            gradInput[2]:clone(), -- grad_prev_c
            gradInput[3]:clone(), -- grad_prev_h
        }

        current_gradOutput = gradInput[1]:clone() -- cloning defensively, TODO: remove
        step_backtrace[i] = layer_step_backtrace
    end
    self.backtrace[timestep] = step_backtrace

    if timestep == 1 then
        self.gradInput = current_gradOutput
    end

    -- get rid of the used trace
    self.trace[timestep] = nil

    return current_gradOutput
end

function ScheduledSharpeningController:buildFinalGradient()
    -- build a set of dummy (zero) gradients for a timestep that didn't happen
    local last_gradient = {}
    for i = 1, #self.network do
        if self.decoder.modules[1].weight:type() == "torch.CudaTensor" then
            table.insert(last_gradient, {
                    torch.zeros(opt.batch_size, self.num_units_per_layer):cuda(), -- dummy gradInput
                    torch.zeros(opt.batch_size, self.num_units_per_layer):cuda(), -- dummy grad_prev_c
                    torch.zeros(opt.batch_size, self.num_units_per_layer):cuda(), -- dummy grad_prev_h
                })
        else
            table.insert(last_gradient, {
                    torch.zeros(opt.batch_size, self.num_units_per_layer), -- dummy gradInput
                    torch.zeros(opt.batch_size, self.num_units_per_layer), -- dummy grad_prev_c
                    torch.zeros(opt.batch_size, self.num_units_per_layer), -- dummy grad_prev_h
                })
        end
    end

    return last_gradient
end

function ScheduledSharpeningController:updateParameters(learningRate)
    for i = 1, #self.network do
        self.network[i]:updateParameters(learningRate)
    end
    self.decoder:updateParameters(learningRate)
end

function ScheduledSharpeningController:zeroGradParameters()
    for i = 1, #self.network do
        self.network[i]:zeroGradParameters()
    end
    self.decoder:zeroGradParameters()
end

-- taken from nn.Container
function ScheduledSharpeningController:parameters()
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
    for i=1,#self.network do
        local mw,mgw = self.network[i]:parameters()
        if mw then
            tinsert(w,mw)
            tinsert(gw,mgw)
        end
    end
    local mw,mgw = self.decoder:parameters()
    if mw then
        tinsert(w,mw)
        tinsert(gw,mgw)
    end
    return w,gw
end

function ScheduledSharpeningController:training()
    for i = 1, #self.network do
        self.network[i]:training()
    end
    self.decoder:training()
end

function ScheduledSharpeningController:evaluate()
    for i = 1, #self.network do
        self.network[i]:evaluate()
    end
    self.decoder:evaluate()
end

function ScheduledSharpeningController:cuda()
    for i = 1, #self.network do
        self.network[i]:cuda()
    end
    self.decoder:cuda()
end

function ScheduledSharpeningController:float()
    for i = 1, #self.network do
        self.network[i]:float()
    end
    self.decoder:float()
end
