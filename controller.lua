require 'nn'
require 'nngraph'

LSTM = require 'LSTM.lua'

Controller, parent = torch.class('nn.Controller')


function Controller:__init(input_size, num_units_per_layer, num_layers)
    self.input_size = input_size
    self.num_units_per_layer = num_units_per_layer

    -- self.state = {}
    self.trace = {}
    self.backtrace = {}

    -- create a first state with all previous outputs & cells set to zeros
    self.state = {}
    for i = 1, num_layers do
        local layer_state = {
                torch.zeros(self.num_units_per_layer),  -- prev_c
                torch.zeros(self.num_units_per_layer),  -- prev_h
            }
        table.insert(first_state, layer_state)
    end
    table.insert(self.state, first_state)

    self.network = {}
    for i = 1, num_layers do
        table.insert(network,
            LSTM.create(self.input_size, self.num_units_per_layer))
    end

end


function Controller:step(input)
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

    table.insert(self.trace, step_trace)
    self.output = output
end


function Controller:backward(inputs, gradOutputs)
    local current_gradOutput

    -- make a set of zero gradients for the timestep after the last one
    -- allows us to use the same code for the last timestep as for the others
    self.backtrace[#self.trace + 1] = self:buildFinalGradient()

    for timestep = #self.trace, 1, -1 do
        local step_trace = self.trace[timestep]
        local step_backtrace = {}

        -- in Torch, a network must have been run forward() before backward()
        -- otherwise operation is not guaranteed to be correct

        -- #self.network is the number of layers
        for i = 1, #self.network do
            local layer_input = step_trace[i].input
            self.network[i]:forward(layer_input)
        end

        for i = #self.network, 1, -1 do

            -- for most layers, current_gradOutput was set by the layer above.
            -- however, for the last layer in the network, current_gradOutput comes
            -- from the outside. the output of the last layer goes to something
            -- else (whether that's a criterion or some differentiable function
            -- that does something else) so that outside thing must provide the
            -- gradient
            if i == #self.network then
                current_gradOutput = gradOutputs[timestep]
            end


            local layer_input = step_trace[i].input

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
                gradInput[1]:clone(), -- cloning defensively, TODO: remove
                gradInput[2]:clone(), -- cloning defensively, TODO: remove
                gradInput[3]:clone(), -- cloning defensively, TODO: remove
            }

            current_gradOutput = gradInput[1]:clone() -- cloning defensively, TODO: remove
            step_backtrace[i] = layer_step_backtrace
        end
        self.backtrace[timestep] = step_backtrace
    end
    self.gradInput = current_gradOutput
    return self.gradInput
end


-- this should only be used after the system has been run to completion
-- at that point, it should be called in the reverse order of computation
function Controller:backstep(timestep, gradOutput)
    -- make sure we have a trace at this timestep
    assert(type(self.trace[timestep]) ~= nil)

    -- if this is the last timestep, and it hasn't been done already,
    -- make a set of zero gradients for the timestep after the last one.
    -- this allows us to use the same code for the last timestep as for the others
    if timestep == #self.trace and type(self.backtrace[timestep + 1] == nil) then
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
        local layer_input = step_trace[i].input
        self.network[i]:forward(layer_input)
    end

    for i = #self.network, 1, -1 do

        -- for most layers, current_gradOutput was set by the layer above.
        -- however, for the last layer in the network, current_gradOutput comes
        -- from the outside. the output of the last layer goes to something
        -- else (whether that's a criterion or some differentiable function
        -- that does something else) so that outside thing must provide the
        -- gradient
        if i == #self.network then
            current_gradOutput = gradOutput
        end


        local layer_input = step_trace[i].input

        -- now we'll build a table of the form
        --      { grad(next_c), grad(next_h) }
        -- that we can run backward through this layer
        local layer_grad_output = {}

        -- grad(next_c) is grad_prev_c from the next timestep
        layer_grad_output[1] = self.backtrace[timestep + 1][i][2]

        -- grad(next_h) contribution from grad_prev_h from the next timestep
        layer_grad_output[2] = self.backtrace[timestep + 1][i][3]

        -- grad(next_h) contribution from next_h being used as this layer's output
        layer_grad_output[2] = layer_grad_output[2] + current_gradOutput


        local gradInput = self.network[i]:backward(layer_input, layer_grad_output)
        local layer_step_backtrace = {
            gradInput[1]:clone(), -- cloning defensively, TODO: remove
            gradInput[2]:clone(), -- cloning defensively, TODO: remove
            gradInput[3]:clone(), -- cloning defensively, TODO: remove
        }

        current_gradOutput = gradInput[1]:clone() -- cloning defensively, TODO: remove
        step_backtrace[i] = layer_step_backtrace
    end
    self.backtrace[timestep] = step_backtrace

    if timestep == 1 then
        self.gradInput = current_gradOutput
    end

    return current_gradOutput
end

function Controller:buildFinalGradient()
    -- build a set of dummy (zero) gradients for a timestep that didn't happen
    local last_gradient = {}
    for i = 1, #self.network do
        table.insert(last_gradient, {
                torch.zeros(self.num_units_per_layer), -- dummy gradInput
                torch.zeros(self.num_units_per_layer), -- dummy grad_prev_c
                torch.zeros(self.num_units_per_layer), -- dummy grad_prev_h
            })
    end
    return last_gradient
end

function Controller:updateParameters(learningRate)
    for i = 1, #self.network do
        self.network[i]:updateParameters(learningRate)
    end
end

function Controller:zeroGradParameters()
    for i = 1, #self.network do
        self.network[i]:zeroGradParameters()
    end
end

--
-- -- 3-layer LSTM network (input and output have 3 dimensions)
-- network = {LSTM.create(3, 4), LSTM.create(4, 4), LSTM.create(4, 3)}
--
-- -- network input
-- local x = torch.randn(1, 3)
-- local previous_state = {
--     {torch.zeros(1, 4), torch.zeros(1,4)},
--     {torch.zeros(1, 4), torch.zeros(1,4)},
--     {torch.zeros(1, 3), torch.zeros(1,3)}
-- }
--
-- -- network output
-- output = nil
-- next_state = {}
--
-- -- forward pass
-- local layer_input = {x, table.unpack(previous_state[1])}
-- for l = 1, #network do
--     -- forward the input
--     local layer_output = network[l]:forward(layer_input)
--     -- save output state for next iteration
--     table.insert(next_state, layer_output)
--     -- extract hidden state from output
--     local layer_h = layer_output[2]
--     -- prepare next layer's input or set the output
--     if l < #network then
--         layer_input = {layer_h, table.unpack(previous_state[l + 1])}
--     else
--         output = layer_h
--     end
-- end
--
-- print(next_state)
-- print(output)
