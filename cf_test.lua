require 'optim'
require 'CFNetwork'
require 'OneHot'
require 'vis'

torch.manualSeed(0)

opt = {
        vocab_size = 20,
        batch_size = 1,
        seq_length = 2,
        rnn_size = 1,
    }

one_hot = OneHot(opt.vocab_size)

model = nn.CFNetwork({
        input_dimension = opt.vocab_size,
        num_functions = opt.vocab_size,
        controller_units_per_layer = opt.rnn_size,
        controller_num_layers = 1,
        controller_dropout = 0,
    })

criterion = nn.CrossEntropyCriterion()

p, gp = model:getParameters()
model:training()


function reset()
    reset_alternation()
    model:reset()
    inputs = {}
    predictions = {}
    grad_outputs = {}
    gp:zero()
    loss = 0
end

function get_example()
    input_value = math.random(opt.vocab_size)
    raw_input = torch.Tensor{input_value}
    input = one_hot:forward(raw_input)

    -- target_value = (input_value + 3) % opt.vocab_size + 1
    target_value = input_value
    target = torch.Tensor{target_value}
    return input, target
end

function get_fib_batch()
    

end

-- function reset_alternation()
--     alternation_value = 1
-- end
--
-- function get_alternation_example()
--     if alternation_value == 2 then
--         alternation_value = 1
--     else
--         alternation_value = 2
--     end
--     return torch.Tensor{{1, 0}}, torch.Tensor{alternation_value}
-- end

function feval()
    reset()
    for t = 1, opt.seq_length do
        -- input, target = get_example()
        input, target = get_example()
        inputs[t] = input

        -- print("input:", vis.simplestr(input[1]))
        -- print("target:", vis.simplestr(target))
        predictions[t] = model:step(input):clone()
        loss = loss + criterion:forward(predictions[t], target)

        grad_outputs[t] = criterion:backward(predictions[t], target):clone()
        -- print("prediction:", vis.simplestr(predictions[t][1]))
        -- print("gradOutput:", vis.simplestr(grad_outputs[t][1]))
        -- print("")
    end

    model:backward(inputs, grad_outputs)

    gp:clamp(-5, 5)
    return loss, gp
end

optim_state = {learningRate = 2e-3, alpha = 0.95}
reset()

train = true
if train then
    for i = 1, 10000 do
        -- if opt.seq_length == 1 then
        --     _, loss = optim.rmsprop(feval_twice, p, optim_state)
        -- else
        _, loss = optim.rmsprop(feval, p, optim_state)
        -- end
        print(loss[1])
        -- print("predictions:")
        -- for pred = 1, #predictions do
        --     print(vis.simplestr(predictions[pred][1]))
        -- end
        -- print("grad_outputs:")
        -- for g_o = 1, #grad_outputs do
        --     print(vis.simplestr(grad_outputs[g_o][1]))
        -- end
        print("\n")
        -- print("params:", p:norm())
        -- print("grad_params:", gp:norm())
    end
else
    error("don't do this")
    -- for t = 1, 2 do
    --     input, target = get_alternation_example()
    --
    --     predictions[t] = model:step(input):clone()
    --     loss = loss + criterion:forward(predictions[t], target)
    --     grad_outputs[t] = criterion:backward(predictions[t], target):clone()
    -- end
end

-- print("predictions:", vis.simplestr(predictions[1]), vis.simplestr(predictions[2]))
-- print("grad_outputs:", vis.simplestr(grad_outputs[1]), vis.simplestr(grad_outputs[2]))
