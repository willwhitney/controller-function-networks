require 'optim'
-- require 'Controller_c_only'
require 'Controller_memoryless'
-- require 'Controller'
require 'OneHot'
require 'vis'

torch.manualSeed(0)

opt = {
        vocab_size = 2,
        batch_size = 1,
        seq_length = 2,
        rnn_size = 1,
    }

one_hot = OneHot(opt.vocab_size)
controller = nn.Controller(opt.vocab_size, opt.rnn_size, 1, 0)
criterion = nn.CrossEntropyCriterion()

p, gp = controller:getParameters()
controller:training()

--[[
-- for i = 1, 10000 do
--     grad_outputs = {}
--     controller:reset()
--
--     for t = 1,opt.seq_length do
--         raw_input = torch.Tensor({math.random(2)})
--         input = one_hot:forward(raw_input)
--         target = raw_input:clone()
--
--         prediction = controller:step(input)
--         loss = criterion:forward(prediction, target)
--         print(loss)
--
--         grad_outputs[t] = criterion:backward(prediction, target)
--     end
--
--     -- for t = opt.seq_length,1,-1 do
--     --     controller:backstep(t, grad_outputs[t])
--     -- end
--     controller:backward(nil, grad_outputs)
--
--     -- gp:clamp(-5, 5)
--     controller:updateParameters(2e-3)
--     controller:zeroGradParameters()
-- end

-- function loss(output)
--   return output:sum()
-- end
--
-- function gradLoss(output)
--   return torch.ones(output:size())
-- end
--
-- function finiteDiff()
--   epsilon = 1e-3
--
--   for batchIndex = 1, batch_size do
--     for i = 1, gradTemplate:size()[2] do
--       tempTemplate = template:clone()
--
--       tempTemplate[batchIndex][i] = template[batchIndex][i] - epsilon
--       outputNeg = acr:forward({tempTemplate, iGeoPose})
--       lossNeg = loss(outputNeg)
--
--       tempTemplate[batchIndex][i] = template[batchIndex][i] + epsilon
--       outputPos = acr:forward({tempTemplate, iGeoPose})
--       lossPos = loss(outputPos)
--
--       finiteDiffGrad = (lossPos - lossNeg) / (2 * epsilon)
--       gradTemplate[batchIndex][i] = finiteDiffGrad
--     end
--   end
--
--   gradGeoPose = torch.zeros(iGeoPose:size())
--   for batchIndex = 1, batch_size do
--     for i = 1, gradGeoPose:size()[2] do
--       tempGeoPose = iGeoPose:clone()
--
--       tempGeoPose[batchIndex][i] = iGeoPose[batchIndex][i] - epsilon
--       outputNeg = acr:forward({template, tempGeoPose})
--       lossNeg = loss(outputNeg)
--
--       tempGeoPose[batchIndex][i] = iGeoPose[batchIndex][i] + epsilon
--       outputPos = acr:forward({template, tempGeoPose})
--       lossPos = loss(outputPos)
--
--       finiteDiffGrad = (lossPos - lossNeg) / (2 * epsilon)
--       gradGeoPose[batchIndex][i] = finiteDiffGrad
--     end
--   end
--
--   return {gradTemplate, gradGeoPose}
-- end
--]]

function reset()
    controller:reset()
    predictions = {}
    grad_outputs = {}
    gp:zero()
    loss = 0
end

function get_example()
    input_value = math.random(opt.vocab_size)
    raw_input = torch.Tensor{input_value}
    input = one_hot:forward(raw_input)

    target_value = (input_value + 3) % opt.vocab_size + 1
    target = torch.Tensor{target_value}
    return input, target
end

alternation_value = 2
function get_alternation_example()
    if alternation_value == 2 then
        alternation_value = 1
    else
        alternation_value = 2
    end
    return torch.Tensor{{1, 0}}, torch.Tensor{alternation_value}
end

function feval()
    reset()
    for t = 1, opt.seq_length do
        input, target = get_example()

        predictions[t] = controller:step(input)
        loss = loss + criterion:forward(predictions[t], target)
        -- print(vis.simplestr(target), vis.simplestr(prediction))

        grad_outputs[t] = criterion:backward(predictions[t], target):clone()
    end

    -- for t = opt.seq_length,1,-1 do
    --     controller:backstep(t, grad_outputs[t])
    -- end

    controller:backward(nil, grad_outputs)

    gp:clamp(-5, 5)
    return loss, gp
end

function feval_twice()
    reset()
    for t = 1, opt.seq_length do
        input, target = get_alternation_example()

        predictions[t] = controller:step(input)
        loss = loss + criterion:forward(predictions[t], target)
        -- print(vis.simplestr(target), vis.simplestr(prediction))

        grad_outputs[t] = criterion:backward(predictions[t], target)
    end
    controller:backward(nil, grad_outputs)

    controller:reset()
    predictions = {}
    grad_outputs = {}

    for t = 1, opt.seq_length do
        input, target = get_alternation_example()

        predictions[t] = controller:step(input)
        loss = loss + criterion:forward(predictions[t], target)
        -- print(vis.simplestr(target), vis.simplestr(prediction))

        grad_outputs[t] = criterion:backward(predictions[t], target)
    end
    controller:backward(nil, grad_outputs)

    gp:clamp(-5, 5)
    return loss, gp
end

optim_state = {learningRate = 2e-3, alpha = 0.95}
reset()

train = true
if train then
    for i = 1, 10000 do
        if opt.seq_length == 1 then
            _, loss = optim.rmsprop(feval_twice, p, optim_state)
        else
            _, loss = optim.rmsprop(feval, p, optim_state)
        end
        print(loss[1])
        -- print("predictions:")
        -- for pred = 1, #predictions do
        --     print(vis.simplestr(predictions[pred]))
        -- end
        -- print("grad_outputs:")
        -- for g_o = 1, #grad_outputs do
        --     print(vis.simplestr(grad_outputs[g_o]))
        -- end
        -- print("params:", p:norm())
        -- print("grad_params:", gp:norm())
    end
else
    for t = 1, 2 do
        input, target = get_alternation_example()

        predictions[t] = controller:step(input)
        loss = loss + criterion:forward(predictions[t], target)
        grad_outputs[t] = criterion:backward(predictions[t], target)
    end
end

-- print("predictions:", vis.simplestr(predictions[1]), vis.simplestr(predictions[2]))
-- print("grad_outputs:", vis.simplestr(grad_outputs[1]), vis.simplestr(grad_outputs[2]))
