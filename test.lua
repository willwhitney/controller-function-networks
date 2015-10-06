require 'optim'
require 'Controller'
require 'OneHot'

torch.manualSeed(0)

opt = {
        vocab_size = 2,
        batch_size = 1,
        seq_length = 1,
    }

one_hot = OneHot(opt.vocab_size)
controller = nn.Controller(opt.vocab_size, 3, 1, 0)
criterion = nn.CrossEntropyCriterion()

p, gp = controller:getParameters()
controller:training()

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

function feval()
    controller:reset()
    grad_outputs = {}
    gp:zero()

    for t = 1, opt.seq_length do
        raw_input = torch.Tensor({math.random(2)})
        input = one_hot:forward(raw_input)
        target = raw_input:clone()

        prediction = controller:step(input)
        loss = criterion:forward(prediction, target)

        grad_outputs[t] = criterion:backward(prediction, target)
    end

    -- for t = opt.seq_length,1,-1 do
    --     controller:backstep(t, grad_outputs[t])
    -- end

    controller:backward(nil, grad_outputs)

    gp:clamp(-5, 5)
    return loss, gp
end

optim_state = {learningRate = 2e-3, alpha = 0.95}

for i = 1, 10000 do
    _, loss = optim.rmsprop(feval, p, optim_state)
    print(loss[1])
end
