require 'nn'
require 'gnuplot'
require 'optim'
require 'nngraph'

require 'tools'
require 'vis'

require 'utils'
require 'OneHot'
local CharSplitLMMinibatchLoader = require 'CharSplitLMMinibatchLoader'

require 'Controller'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm,gru or rnn')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
-- cmd:option('-eval_val_every',10,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
-- TODO: turn GPU back on
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

-- create the data loader class
local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
local vocab_size = loader.vocab_size  -- the number of distinct characters
local vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)


controller = nn.Controller(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
criterion = nn.CrossEntropyCriterion()
one_hot = OneHot(vocab_size)

local params, grad_params = controller:getParameters()
params:uniform(-0.08, 0.08) -- small numbers uniform

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    controller:reset()

    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end
        if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
            x = x:cl()
            y = y:cl()
        end
        -- forward pass
        for t=1,opt.seq_length do
            local input = one_hot:forward(x[{{}, t}])

            local prediction = controller:step(input)
            loss = loss + criterion:forward(prediction, y[{{}, t}])
        end
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
    end
    ------------------- forward pass -------------------
    local predictions = {}           -- softmax outputs
    local grad_outputs = {}
    local loss = 0

    controller:training() -- make sure we are in correct mode (this is cheap, sets flag)
    for t=1,opt.seq_length do
        local input = one_hot:forward(x[{{}, t}])

        predictions[t] = controller:step(input)
        loss = loss + criterion:forward(predictions[t], y[{{}, t}])

        grad_outputs[t] = criterion:backward(predictions[t], y[{{}, t}])

        -- print("pred:", predictions[t][1])
        -- print("truth:", y[{{}, t}][1])
        -- vis.diff(predictions[t][1], y[{{}, t}][1])
        -- print(grad_outputs[t][1])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------

    controller:backward(x, grad_outputs)
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- [[
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil

for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    controller:reset()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.controller = controller
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
        os.execute("say 'Checkpoint saved.'")
        os.execute(string.format("say 'Epoch %.2f'", epoch))
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then
        loss0 = loss[1]
    end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        print("loss0:", loss0, "loss[1]:", loss[1])
        break -- halt
    end
end




--[[
for i = 1, STEPS do
    -- input1 = math.random(1, 5)
    -- input2 = math.random(1, 5)
    --
    -- inputTensor = torch.zeros(LC.DIM_DATA)
    -- inputTensor[input1] = 1
    -- inputTensor[5 + input2] = 1
    --
    -- target = input1 + input2
    -- targetTensor = torch.zeros(LC.DIM_DATA)
    -- targetTensor[target] = 1

    -- print("layer 1 gate weights:", vn.layers[1].container.modules[3]:getParameters():sum())



    -- inputTensor = torch.rand(10)
    -- targetTensor = inputTensor:clone()

    local inputTensor, targetTensor = problems.sort()

    output = vn:forward(inputTensor)
    score = vn:backward(inputTensor, targetTensor)
    print("Score "..i..":", score)
    scores[i] = score
    -- local params, gradParams = vn:getParameters()
    -- print(params)
    -- print(gradParams)
    -- print("Sum of gradParams:", gradParams:sum())

    -- print(vn.layers[1].container.modules[1])
    -- local p, gp = vn.layers[1].container.modules[1]:getParameters()
    -- print("layer 1 data gP:", gp:sum())

    if i % BATCH_SIZE == 0 then
        print("---------------------------------------------------------------")
        print("---------------------------------------------------------------")
        print("---------------------------------------------------------------")
        -- print(vn.layerSequence)
        -- print(output[{{90, 100}}])
        -- print(targetTensor[{{90, 100}}])

        gradJs = torch.Tensor(vn.gradJs)
        print("batch average gradJ:", gradJs:mean())
        print("last 20 avg gradJ:", gradJs[{{gradJs:size(1)-19, gradJs:size(1)}}]:mean())

        vn:updateProbabilisticParameters(LEARNING_RATE * 5 * (1 + i / 20000)^-1)
        -- vn:updateProbabilisticParameters(LEARNING_RATE)
        vn:zeroProbabilisticGradParameters()
    end

    if i % 5000 == 0 then
        print(vn.layerSequence)
        print(vis.prettyError(output - targetTensor))
        -- print(output)
        -- print(targetTensor)
    end

    local prob = vn.layerSequence[#vn.layerSequence].prob
    probAvg = .999 * probAvg + .001 * prob

    -- if prob < probAvg / 5 then
    --     print("Unlikely!")
    --     print(vn.layerSequence)
    -- end
    --
    -- if probAvg > 0.999999 then
    --     break
    -- end

    vn:updateDataParameters(LEARNING_RATE / 100 * (1 + i / 20000)^-1)
    vn:zeroDataGradParameters()
end

gnuplot.plot(scores, '-')


--
-- avgScore = 0
-- for i = 1, 100 do
--     input1 = math.random(0, 4)
--     input2 = math.random(0, 4)
--
--     inputTensor = torch.zeros(10)
--     inputTensor[1] = input1
--     inputTensor[2] = input2
--
--     target = input1 + input2
--     targetTensor = torch.Tensor({target})
--     -- targetTensor = torch.zeros(10)
--     -- targetTensor[target + 1] = 1
--
--
--     currentLayer = 1
--     currentInput = inputTensor:clone()
--     while true do
--         layer = layers[currentLayer]
--         currentOutput = layer:forward(currentInput):clone()
--
--         shouldReturn = torch.bernoulli(currentOutput[RETURN_GATE]) == 1
--         currentInput = currentOutput[{{DATA_START, DATA_END}}]:clone()
--
--         shouldJump = torch.bernoulli(currentOutput[JUMP_GATE]) == 1
--         if shouldJump then
--             rawAddress = currentOutput[{{ADDRESS_START, ADDRESS_END}}]
--             addressDistribution = softmax:forward(rawAddress)
--             currentLayer = distributions.cat.rnd(addressDistribution)[1]
--         else
--             currentLayer = currentLayer + 1
--         end
--
--         if shouldReturn or currentLayer > DIM_ADDRESS then
--             break
--         end
--     end
--     resultTensor = currentInput:clone()
--     result = outputLayer:forward(resultTensor)
--
--     score = criterion:forward(result, targetTensor)
--
--
--     snet:zeroGradParameters()
--     snet:backward(sinputTensor, scriterion:backward(snet.output, starget))
--     snet:updateParameters(0.01)
-- end
--]]
